"""
TCM-Sage Multi-Source Ingestion Pipeline

This script implements the complete data processing pipeline for the TCM-Sage RAG system:
1. Reads all .txt source files from data/source/
2. Extracts book name from filename
3. Splits into chapters with character offset tracking
4. Generates vector embeddings using sentence transformers
5. Stores embeddings in ChromaDB vector store

Supports provenance tracking with book, chapter, chunk_index, char_start, char_end metadata.
"""

import pathlib
import re
import json
from typing import List, Dict, Tuple, Optional
from langchain_chroma import Chroma

from embeddings import get_embedding_model


class SentenceAwareChineseTextSplitter:
    """
    Text splitter that respects Chinese sentence boundaries.
    
    Splits at sentence endings (。；！？) before applying character limits,
    ensuring chunks contain complete sentences rather than cut-off text.
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Chinese sentence terminators
        self.sentence_endings = re.compile(r'([。；！？\n]+)')
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks respecting sentence boundaries."""
        # First split by sentence endings, keeping the delimiters
        parts = self.sentence_endings.split(text)
        
        # Recombine sentences with their endings
        sentences = []
        i = 0
        while i < len(parts):
            sentence = parts[i]
            # Check if next part is a delimiter
            if i + 1 < len(parts) and self.sentence_endings.match(parts[i + 1]):
                sentence += parts[i + 1]
                i += 2
            else:
                i += 1
            if sentence.strip():
                sentences.append(sentence)
        
        # Now combine sentences into chunks
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence exceeds chunk_size
            if len(current_chunk) + len(sentence) > self.chunk_size:
                # Save current chunk if it has content
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                # Start new chunk with overlap from previous
                if self.chunk_overlap > 0 and current_chunk:
                    # Take last chunk_overlap characters as overlap
                    current_chunk = current_chunk[-self.chunk_overlap:] + sentence
                else:
                    current_chunk = sentence
            else:
                current_chunk += sentence
        
        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks


def extract_book_name(filename: str) -> str:
    """
    Extract book name from filename.
    
    Examples:
        "437-黄帝内经素问.txt" -> "黄帝内经素问"
        "431-黄帝内经灵枢集注.txt" -> "黄帝内经灵枢集注"
    
    Args:
        filename: The file basename including extension
        
    Returns:
        The extracted book name without number prefix and extension
    """
    # Remove .txt extension
    name = filename.replace('.txt', '')
    # Remove numeric prefix (e.g., "437-")
    name = re.sub(r'^\d+-', '', name)
    return name


# Texts that use numbered clause structure (N．) instead of chapter-based splitting
_CLAUSE_BASED_TEXTS = {'伤寒论', '金匮要略方论'}


def split_into_clauses(content: str, book_name: str) -> List[Dict]:
    """Split texts with numbered clause structure (e.g., 伤寒论's 398条).
    
    Each clause (N．...) becomes a single chunk, preserving the natural
    semantic unit of one diagnostic statement + its formula.
    """
    import re
    
    # Find all clause boundaries: N．
    clause_pattern = re.compile(r'(?=(?:^|\n)(\d+)．)', re.MULTILINE)
    matches = list(clause_pattern.finditer(content))
    
    if len(matches) < 10:
        return []  # Not enough clauses, fall back to chapter splitting
    
    # Determine which chapter each clause belongs to
    chapter_pattern = re.compile(r'<篇名>([^\n]+)')
    chapter_matches = list(chapter_pattern.finditer(content))
    
    def get_chapter_for_pos(pos: int) -> str:
        """Find which chapter a position belongs to."""
        current_chapter = book_name
        for cm in chapter_matches:
            if cm.start() <= pos:
                current_chapter = cm.group(1).strip()
            else:
                break
        return current_chapter
    
    chunks = []
    for i, match in enumerate(matches):
        clause_num = match.group(1)
        clause_start = match.start()
        
        # End is start of next clause or end of content
        if i + 1 < len(matches):
            clause_end = matches[i + 1].start()
        else:
            clause_end = len(content)
        
        clause_text = content[clause_start:clause_end].strip()
        if not clause_text or len(clause_text) < 5:
            continue
        
        chapter = get_chapter_for_pos(clause_start)
        chapter_hash = hash(chapter) % 10000
        chunk_id = f"{book_name}_clause_{clause_num}_{chapter_hash}"
        
        # Detect formula names in the clause
        formula_match = re.search(r'([\u4e00-\u9fff]{2,6}汤|[\u4e00-\u9fff]{2,6}散|[\u4e00-\u9fff]{2,6}丸|[\u4e00-\u9fff]{2,6}饮)主之', clause_text)
        formula_name = formula_match.group(1) if formula_match else None
        # Build contextual header for better embedding quality
        # e.g., "《伤寒论》第35条 麻黄汤：" helps the embedding model
        # associate formula names with clause content
        header_parts = [f'《{book_name}》第{clause_num}条']
        if formula_name:
            header_parts.append(formula_name)
        header = ' '.join(header_parts) + '：'
        content_with_header = f'{header}\n{clause_text}'
        
        chunks.append({
            'id': chunk_id,
            'content': content_with_header,
            'metadata': {
                'book': book_name,
                'source': chapter,
                'chunk_index': int(clause_num),
                'char_start': clause_start,
                'char_end': clause_end,
                'clause_number': int(clause_num),
                'formula': formula_name or '',
            }
        })
    
    return chunks

def split_into_chapters_with_offsets(content: str) -> List[Tuple[str, str, int, int]]:
    """
    Split content into chapters while tracking character offsets.
    
    Supports multiple chapter title patterns found in TCM classical texts:
    - "<篇名>XXX" (standard format in TCM-Ancient-Books repo)
    - "篇第一", "篇第二" (Huangdi Neijing format)
    - "卷一", "卷二" (volume-based format)
    - "XXX第一" (inline chapter titles like 灵枢 lfglib format)
    - Sections separated by multiple newlines
    
    Args:
        content: The full text content
        
    Returns:
        List of tuples: (chapter_title, chapter_content, char_start, char_end)
    """
    # Multiple pattern formats for different TCM texts (order matters: most specific first)
    patterns = [
        r'<篇名>([^\n]+)',                                    # <篇名>XXX (TCM-Ancient-Books repo)
        r'([^\n]*篇第[一二三四五六七八九十百千万]+)',              # 篇第X format
        r'(卷[一二三四五六七八九十百千万上中下]+[^\n]*)',           # 卷X format
        r'(第[一二三四五六七八九十百千万]+章[^\n]*)',             # 第X章 format
        r'([^\n]{2,15}第[一二三四五六七八九十百]+)',             # XXX第一 inline titles (灵枢)
    ]
    
    chapters = []
    
    # Try each pattern to find chapter boundaries
    for pattern in patterns:
        matches = list(re.finditer(pattern, content))
        if len(matches) >= 3:  # At least 3 chapters found
            for i, match in enumerate(matches):
                chapter_title = match.group(1).strip()
                # Skip preface/序/目录 sections for cleaner chapter splits
                if chapter_title in ('序', '目录') or chapter_title.startswith('序') and len(chapter_title) <= 3:
                    continue
                char_start = match.start()
                
                # End is start of next chapter or end of content
                if i + 1 < len(matches):
                    char_end = matches[i + 1].start()
                else:
                    char_end = len(content)
                
                chapter_content = content[char_start:char_end].strip()
                if chapter_content:  # Skip empty chapters
                    chapters.append((chapter_title, chapter_content, char_start, char_end))
            
            if chapters:
                return chapters
    
    # Fallback: treat entire content as single chapter
    return [("全文", content, 0, len(content))]


def process_single_source(
    file_path: pathlib.Path,
    book_name: str,
    text_splitter: SentenceAwareChineseTextSplitter
) -> List[Dict]:
    """
    Process a single source file with character offset tracking.
    
    Args:
        file_path: Path to the source text file
        book_name: Name of the book (extracted from filename)
        text_splitter: Configured text splitter for chunking
        
    Returns:
        List of chunk dictionaries with content and metadata
    """
    try:
        # Try different encodings
        for encoding in ['utf-8', 'gbk', 'gb2312', 'gb18030']:
            try:
                content = file_path.read_text(encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            print(f"⚠️ Could not decode {file_path.name} with any known encoding")
            return []
    except Exception as e:
        print(f"❌ Error reading {file_path.name}: {e}")
        return []
    
    print(f"📖 Processing: {book_name} ({len(content):,} characters)")
    
    # Use clause-level splitting for texts with numbered clause structure
    if book_name in _CLAUSE_BASED_TEXTS:
        clause_chunks = split_into_clauses(content, book_name)
        if clause_chunks:
            print(f"   📝 Clause-level splitting: {len(clause_chunks)} clauses")
            return clause_chunks
        print(f"   ⚠️ Clause splitting failed, falling back to chapter splitting")
    
    # Default: chapter-based splitting
    chapters = split_into_chapters_with_offsets(content)
    print(f"   📚 Found {len(chapters)} chapters")
    
    chunks = []
    chunk_counter = 0
    
    for chapter_title, chapter_content, chapter_start, chapter_end in chapters:
        # Split chapter into smaller chunks
        chapter_chunks = text_splitter.split_text(chapter_content)
        
        # Track position within chapter for offset calculation
        search_pos = 0
        
        for chunk_text in chapter_chunks:
            # Find chunk position within chapter content
            chunk_pos = chapter_content.find(chunk_text, search_pos)
            if chunk_pos == -1:
                # Fallback: use search position
                chunk_pos = search_pos
            
            # Calculate absolute character offsets
            abs_start = chapter_start + chunk_pos
            abs_end = abs_start + len(chunk_text)
            
            chunk_counter += 1
            chunks.append({
                "id": f"{book_name}_chunk_{chunk_counter}",
                "content": chunk_text.strip(),
                "metadata": {
                    "book": book_name,
                    "source": chapter_title,  # Kept for backward compatibility
                    "chunk_index": chunk_counter,  # 1-based index per book
                    "char_start": abs_start,
                    "char_end": abs_end
                }
            })
            
            # Update search position to avoid matching same text again
            search_pos = chunk_pos + len(chunk_text)
    
    return chunks


def ingest_all_sources(source_dir: pathlib.Path) -> List[Dict]:
    """
    Ingest all .txt files from the source directory.
    
    Args:
        source_dir: Path to directory containing source .txt files
        
    Returns:
        List of all chunks from all sources
    """
    # Initialize sentence-aware Chinese text splitter
    text_splitter = SentenceAwareChineseTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    
    all_chunks = []
    source_files = sorted(source_dir.glob("*.txt"))
    
    if not source_files:
        print(f"⚠️ No .txt files found in {source_dir}")
        return []
    
    print(f"📁 Found {len(source_files)} source files")
    
    for file_path in source_files:
        book_name = extract_book_name(file_path.name)
        chunks = process_single_source(file_path, book_name, text_splitter)
        all_chunks.extend(chunks)
        print(f"   ✅ {book_name}: {len(chunks)} chunks")
    
    return all_chunks


def main():
    """
    Main function to ingest all sources and build vector store.
    Supports checkpoint/resume: re-run to continue from where it stopped.
    """
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    # Define paths
    script_dir = pathlib.Path(__file__).parent
    source_dir = script_dir.parent / "data" / "source"
    chunks_file_path = script_dir.parent / "data" / "processed" / "chunks.json"
    vectorstore_path = script_dir.parent / "vectorstore" / "chroma"
    
    # Ensure directories exist
    chunks_file_path.parent.mkdir(parents=True, exist_ok=True)
    vectorstore_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("🚀 TCM-Sage Multi-Source Ingestion Pipeline")
    print("=" * 60)
    
    # Ingest all sources
    all_chunks = ingest_all_sources(source_dir)
    
    if not all_chunks:
        print("❌ No chunks generated. Check source files.")
        return
    
    print(f"\n📊 Total chunks across all sources: {len(all_chunks)}")
    
    # Save chunks to JSON
    print("\n💾 Saving chunks to JSON...")
    with open(chunks_file_path, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"   ✅ Saved to {chunks_file_path}")
    
    # Generate embeddings and store in ChromaDB with checkpoint/resume
    print("\n\U0001f916 Initializing embedding model (text-embedding-v4)...")
    embeddings = get_embedding_model()
    
    # Checkpoint file tracks which chunk IDs have been successfully embedded
    checkpoint_path = script_dir.parent / "data" / "processed" / "ingest_checkpoint.json"
    ingested_ids: set = set()
    if checkpoint_path.exists():
        import json as _json
        ingested_ids = set(_json.load(open(checkpoint_path, encoding='utf-8')))
        print(f"   \U0001f504 Resuming from checkpoint: {len(ingested_ids)} chunks already ingested")
    
    # Filter out already-ingested chunks
    remaining = [(c, i) for i, c in enumerate(all_chunks) if c['id'] not in ingested_ids]
    print(f"\n\U0001f4dd Chunks to embed: {len(remaining)} (of {len(all_chunks)} total)")
    
    if not remaining:
        print("\u2705 All chunks already ingested!")
    else:
        # Process in batches of 10 (DashScope text-embedding-v4 batch limit)
        BATCH_SIZE = 10
        total_batches = (len(remaining) + BATCH_SIZE - 1) // BATCH_SIZE
        
        vectorstore = Chroma(
            persist_directory=str(vectorstore_path),
            embedding_function=embeddings,
        )
        
        for batch_idx in range(0, len(remaining), BATCH_SIZE):
            batch = remaining[batch_idx:batch_idx + BATCH_SIZE]
            batch_num = batch_idx // BATCH_SIZE + 1
            
            batch_texts = [c['content'] for c, _ in batch]
            batch_ids = [c['id'] for c, _ in batch]
            batch_metas = [c['metadata'] for c, _ in batch]
            
            try:
                vectorstore.add_texts(
                    texts=batch_texts,
                    metadatas=batch_metas,
                    ids=batch_ids,
                )
                
                # Update checkpoint after each successful batch
                ingested_ids.update(batch_ids)
                with open(checkpoint_path, 'w', encoding='utf-8') as cp:
                    json.dump(sorted(ingested_ids), cp, ensure_ascii=False)
                
                print(f"   Batch {batch_num}/{total_batches}: +{len(batch)} chunks ({len(ingested_ids)}/{len(all_chunks)} total)")
                
            except Exception as e:
                # Save checkpoint before exiting on failure
                with open(checkpoint_path, 'w', encoding='utf-8') as cp:
                    json.dump(sorted(ingested_ids), cp, ensure_ascii=False)
                print(f"\n\u26a0\ufe0f  API error at batch {batch_num}/{total_batches}: {e}")
                print(f"\U0001f4be Checkpoint saved: {len(ingested_ids)}/{len(all_chunks)} chunks ingested")
                print(f"\n\U0001f504 To resume, run this script again. It will skip already-ingested chunks.")
                print(f"   To start fresh, delete: {checkpoint_path}")
                return
    
    # Statistics
    print("\n" + "=" * 60)
    print("✅ Ingestion Complete!")
    print("=" * 60)
    print(f"📁 Source directory: {source_dir}")
    print(f"📁 Chunks file: {chunks_file_path}")
    print(f"📁 Vector store: {vectorstore_path}")
    print(f"📊 Total chunks: {len(all_chunks)}")
    
    # Show chunk size stats
    chunk_sizes = [len(c['content']) for c in all_chunks]
    print(f"📊 Average chunk size: {sum(chunk_sizes) / len(chunk_sizes):.1f} characters")
    print(f"📊 Chunk size range: {min(chunk_sizes)} - {max(chunk_sizes)} characters")
    
    # Show sample from each book
    print("\n📖 Sample chunks by book:")
    seen_books = set()
    for chunk in all_chunks:
        book = chunk['metadata']['book']
        if book not in seen_books:
            seen_books.add(book)
            preview = chunk['content'][:100].replace('\n', ' ')
            print(f"   • {book}: \"{preview}...\"")


if __name__ == "__main__":
    main()
