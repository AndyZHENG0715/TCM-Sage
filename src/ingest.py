"""
TCM-Sage Multi-Source Ingestion Pipeline

This script implements the complete data processing pipeline for the TCM-Sage RAG system:
1. Reads all .txt source files from data/source/
2. Extracts book name from filename
3. Splits into chapters with character offset tracking
4. Generates vector embeddings using sentence transformers
5. Stores embeddings in ChromaDB vector store

Supports provenance tracking with book, chapter, char_start, char_end metadata.
"""

import pathlib
import re
import json
from typing import List, Dict, Tuple, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma


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


def split_into_chapters_with_offsets(content: str) -> List[Tuple[str, str, int, int]]:
    """
    Split content into chapters while tracking character offsets.
    
    Supports multiple chapter title patterns found in TCM classical texts:
    - "篇第一", "篇第二" (standard Huangdi Neijing format)
    - "卷一", "卷二" (volume-based format)
    - Sections separated by multiple newlines
    
    Args:
        content: The full text content
        
    Returns:
        List of tuples: (chapter_title, chapter_content, char_start, char_end)
    """
    # Multiple pattern formats for different TCM texts
    patterns = [
        r'([^\n]*篇第[一二三四五六七八九十百千万]+)',  # 篇第X format
        r'(卷[一二三四五六七八九十百千万]+[^\n]*)',      # 卷X format
        r'(第[一二三四五六七八九十百千万]+章[^\n]*)',    # 第X章 format
    ]
    
    chapters = []
    
    # Try each pattern to find chapter boundaries
    for pattern in patterns:
        matches = list(re.finditer(pattern, content))
        if len(matches) >= 3:  # At least 3 chapters found
            for i, match in enumerate(matches):
                chapter_title = match.group(1).strip()
                char_start = match.start()
                
                # End is start of next chapter or end of content
                if i + 1 < len(matches):
                    char_end = matches[i + 1].start()
                else:
                    char_end = len(content)
                
                chapter_content = content[char_start:char_end].strip()
                chapters.append((chapter_title, chapter_content, char_start, char_end))
            
            return chapters
    
    # Fallback: treat entire content as single chapter
    return [("全文", content, 0, len(content))]


def process_single_source(
    file_path: pathlib.Path,
    book_name: str,
    text_splitter: RecursiveCharacterTextSplitter
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
    
    # Split into chapters with offset tracking
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
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
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
    """
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
    
    # Generate embeddings and store in ChromaDB
    print("\n🤖 Initializing embedding model...")
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print("📝 Preparing documents for vector store...")
    chunk_contents = [chunk['content'] for chunk in all_chunks]
    chunk_ids = [chunk['id'] for chunk in all_chunks]
    chunk_metadatas = [chunk['metadata'] for chunk in all_chunks]
    
    print("🗄️ Creating ChromaDB vector store...")
    vectorstore = Chroma.from_texts(
        texts=chunk_contents,
        embedding=embeddings,
        metadatas=chunk_metadatas,
        ids=chunk_ids,
        persist_directory=str(vectorstore_path)
    )
    
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
