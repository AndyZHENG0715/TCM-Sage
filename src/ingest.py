"""
TCM-Sage Complete RAG Pipeline Script

This script implements the complete data processing pipeline for the TCM-Sage RAG system:
1. Reads the raw Huangdi Neijing text file
2. Cleans it by removing unwanted sections (modern translations, table of contents)
3. Splits it into semantically meaningful chunks
4. Generates vector embeddings using sentence transformers
5. Stores the embeddings in a ChromaDB vector store

This creates a searchable knowledge base ready for retrieval-augmented generation.
"""

import pathlib
import re
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma


def clean_and_prepare_text(raw_text):
    """
    Clean and prepare the raw text by removing modern translations, 
    table of contents, and formatting issues.
    
    Args:
        raw_text (str): The raw text content from the source file
        
    Returns:
        str: The cleaned and prepared text
    """
    
    # Step 1: Remove modern translations
    # Remove all sections that start with a chapter title followed by "参考译文"
    # and end before the next chapter title
    translation_pattern = r'([^。！？]*篇第[^。！？]*参考译文.*?)(?=[^。！？]*篇第[^。！？]*[^参考译文]|$)'
    cleaned_text = re.sub(translation_pattern, '', raw_text, flags=re.DOTALL)
    
    # Step 2: Remove table of contents
    # Find the first actual chapter heading and discard everything before it
    first_chapter_pattern = r'^.*?(?=上古天真论篇第一)'
    cleaned_text = re.sub(first_chapter_pattern, '', cleaned_text, flags=re.DOTALL)
    
    # Step 3: Fix formatting issues
    
    # Remove page number markers (e.g., ---------------------------002)
    page_marker_pattern = r'-{20,}\d+'
    cleaned_text = re.sub(page_marker_pattern, '', cleaned_text)
    
    # Remove excessive blank lines (more than one consecutive blank line)
    excessive_blanks_pattern = r'\n\s*\n\s*\n+'
    cleaned_text = re.sub(excessive_blanks_pattern, '\n\n', cleaned_text)
    
    # Remove leading/trailing whitespace from each line
    lines = cleaned_text.split('\n')
    cleaned_lines = [line.strip() for line in lines]
    
    # Remove empty lines at the beginning and end
    while cleaned_lines and not cleaned_lines[0]:
        cleaned_lines.pop(0)
    while cleaned_lines and not cleaned_lines[-1]:
        cleaned_lines.pop()
    
    # Join the lines back together
    cleaned_text = '\n'.join(cleaned_lines)
    
    # Additional cleanup: Remove any remaining unwanted patterns
    # Remove any remaining reference to "参考译文" lines
    cleaned_text = re.sub(r'.*参考译文.*\n?', '', cleaned_text)
    
    # Remove any standalone numbers that might be page references
    cleaned_text = re.sub(r'^\d+$', '', cleaned_text, flags=re.MULTILINE)
    
    # Clean up any remaining excessive whitespace
    cleaned_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_text)
    cleaned_text = re.sub(r'[ \t]+', ' ', cleaned_text)  # Replace multiple spaces/tabs with single space
    
    return cleaned_text


def main():
    """
    Main function to process the source text file, clean it, chunk it, and save the results.
    """
    
    # Define file paths using pathlib.Path
    script_dir = pathlib.Path(__file__).parent
    source_file_path = script_dir.parent / "data" / "source" / "huangdi_neijing.txt"
    cleaned_file_path = script_dir.parent / "data" / "cleaned" / "cleaned_huangdi_neijing.txt"
    chunks_file_path = script_dir.parent / "data" / "processed" / "chunks.json"
    
    # Ensure the directories exist
    cleaned_file_path.parent.mkdir(parents=True, exist_ok=True)
    chunks_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Read the content from the source file
        print("📖 Reading source text file...")
        with open(source_file_path, 'r', encoding='utf-8') as file:
            raw_content = file.read()
        
        print(f"📊 Original file size: {len(raw_content)} characters")
        
        # Clean and prepare the text
        print("🧹 Cleaning and preparing text...")
        cleaned_content = clean_and_prepare_text(raw_content)
        
        print(f"📊 Cleaned file size: {len(cleaned_content)} characters")
        print(f"📉 Text reduction: {len(raw_content) - len(cleaned_content)} characters removed")
        
        # Write the cleaned text to the cleaned output file
        with open(cleaned_file_path, 'w', encoding='utf-8') as file:
            file.write(cleaned_content)
        
        # Instantiate a Text Splitter for chunking
        print("✂️ Creating text splitter...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,      # Each chunk will be around 500 characters
            chunk_overlap=50     # Each chunk will overlap with the previous one by 50 characters
        )
        
        # Split the document into chunks
        print("🔄 Splitting text into chunks...")
        document_chunks = text_splitter.split_text(cleaned_content)
        
        # Prepare chunks for storage as dictionaries
        print("📝 Preparing chunks for storage...")
        chunk_dicts = []
        for i, chunk in enumerate(document_chunks, 1):
            chunk_dict = {
                "id": f"chunk_{i}",
                "content": chunk.strip()
            }
            chunk_dicts.append(chunk_dict)
        
        # Save the chunks to JSON file
        print("💾 Saving chunks to JSON file...")
        with open(chunks_file_path, 'w', encoding='utf-8') as file:
            json.dump(chunk_dicts, file, ensure_ascii=False, indent=4)
        
        # Load the chunks for embedding generation
        print("📖 Loading chunks for embedding generation...")
        with open(chunks_file_path, 'r', encoding='utf-8') as file:
            loaded_chunks = json.load(file)
        
        # Instantiate embedding model
        print("🤖 Initializing embedding model...")
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Prepare documents for ChromaDB
        print("📝 Preparing documents for vector store...")
        chunk_contents = [chunk['content'] for chunk in loaded_chunks]
        chunk_ids = [{"id": chunk['id']} for chunk in loaded_chunks]
        
        # Create the vector store
        print("🗄️ Creating ChromaDB vector store...")
        vectorstore_path = script_dir.parent / "vectorstore" / "chroma"
        vectorstore_path.mkdir(parents=True, exist_ok=True)
        
        vectorstore = Chroma.from_texts(
            texts=chunk_contents,
            embedding=embeddings,
            ids=[d['id'] for d in chunk_ids],
            persist_directory=str(vectorstore_path)
        )
        
        # Print success message
        print("✅ Successfully cleaned, chunked, embedded, and saved to vector store!")
        print(f"📁 Source file: {source_file_path}")
        print(f"📁 Cleaned file: {cleaned_file_path}")
        print(f"📁 Chunks file: {chunks_file_path}")
        print(f"📁 Vector store: {vectorstore_path}")
        print(f"📊 Final cleaned file size: {len(cleaned_content)} characters")
        print(f"📊 Total number of chunks created: {len(chunk_dicts)}")
        print(f"📊 Vector store entries: {len(chunk_contents)}")
        
        # Show statistics about chunk sizes
        chunk_sizes = [len(chunk['content']) for chunk in chunk_dicts]
        avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes)
        print(f"📊 Average chunk size: {avg_chunk_size:.1f} characters")
        print(f"📊 Chunk size range: {min(chunk_sizes)} - {max(chunk_sizes)} characters")
        
        # Show a preview of the first chunk
        print("\n" + "="*60)
        print("📖 Preview of first chunk:")
        print("="*60)
        if chunk_dicts:
            first_chunk = chunk_dicts[0]
            print(f"ID: {first_chunk['id']}")
            print(f"Content: {first_chunk['content'][:200]}...")
            print(f"Length: {len(first_chunk['content'])} characters")
        print("="*60)
        
    except FileNotFoundError:
        print("❌ Error: Could not find the source text file!")
        print(f"Expected file location: {source_file_path}")
        print("Please ensure the file exists and the path is correct.")
        
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()