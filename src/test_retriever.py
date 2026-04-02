"""
TCM-Sage Vector Store Retriever Test Script

This script tests the ChromaDB vector store by loading the persistent database,
performing a similarity search with a sample query, and displaying the most
relevant text chunks from the Huangdi Neijing.
"""

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from pathlib import Path


def main():
    """
    Main function to test the ChromaDB vector store retrieval functionality.
    """
    
    # Define paths
    vectorstore_path = "vectorstore/chroma"
    
    print("🔍 Testing ChromaDB Vector Store Retrieval")
    print("=" * 60)
    
    try:
        # Load the vector store
        print("📂 Loading vector store from disk...")
        embeddings = HuggingFaceEmbeddings(
            model_name="nomic-ai/nomic-embed-text-v1.5",
            model_kwargs={'trust_remote_code': True}
        )
        vectorstore = Chroma(
            persist_directory=str(vectorstore_path),
            embedding_function=embeddings
        )
        print(f"✅ Vector store loaded successfully from: {vectorstore_path}")
        
        # Define a test query
        query = "阴阳者，天地之道也"
        print(f"\n🔍 Test Query: {query}")
        print("=" * 60)
        
        # Perform similarity search
        print("🔎 Performing similarity search...")
        results = vectorstore.similarity_search(query, k=3)
        print(f"✅ Found {len(results)} relevant chunks")
        
        # Print the results
        print("\n📖 Most Relevant Results:")
        print("=" * 60)
        
        for i, result in enumerate(results, 1):
            print(f"\n--- Result {i} ---")
            
            # Display metadata fields
            if hasattr(result, 'metadata') and result.metadata:
                meta = result.metadata
                book = meta.get('book', 'Unknown')
                chapter = meta.get('source', 'Unknown')
                chunk_idx = meta.get('chunk_index', 'N/A')
                print(f"Book: {book}")
                print(f"Chapter: {chapter}")
                print(f"Chunk Index: {chunk_idx}")
            else:
                print("Metadata: Not available")
            
            print(f"Content: {result.page_content}")
            print("-" * 40)
        
        # Additional test: Test with different query
        print(f"\n🔍 Additional Test Query: 黄帝问什么？")
        print("=" * 60)
        
        query2 = "黄帝问什么？"
        results2 = vectorstore.similarity_search(query2, k=2)
        
        for i, result in enumerate(results2, 1):
            print(f"\n--- Additional Result {i} ---")
            
            # Display metadata fields
            if hasattr(result, 'metadata') and result.metadata:
                meta = result.metadata
                book = meta.get('book', 'Unknown')
                chunk_idx = meta.get('chunk_index', 'N/A')
                print(f"Book: {book} | Chunk Index: {chunk_idx}")
            
            print(f"Content: {result.page_content[:200]}...")  # Show first 200 chars
            print("-" * 40)
        
        print("\n✅ Vector store retrieval test completed successfully!")
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find the vector store directory!")
        print(f"Expected location: {vectorstore_path}")
        print("Please ensure the vector store was created successfully by running src/ingest.py")
        
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        print("Please check that the vector store was created correctly.")


if __name__ == "__main__":
    main()
