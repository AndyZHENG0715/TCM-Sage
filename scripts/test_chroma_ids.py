from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from pathlib import Path

def test_chroma_ids():
    vectorstore_path = Path("vectorstore/chroma")
    embeddings = HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1.5",
        model_kwargs={"trust_remote_code": True},
    )
    vectorstore = Chroma(
        persist_directory=str(vectorstore_path),
        embedding_function=embeddings,
    )
    
    results = vectorstore.similarity_search_with_score("黄帝内经", k=1)
    for doc, score in results:
        print(f"Document type: {type(doc)}")
        print(f"Document metadata keys: {doc.metadata.keys()}")
        print(f"Document ID attribute: {getattr(doc, 'id', 'N/A')}")
        print(f"Document ID in metadata: {doc.metadata.get('id', 'N/A')}")
        print(f"Document page_content length: {len(doc.page_content)}")

if __name__ == "__main__":
    test_chroma_ids()
