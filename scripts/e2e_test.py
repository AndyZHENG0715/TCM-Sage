"""
E2E Query Tests for TCM-Sage RAG System.
Tests queries with the current KG and vector store.
"""
import sys
sys.path.insert(0, 'src')

from retriever import HybridRetriever

print("=" * 60)
print("E2E Query Tests")
print("=" * 60)

# Initialize retriever
print("\nInitializing retriever...")
retriever = HybridRetriever(
    vectorstore_path='vectorstore/chroma',
    graph_data_path='data/graph/entities_partial.json'
)
print(f"Graph loaded: {retriever.knowledge_graph.graph.number_of_nodes()} entities, {retriever.knowledge_graph.graph.number_of_edges()} relationships")

# Test queries
test_queries = [
    "川芎治什么?",
    "手太阴肺经的循行路线是什么?",
    "营气和卫气有什么区别?",
]

for query in test_queries:
    print("\n" + "=" * 60)
    print(f"QUERY: {query}")
    print("=" * 60)
    
    # Use hybrid_search for combined vector + graph
    results = retriever.hybrid_search(query, vector_k=3, graph_depth=1)
    
    # Separate text chunks from graph facts
    text_results = [r for r in results if r.metadata.get('source_type') != 'knowledge_graph']
    graph_results = [r for r in results if r.metadata.get('source_type') == 'knowledge_graph']
    
    print(f"\n📚 Text Results ({len(text_results)} chunks):")
    for i, doc in enumerate(text_results[:2], 1):
        content = doc.page_content[:100] if hasattr(doc, 'page_content') else str(doc)[:100]
        print(f"  {i}. {content}...")
    
    print(f"\n🔗 Graph Results ({len(graph_results)} facts):")
    for i, doc in enumerate(graph_results[:5], 1):
        print(f"  {i}. {doc.page_content}")
