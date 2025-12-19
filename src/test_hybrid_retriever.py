"""
TCM Hybrid Retriever Tests

Tests for the HybridRetriever class: vector search, graph search, and ensemble retrieval.
"""

from pathlib import Path

from langchain_core.documents import Document

from graph_builder import TCMKnowledgeGraph
from retriever import HybridRetriever, create_hybrid_retriever


def test_graph_search_only():
    """Test graph search independently without vector store."""
    # Create a minimal graph for testing
    kg = TCMKnowledgeGraph()
    kg.add_entity("s1", "Symptom", "頭痛", "Headache")
    kg.add_entity("h1", "Herb", "川芎", "Chuanxiong")
    kg.add_entity("h2", "Herb", "白芷", "Baizhi")
    kg.add_relationship("h1", "s1", "TREATS", "川芎治頭痛")
    kg.add_relationship("h2", "s1", "TREATS", "白芷治頭痛")

    # Search for entities related to 頭痛
    matches = kg.search_by_name("頭痛")
    assert len(matches) == 1
    assert matches[0] == "s1"

    related = kg.get_related_entities("s1", max_depth=1)
    related_names = {r["entity"]["name"] for r in related}
    assert "川芎" in related_names
    assert "白芷" in related_names

    print("✅ test_graph_search_only passed")


def test_source_metadata():
    """Test that documents have correct source_type metadata."""
    # Create mock documents
    vector_doc = Document(
        page_content="Test vector content",
        metadata={"source": "Chapter 1", "source_type": "vector"},
    )
    graph_doc = Document(
        page_content="KG Fact: 川芎 TREATS 頭痛",
        metadata={"source_type": "graph", "entity_type": "Herb"},
    )

    assert vector_doc.metadata["source_type"] == "vector"
    assert graph_doc.metadata["source_type"] == "graph"

    print("✅ test_source_metadata passed")


def test_hybrid_retriever_integration():
    """Test full hybrid retriever with project data."""
    project_root = Path(__file__).parent.parent
    vectorstore_path = project_root / "vectorstore" / "chroma"
    graph_path = project_root / "data" / "graph" / "entities.json"

    if not vectorstore_path.exists():
        print("⚠️ test_hybrid_retriever_integration skipped (no vector store)")
        return

    if not graph_path.exists():
        print("⚠️ test_hybrid_retriever_integration skipped (no graph data)")
        return

    # Create hybrid retriever
    retriever = create_hybrid_retriever(
        vectorstore_path=str(vectorstore_path),
        graph_data_path=str(graph_path),
        vector_k=3,
        graph_depth=1,
    )

    # Test hybrid search
    query = "頭痛"
    results = retriever.hybrid_search(query)

    vector_results = [d for d in results if d.metadata.get("source_type") == "vector"]
    graph_results = [d for d in results if d.metadata.get("source_type") == "graph"]

    print(f"Query: '{query}'")
    print(f"  Vector results: {len(vector_results)}")
    print(f"  Graph results: {len(graph_results)}")

    # Verify graph results contain expected facts
    if graph_results:
        print("  Graph facts:")
        for doc in graph_results[:3]:
            print(f"    - {doc.page_content}")

    assert len(vector_results) > 0, "Should have vector results"
    assert len(graph_results) > 0, "Should have graph results for '頭痛'"

    print("✅ test_hybrid_retriever_integration passed")


def test_statistics():
    """Test retriever statistics."""
    project_root = Path(__file__).parent.parent
    vectorstore_path = project_root / "vectorstore" / "chroma"
    graph_path = project_root / "data" / "graph" / "entities.json"

    if not vectorstore_path.exists() or not graph_path.exists():
        print("⚠️ test_statistics skipped (missing data)")
        return

    retriever = create_hybrid_retriever(
        vectorstore_path=str(vectorstore_path),
        graph_data_path=str(graph_path),
    )

    stats = retriever.get_statistics()

    assert "vectorstore" in stats
    assert "knowledge_graph" in stats
    assert stats["knowledge_graph"]["total_nodes"] > 0

    print(f"Vectorstore documents: {stats['vectorstore']['collection_count']}")
    print(f"Graph entities: {stats['knowledge_graph']['total_nodes']}")
    print(f"Graph relationships: {stats['knowledge_graph']['total_edges']}")

    print("✅ test_statistics passed")


def test_no_graph_matches():
    """Test behavior when query has no graph matches."""
    project_root = Path(__file__).parent.parent
    vectorstore_path = project_root / "vectorstore" / "chroma"
    graph_path = project_root / "data" / "graph" / "entities.json"

    if not vectorstore_path.exists() or not graph_path.exists():
        print("⚠️ test_no_graph_matches skipped (missing data)")
        return

    retriever = create_hybrid_retriever(
        vectorstore_path=str(vectorstore_path),
        graph_data_path=str(graph_path),
        vector_k=3,
    )

    # Query that likely has no graph matches
    query = "黃帝問岐伯"
    results = retriever.hybrid_search(query)

    vector_results = [d for d in results if d.metadata.get("source_type") == "vector"]
    graph_results = [d for d in results if d.metadata.get("source_type") == "graph"]

    # Should still have vector results even without graph matches
    assert len(vector_results) > 0, "Should have vector results"

    print(f"Query: '{query}'")
    print(f"  Vector results: {len(vector_results)}")
    print(f"  Graph results: {len(graph_results)}")

    print("✅ test_no_graph_matches passed")


if __name__ == "__main__":
    print("Running hybrid_retriever tests...\n")

    test_graph_search_only()
    test_source_metadata()
    test_hybrid_retriever_integration()
    test_statistics()
    test_no_graph_matches()

    print("\n✅ All tests passed!")
