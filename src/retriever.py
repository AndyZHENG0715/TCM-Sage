"""
TCM-Sage Hybrid Retriever

This module provides the HybridRetriever class that combines vector-based
semantic search with knowledge graph traversal using Ensemble Context Aggregation.

The hybrid approach retrieves results from both sources independently:
- Vector search: Returns text chunks with semantic similarity
- Graph search: Returns entity facts based on knowledge graph traversal

Results are combined with source metadata, allowing the LLM prompt to
format them as distinct context sections.
"""

import os
from pathlib import Path
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

from crosswalk_bridge import resolve_query_to_symmap_ids
from config import GRAPH_DATA_PATH
from embeddings import get_embedding_model, rerank_documents
from graph_builder import TCMKnowledgeGraph, create_graph_from_json


class HybridRetriever:
    """
    Hybrid retriever combining vector search with knowledge graph traversal.

    Uses Ensemble Context Aggregation: both sources are queried independently
    and results are combined with source metadata for the downstream prompt.

    Attributes:
        vectorstore: ChromaDB vector store instance.
        knowledge_graph: TCMKnowledgeGraph instance.
        vector_k: Number of vector search results to return.
        graph_depth: Maximum traversal depth for graph search.
    """

    def __init__(
        self,
        vectorstore_path: str,
        graph_data_path: str,
        embedding_model: str = "text-embedding-v4",
        vector_k: int = 5,
        graph_depth: int = 1,
    ):
        """
        Initialize the hybrid retriever.

        Args:
            vectorstore_path: Path to the ChromaDB vector store.
            graph_data_path: Path to the knowledge graph JSON file.
            embedding_model: Sentence transformer model name.
            vector_k: Number of vector search results.
            graph_depth: Maximum graph traversal depth.

        Raises:
            FileNotFoundError: If vectorstore or graph data not found.
        """
        self.vector_k = vector_k
        self.graph_depth = graph_depth

        # Initialize vector store
        if not Path(vectorstore_path).exists():
            raise FileNotFoundError(f"Vector store not found: {vectorstore_path}")

        _ = embedding_model
        embeddings = get_embedding_model()
        self.vectorstore = Chroma(
            persist_directory=str(vectorstore_path),
            embedding_function=embeddings,
        )

        # Initialize knowledge graph
        self.knowledge_graph = create_graph_from_json(graph_data_path)

    def vector_search(self, query: str, k: Optional[int] = None) -> list[Document]:
        """
        Perform semantic vector search.

        Args:
            query: Search query string.
            k: Number of results (defaults to self.vector_k).

        Returns:
            List of Document objects with source and score metadata.
        """
        k = k or self.vector_k
        # Use similarity_search_with_score to capture relevance scores for debugging
        results_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)

        docs = []
        for doc, score in results_with_scores:
            if doc.metadata is None:
                doc.metadata = {}
            doc.metadata["source_type"] = "vector"
            doc.metadata["score"] = round(score, 3)  # Distance score (lower = better match)
            docs.append(doc)

        return docs

    def graph_search(
        self,
        query: str,
        depth: Optional[int] = None,
    ) -> list[Document]:
        """
        Perform knowledge graph traversal search.

        Searches for entities matching the query, then traverses the graph
        to find related entities and formats them as fact documents.

        Args:
            query: Search query string.
            depth: Traversal depth (defaults to self.graph_depth).

        Returns:
            List of Document objects containing graph facts.
        """
        depth = depth or self.graph_depth
        graph_docs = []

        # Find matching entities in the graph
        matching_ids = set(self.knowledge_graph.search_by_name(query))
        matching_ids.update(resolve_query_to_symmap_ids(query))

        for entity_id in matching_ids:
            entity = self.knowledge_graph.get_entity(entity_id)
            if not entity:
                continue

            # Get related entities via traversal
            related = self.knowledge_graph.get_related_entities(
                entity_id,
                max_depth=depth,
            )

            for item in related:
                rel_entity = item["entity"]
                relationship = item["relationship"]

                # Use original graph direction: source --> target
                # relationship["source"] and ["target"] are the actual node IDs
                source_id = relationship["source"]
                target_id = relationship["target"]

                # Get entity names from the graph using the original direction
                source_entity = self.knowledge_graph.get_entity(source_id)
                target_entity = self.knowledge_graph.get_entity(target_id)

                source_name = source_entity.get("name", source_id) if source_entity else source_id
                target_name = target_entity.get("name", target_id) if target_entity else target_id
                target_type = target_entity.get("type", "Unknown") if target_entity else "Unknown"

                # Format as a readable fact with correct direction
                fact_text = self._format_graph_fact(
                    source_name=source_name,
                    relationship_type=relationship["type"],
                    target_name=target_name,
                    target_type=target_type,
                    description=relationship.get("description", ""),
                )

                graph_docs.append(
                    Document(
                        page_content=fact_text,
                        metadata={
                            "source_type": "graph",
                            "entity_id": rel_entity.get("id"),
                            "entity_type": rel_entity.get("type"),
                            "relationship_type": relationship["type"],
                            "depth": item["depth"],
                            "source_ref": relationship.get("source_ref"),  # Provenance
                        },
                    )
                )

        return graph_docs

    def _format_graph_fact(
        self,
        source_name: str,
        relationship_type: str,
        target_name: str,
        target_type: str,
        description: str = "",
    ) -> str:
        """
        Format a graph relationship as a readable fact string.

        Uses arrow notation for debug-friendly output:
        Source --RELATIONSHIP--> Target (Type)

        Args:
            source_name: Name of source entity.
            relationship_type: Type of relationship.
            target_name: Name of target entity.
            target_type: Type of target entity.
            description: Optional description.

        Returns:
            Formatted fact string with arrow notation.
        """
        # Use arrow notation for debug-friendly format
        fact = f"{source_name} --{relationship_type}--> {target_name} ({target_type})"
        if description:
            fact += f" | {description}"

        return fact

    def hybrid_search(
        self,
        query: str,
        vector_k: Optional[int] = None,
        graph_depth: Optional[int] = None,
    ) -> list[Document]:
        """
        Perform ensemble hybrid search combining vector and graph results.

        Args:
            query: Search query string.
            vector_k: Number of vector results.
            graph_depth: Graph traversal depth.

        Returns:
            Combined list of Documents from both sources.
            Vector results appear first, followed by graph facts.
        """
        # Get vector results (text chunks)
        vector_docs = self.vector_search(query, k=vector_k)

        if vector_docs:
            doc_texts = [doc.page_content for doc in vector_docs]
            reranked = rerank_documents(query, doc_texts, top_n=len(vector_docs))

            if reranked:
                ranked_vector_docs = []
                for item in reranked:
                    doc = vector_docs[item["index"]]
                    if doc.metadata is None:
                        doc.metadata = {}
                    doc.metadata["relevance_score"] = item["relevance_score"]
                    ranked_vector_docs.append(doc)
                vector_docs = ranked_vector_docs

        # Get graph results (entity facts)
        graph_docs = self.graph_search(query, depth=graph_depth)

        # Combine with vector results first, then graph facts
        return vector_docs + graph_docs

    def get_statistics(self) -> dict:
        """
        Get statistics about the retriever's data sources.

        Returns:
            Dictionary with vectorstore and graph statistics.
        """
        graph_stats = self.knowledge_graph.get_statistics()

        return {
            "vectorstore": {
                "collection_count": self.vectorstore._collection.count(),
            },
            "knowledge_graph": graph_stats,
        }


def create_hybrid_retriever(
    vectorstore_path: Optional[str] = None,
    graph_data_path: Optional[str] = None,
    vector_k: int = 5,
    graph_depth: int = 1,
) -> HybridRetriever:
    """
    Factory function to create a HybridRetriever with default paths.

    Args:
        vectorstore_path: Path to vector store (defaults to project vectorstore).
        graph_data_path: Path to graph JSON (defaults to project graph data).
        vector_k: Number of vector results.
        graph_depth: Graph traversal depth.

    Returns:
        Configured HybridRetriever instance.
    """
    project_root = Path(__file__).parent.parent

    if vectorstore_path is None:
        vectorstore_path = str(project_root / "vectorstore" / "chroma")

    if graph_data_path is None:
        graph_data_path = os.getenv("GRAPH_DATA_PATH", str(GRAPH_DATA_PATH))

    return HybridRetriever(
        vectorstore_path=vectorstore_path,
        graph_data_path=graph_data_path,
        vector_k=vector_k,
        graph_depth=graph_depth,
    )


if __name__ == "__main__":
    # Quick test when run directly
    print("Testing Hybrid Retriever...")

    try:
        retriever = create_hybrid_retriever()
        stats = retriever.get_statistics()
        print(f"Vectorstore documents: {stats['vectorstore']['collection_count']}")
        print(f"Graph entities: {stats['knowledge_graph']['total_nodes']}")
        print(f"Graph relationships: {stats['knowledge_graph']['total_edges']}")

        # Test hybrid search
        query = "頭痛"
        print(f"\nTesting hybrid search for: '{query}'")

        results = retriever.hybrid_search(query)
        print(f"Total results: {len(results)}")

        vector_count = sum(1 for d in results if d.metadata.get("source_type") == "vector")
        graph_count = sum(1 for d in results if d.metadata.get("source_type") == "graph")
        print(f"  Vector results: {vector_count}")
        print(f"  Graph results: {graph_count}")

        print("\nGraph facts found:")
        for doc in results:
            if doc.metadata.get("source_type") == "graph":
                print(f"  - {doc.page_content}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure to run 'python src/ingest.py' first to create the vector store.")
