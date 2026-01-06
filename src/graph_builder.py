"""
TCM Knowledge Graph Builder

This module provides the TCMKnowledgeGraph class for building and querying
an in-memory knowledge graph of Traditional Chinese Medicine entities.

Entity Types:
    - Symptom: Clinical symptoms and conditions
    - Herb: Medicinal herbs
    - Formula: Classical prescriptions

Relationship Types:
    - TREATS: Herb/Formula treats a Symptom
    - CONTAINS: Formula contains an Herb
    - ASSOCIATED_WITH: Symptom associated with another Symptom
"""

import json
import pickle
from pathlib import Path
from typing import Optional

import networkx as nx


class TCMKnowledgeGraph:
    """
    In-memory knowledge graph for TCM entities using NetworkX.

    Attributes:
        graph: NetworkX DiGraph containing entities as nodes and relationships as edges.
    """

    # Valid entity and relationship types
    ENTITY_TYPES = {"Symptom", "Herb", "Formula"}
    RELATIONSHIP_TYPES = {"TREATS", "CONTAINS", "ASSOCIATED_WITH"}

    def __init__(self):
        """Initialize an empty knowledge graph."""
        self.graph = nx.DiGraph()

    def add_entity(
        self,
        entity_id: str,
        entity_type: str,
        name: str,
        name_en: str = "",
        **attributes,
    ) -> None:
        """
        Add an entity node to the graph.

        Args:
            entity_id: Unique identifier for the entity.
            entity_type: Type of entity (Symptom, Herb, Formula).
            name: Chinese name of the entity.
            name_en: English name of the entity.
            **attributes: Additional attributes (description, pinyin, etc.).

        Raises:
            ValueError: If entity_type is not valid.
        """
        if entity_type not in self.ENTITY_TYPES:
            raise ValueError(
                f"Invalid entity type: {entity_type}. Must be one of {self.ENTITY_TYPES}"
            )

        self.graph.add_node(
            entity_id,
            type=entity_type,
            name=name,
            name_en=name_en,
            **attributes,
        )

    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        description: str = "",
    ) -> None:
        """
        Add a relationship edge between two entities.

        Args:
            source_id: ID of the source entity.
            target_id: ID of the target entity.
            relationship_type: Type of relationship (TREATS, CONTAINS, ASSOCIATED_WITH).
            description: Optional description of the relationship.

        Raises:
            ValueError: If relationship_type is not valid or entities don't exist.
        """
        if relationship_type not in self.RELATIONSHIP_TYPES:
            raise ValueError(
                f"Invalid relationship type: {relationship_type}. "
                f"Must be one of {self.RELATIONSHIP_TYPES}"
            )

        if source_id not in self.graph:
            raise ValueError(f"Source entity not found: {source_id}")
        if target_id not in self.graph:
            raise ValueError(f"Target entity not found: {target_id}")

        self.graph.add_edge(
            source_id,
            target_id,
            type=relationship_type,
            description=description,
        )

    def get_entity(self, entity_id: str) -> Optional[dict]:
        """
        Get entity attributes by ID.

        Args:
            entity_id: ID of the entity to retrieve.

        Returns:
            Dictionary of entity attributes, or None if not found.
        """
        if entity_id in self.graph:
            return dict(self.graph.nodes[entity_id])
        return None

    def find_entity_by_name(self, name: str) -> Optional[str]:
        """
        Find entity ID by Chinese or English name (exact match).

        Args:
            name: Name to search for.

        Returns:
            Entity ID if found, None otherwise.
        """
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get("name") == name or attrs.get("name_en") == name:
                return node_id
        return None

    def get_related_entities(
        self,
        entity_id: str,
        relationship_type: Optional[str] = None,
        direction: str = "both",
        max_depth: int = 1,
    ) -> list[dict]:
        """
        Get entities related to the given entity via graph traversal.

        Args:
            entity_id: ID of the starting entity.
            relationship_type: Filter by relationship type (optional).
            direction: Traversal direction - 'outgoing', 'incoming', or 'both'.
            max_depth: Maximum traversal depth (1 = direct neighbors, 2 = neighbors' neighbors).

        Returns:
            List of related entities with relationship info:
            [{"entity": {...}, "relationship": {...}, "depth": int}, ...]
        """
        if entity_id not in self.graph:
            return []

        results = []
        visited = {entity_id}

        def traverse(current_id: str, depth: int):
            if depth > max_depth:
                return

            edges_to_check = []

            if direction in ("outgoing", "both"):
                edges_to_check.extend(
                    (current_id, successor, self.graph.edges[current_id, successor])
                    for successor in self.graph.successors(current_id)
                )

            if direction in ("incoming", "both"):
                edges_to_check.extend(
                    (predecessor, current_id, self.graph.edges[predecessor, current_id])
                    for predecessor in self.graph.predecessors(current_id)
                )

            for source, target, edge_attrs in edges_to_check:
                # Filter by relationship type if specified
                if relationship_type and edge_attrs.get("type") != relationship_type:
                    continue

                # Determine the related entity (the one that isn't current_id)
                related_id = target if source == current_id else source

                if related_id in visited:
                    continue

                visited.add(related_id)

                entity_attrs = dict(self.graph.nodes[related_id])
                entity_attrs["id"] = related_id

                results.append({
                    "entity": entity_attrs,
                    "relationship": {
                        "type": edge_attrs.get("type"),
                        "description": edge_attrs.get("description", ""),
                        "source": source,
                        "target": target,
                    },
                    "depth": depth,
                })

                # Recurse for deeper traversal
                traverse(related_id, depth + 1)

        traverse(entity_id, 1)
        return results

    def search_by_name(self, query: str) -> list[str]:
        """
        Search for entities whose name appears in the query OR contains the query.

        This bidirectional search enables:
        - Exact/partial entity name matches (e.g., query "頭痛" matches entity "頭痛")
        - Entity extraction from long queries (e.g., query "患者頭痛三十年" matches entity "頭痛")
        - Cross-variant Chinese matching (Simplified ↔ Traditional)

        Args:
            query: Search string (can be a single term or a long sentence).

        Returns:
            List of matching entity IDs.
        """
        matches = []
        query_lower = query.lower()

        # Common Simplified ↔ Traditional mappings for TCM terms
        simp_trad_map = {
            "头痛": "頭痛", "头": "頭", "痛": "痛",
            "眩晕": "眩暈", "晕": "暈",
            "失眠": "失眠",
            "疲劳": "疲勞", "劳": "勞",
            "咳嗽": "咳嗽",
            "川芎": "川芎",
            "白芷": "白芷",
            "天麻": "天麻",
            "酸枣仁": "酸棗仁", "枣": "棗",
            "黄芪": "黃芪", "黄": "黃",
            "杏仁": "杏仁",
        }

        # Build query variants (original + converted)
        query_variants = {query}
        for simp, trad in simp_trad_map.items():
            if simp in query:
                query_variants.add(query.replace(simp, trad))
            if trad in query:
                query_variants.add(query.replace(trad, simp))

        for node_id, attrs in self.graph.nodes(data=True):
            name = attrs.get("name", "")
            name_en = attrs.get("name_en", "").lower()

            for q in query_variants:
                # Check if entity name appears in query (for extracting entities from sentences)
                # OR if query appears in entity name (for partial name searches)
                if name in q or name_en in query_lower or q in name or query_lower in name_en:
                    matches.append(node_id)
                    break  # Avoid duplicate matches for same entity

        return matches

    def load_from_json(self, json_path: str) -> None:
        """
        Load entities and relationships from a JSON file.

        Args:
            json_path: Path to the JSON file.

        Raises:
            FileNotFoundError: If file doesn't exist.
            json.JSONDecodeError: If file is not valid JSON.
        """
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"Graph data file not found: {json_path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Load entities
        for entity in data.get("entities", []):
            entity_id = entity.pop("id")
            entity_type = entity.pop("type")
            name = entity.pop("name")
            name_en = entity.pop("name_en", "")
            self.add_entity(entity_id, entity_type, name, name_en, **entity)

        # Load relationships
        for rel in data.get("relationships", []):
            self.add_relationship(
                source_id=rel["source"],
                target_id=rel["target"],
                relationship_type=rel["type"],
                description=rel.get("description", ""),
            )

    def save_graph(self, pickle_path: str) -> None:
        """
        Save the graph to a pickle file for fast loading.

        Args:
            pickle_path: Path to save the pickle file.
        """
        path = Path(pickle_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self.graph, f)

    def load_graph(self, pickle_path: str) -> None:
        """
        Load the graph from a pickle file.

        Args:
            pickle_path: Path to the pickle file.

        Raises:
            FileNotFoundError: If file doesn't exist.
        """
        path = Path(pickle_path)
        if not path.exists():
            raise FileNotFoundError(f"Graph pickle file not found: {pickle_path}")

        with open(path, "rb") as f:
            self.graph = pickle.load(f)

    def get_statistics(self) -> dict:
        """
        Get statistics about the knowledge graph.

        Returns:
            Dictionary with node/edge counts by type.
        """
        stats = {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "nodes_by_type": {},
            "edges_by_type": {},
        }

        for _, attrs in self.graph.nodes(data=True):
            entity_type = attrs.get("type", "Unknown")
            stats["nodes_by_type"][entity_type] = (
                stats["nodes_by_type"].get(entity_type, 0) + 1
            )

        for _, _, attrs in self.graph.edges(data=True):
            rel_type = attrs.get("type", "Unknown")
            stats["edges_by_type"][rel_type] = (
                stats["edges_by_type"].get(rel_type, 0) + 1
            )

        return stats


def create_graph_from_json(json_path: str) -> TCMKnowledgeGraph:
    """
    Factory function to create and load a graph from JSON.

    Args:
        json_path: Path to the JSON file.

    Returns:
        Loaded TCMKnowledgeGraph instance.
    """
    graph = TCMKnowledgeGraph()
    graph.load_from_json(json_path)
    return graph


if __name__ == "__main__":
    # Quick test when run directly
    from pathlib import Path

    json_path = Path(__file__).parent.parent / "data" / "graph" / "entities.json"

    print("Loading TCM Knowledge Graph...")
    kg = create_graph_from_json(str(json_path))

    stats = kg.get_statistics()
    print(f"Loaded {stats['total_nodes']} entities and {stats['total_edges']} relationships")
    print(f"Entities by type: {stats['nodes_by_type']}")
    print(f"Relationships by type: {stats['edges_by_type']}")

    # Test traversal
    headache_id = kg.find_entity_by_name("頭痛")
    if headache_id:
        print(f"\nEntities related to '頭痛' (Headache):")
        related = kg.get_related_entities(headache_id, max_depth=1)
        for item in related:
            entity = item["entity"]
            rel = item["relationship"]
            print(f"  - {entity['name']} ({entity['type']}) via {rel['type']}")
