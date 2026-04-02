"""
TCM Knowledge Graph Builder

This module provides the TCMKnowledgeGraph class for building and querying
an in-memory knowledge graph of Traditional Chinese Medicine entities.

Entity Types:
    - Symptom: Clinical symptoms and conditions
    - Herb: Medicinal herbs
    - Formula: Classical prescriptions
    - Disease: Modern disease / phenotype (e.g., SymMap MM layer)
    - Ingredient: Molecular constituents (e.g., SymMap IM)
    - Target: Gene/protein targets (e.g., SymMap TM)
    - Syndrome: TCM pattern / syndrome (e.g., SymMap SMYS)

Relationship Types:
    - TREATS: Herb/Formula treats a Symptom
    - CONTAINS: Formula contains an Herb; Herb contains Ingredient (SymMap HM–IM)
    - INDICATES: Symptom indicates Syndrome / TCM pattern (SymMap SMYS links)
    - TARGETS: Ingredient acts on Target (SymMap IM–TM)
    - MAPS_TO: MM symptom to TCM symptom or cross-vocabulary alignment
    - ASSOCIATED_WITH: Target–disease and generic associations
    - CORRELATES_WITH: Symptom–disease links (direct or inferred, e.g. SymMap SM–MM)
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import jieba
import networkx as nx


jieba.setLogLevel(logging.WARNING)

_COLLOQUIAL_TO_TCM = {
    "睡眠": "失眠", "睡不着": "失眠", "睡不好": "失眠",
    "入睡困难": "失眠", "多梦": "多梦",
    "头疼": "头痛", "肚子疼": "腹痛", "肚子痛": "腹痛",
    "胃疼": "胃痛", "腰疼": "腰痛",
    "拉肚子": "泄泻", "便秘": "便秘", "没胃口": "食欲不振",
    "不想吃饭": "食欲不振",
    "心慌": "心悸", "胸闷": "胸闷",
    "上火": "火热", "怕冷": "畏寒", "出汗多": "多汗",
    "没力气": "乏力", "疲劳": "疲劳", "累": "乏力",
}

_GENERIC_FILLER_WORDS = {
    "会导致", "导致", "什么", "问题", "怎么", "应该", "可以",
    "为什么", "如何", "哪些", "怎样", "能不能", "有没有",
    "用吗", "药方", "调理", "治疗", "原因", "方法", "建议",
    "请问", "想问", "帮我", "告诉我", "缺乏",
}

_PUNCTUATION_CHARS = "。，！？、；：\"'()《》【】.,!?;:"


class TCMKnowledgeGraph:
    """
    In-memory knowledge graph for TCM entities using NetworkX.

    Attributes:
        graph: NetworkX DiGraph containing entities as nodes and relationships as edges.
    """

    # Valid entity and relationship types (expanded for Neijing content)
    ENTITY_TYPES = {
        "Symptom", "Pattern", "Herb", "Formula", "TreatmentMethod",
        "Meridian", "Acupoint", "BodyPart", "Substance",
        "Disease", "Ingredient", "Target", "Syndrome",
    }
    RELATIONSHIP_TYPES = {
        "TREATS", "CONTAINS", "INDICATES", "APPLIES_TO", "LOCATED_ON",
        "ORIGINATES_FROM", "FLOWS_THROUGH", "DERIVED_FROM", "ENTERS", "GOVERNS",
        "MAPS_TO", "TARGETS", "ASSOCIATED_WITH", "CORRELATES_WITH",
    }

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
        max_results: int = 20,
    ) -> list[dict]:
        """
        Get entities related to the given entity via graph traversal.

        Args:
            entity_id: ID of the starting entity.
            relationship_type: Filter by relationship type (optional).
            direction: Traversal direction - 'outgoing', 'incoming', or 'both'.
            max_depth: Maximum traversal depth (1 = direct neighbors, 2 = neighbors' neighbors).
            max_results: Maximum number of related entities returned after traversal.

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
                        "source_ref": edge_attrs.get("source_ref"),  # Provenance data
                    },
                    "depth": depth,
                })

                # Recurse for deeper traversal
                traverse(related_id, depth + 1)

        traverse(entity_id, 1)
        return results[:max_results]

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

        cleaned_query = query.translate(str.maketrans("", "", _PUNCTUATION_CHARS)).strip()
        for filler in sorted(_GENERIC_FILLER_WORDS, key=len, reverse=True):
            cleaned_query = cleaned_query.replace(filler, "")
        cleaned_query = cleaned_query.strip()

        alias_expanded_terms = set()
        alias_source = cleaned_query or query
        for colloquial, canonical in _COLLOQUIAL_TO_TCM.items():
            if colloquial in alias_source:
                alias_expanded_terms.add(canonical)

        jieba_source = cleaned_query or query
        jieba_segments = {
            segment.strip()
            for segment in jieba.lcut(jieba_source)
            if segment.strip()
        }

        search_terms = set(alias_expanded_terms)
        search_terms.update(jieba_segments)
        search_terms.add(query)
        if cleaned_query:
            search_terms.add(cleaned_query)

        query_variants = set(search_terms)
        for term in list(search_terms):
            if not term:
                continue
            query_variants.add(term)
            term_variants = {term}
            for simp, trad in simp_trad_map.items():
                next_variants = set(term_variants)
                for variant in term_variants:
                    if simp in variant:
                        next_variants.add(variant.replace(simp, trad))
                    if trad in variant:
                        next_variants.add(variant.replace(trad, simp))
                term_variants = next_variants
            query_variants.update(term_variants)

        normalized_query_variants = {variant.strip() for variant in query_variants if variant.strip()}
        for simp, trad in simp_trad_map.items():
            current_variants = list(normalized_query_variants)
            for variant in current_variants:
                if simp in variant:
                    normalized_query_variants.add(variant.replace(simp, trad))
                if trad in variant:
                    normalized_query_variants.add(variant.replace(trad, simp))

        for node_id, attrs in self.graph.nodes(data=True):
            name = attrs.get("name", "")
            name_en = attrs.get("name_en", "").lower()
            
            # Skip empty names (would match everything due to "" in q being True)
            if not name and not name_en:
                continue

            for q in normalized_query_variants:
                q_lower = q.lower()
                # Check if entity name appears in query (for extracting entities from sentences)
                # OR if query appears in entity name (for partial name searches)
                if (name and (name in q or q in name)) or (name_en and (name_en in q_lower or q_lower in name_en)):
                    matches.append(node_id)
                    break  # Avoid duplicate matches for same entity

        if matches:
            return matches

        keyword_segments = [segment for segment in jieba_segments if len(segment) >= 2]
        if not keyword_segments:
            return []

        fallback_matches = []
        for node_id, attrs in self.graph.nodes(data=True):
            name = attrs.get("name", "")
            if not name:
                continue

            for segment in keyword_segments:
                if segment in name or name in segment:
                    fallback_matches.append(node_id)
                    break

        return fallback_matches

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

        # Load entities (support both old 'name' and new 'mention' format)
        for entity in data.get("entities", []):
            entity = entity.copy()  # Avoid modifying original
            
            # Get the name FIRST before any pops (new format uses 'mention')
            name = entity.get("name") or entity.get("mention", "")
            
            entity_id = entity.pop("id", f"entity_{name}" if name else "unknown")
            entity_type = entity.pop("type", "Unknown")
            
            # Clean up fields we've already extracted
            entity.pop("name", None)
            entity.pop("mention", None)
            name_en = entity.pop("name_en", "")
            
            # Skip entities with no name
            if not name:
                continue
            
            try:
                self.graph.add_node(
                    entity_id,
                    type=entity_type,
                    name=name,
                    name_en=name_en,
                    **{k: v for k, v in entity.items() if k not in ('source_ref',)}
                )
            except Exception:
                pass  # Skip malformed entries

        # Load relationships (support both old and new format)
        for rel in data.get("relationships", []):
            try:
                source_id = rel.get("source") or rel.get("head", "")
                target_id = rel.get("target") or rel.get("tail", "")
                rel_type = rel.get("type") or rel.get("relation", "RELATED_TO")
                
                if source_id in self.graph and target_id in self.graph:
                    self.graph.add_edge(
                        source_id,
                        target_id,
                        type=rel_type,
                        description=rel.get("description", rel.get("evidence", "")),
                        source_ref=rel.get("source_ref"),  # Provenance for citations
                    )
            except Exception:
                pass  # Skip malformed relationships

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
