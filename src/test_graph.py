"""
TCM Knowledge Graph Unit Tests

Tests for the graph_builder module: entity loading, traversal, and persistence.
"""

import json
import tempfile
from pathlib import Path

from graph_builder import TCMKnowledgeGraph, create_graph_from_json


def test_add_entities():
    """Test adding entities to the graph."""
    kg = TCMKnowledgeGraph()

    kg.add_entity("s1", "Symptom", "頭痛", "Headache")
    kg.add_entity("h1", "Herb", "川芎", "Chuanxiong", pinyin="Chuan Xiong")

    assert kg.graph.number_of_nodes() == 2

    entity = kg.get_entity("s1")
    assert entity["name"] == "頭痛"
    assert entity["type"] == "Symptom"

    entity = kg.get_entity("h1")
    assert entity["pinyin"] == "Chuan Xiong"

    print("✅ test_add_entities passed")


def test_add_relationships():
    """Test adding relationships between entities."""
    kg = TCMKnowledgeGraph()

    kg.add_entity("s1", "Symptom", "頭痛", "Headache")
    kg.add_entity("h1", "Herb", "川芎", "Chuanxiong")

    kg.add_relationship("h1", "s1", "TREATS", "川芎治頭痛")

    assert kg.graph.number_of_edges() == 1

    edge_data = kg.graph.edges["h1", "s1"]
    assert edge_data["type"] == "TREATS"
    assert edge_data["description"] == "川芎治頭痛"

    print("✅ test_add_relationships passed")


def test_find_entity_by_name():
    """Test finding entities by Chinese or English name."""
    kg = TCMKnowledgeGraph()

    kg.add_entity("s1", "Symptom", "頭痛", "Headache")
    kg.add_entity("h1", "Herb", "川芎", "Chuanxiong")

    # Find by Chinese name
    assert kg.find_entity_by_name("頭痛") == "s1"

    # Find by English name
    assert kg.find_entity_by_name("Chuanxiong") == "h1"

    # Not found
    assert kg.find_entity_by_name("NonExistent") is None

    print("✅ test_find_entity_by_name passed")


def test_get_related_entities():
    """Test graph traversal to find related entities."""
    kg = TCMKnowledgeGraph()

    kg.add_entity("s1", "Symptom", "頭痛", "Headache")
    kg.add_entity("h1", "Herb", "川芎", "Chuanxiong")
    kg.add_entity("h2", "Herb", "白芷", "Baizhi")
    kg.add_entity("f1", "Formula", "川芎茶調散", "Chuanxiong Chatiao San")

    kg.add_relationship("h1", "s1", "TREATS")
    kg.add_relationship("h2", "s1", "TREATS")
    kg.add_relationship("f1", "h1", "CONTAINS")
    kg.add_relationship("f1", "s1", "TREATS")

    # Find herbs that treat 頭痛 (depth=1)
    related = kg.get_related_entities("s1", max_depth=1)
    related_names = {r["entity"]["name"] for r in related}

    assert "川芎" in related_names
    assert "白芷" in related_names
    assert "川芎茶調散" in related_names

    print("✅ test_get_related_entities passed")


def test_load_from_json():
    """Test loading graph from JSON file."""
    # Create a temp JSON file
    data = {
        "entities": [
            {"id": "s1", "type": "Symptom", "name": "頭痛", "name_en": "Headache"},
            {"id": "h1", "type": "Herb", "name": "川芎", "name_en": "Chuanxiong"},
        ],
        "relationships": [
            {"source": "h1", "target": "s1", "type": "TREATS", "description": "Test"}
        ],
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
        temp_path = f.name

    try:
        kg = create_graph_from_json(temp_path)
        assert kg.graph.number_of_nodes() == 2
        assert kg.graph.number_of_edges() == 1
        print("✅ test_load_from_json passed")
    finally:
        Path(temp_path).unlink()


def test_search_by_name():
    """Test partial name search."""
    kg = TCMKnowledgeGraph()

    kg.add_entity("s1", "Symptom", "頭痛", "Headache")
    kg.add_entity("s2", "Symptom", "胃痛", "Stomachache")
    kg.add_entity("h1", "Herb", "川芎", "Chuanxiong")

    # Search for entities containing "痛"
    matches = kg.search_by_name("痛")
    assert len(matches) == 2
    assert "s1" in matches
    assert "s2" in matches

    print("✅ test_search_by_name passed")


def test_statistics():
    """Test graph statistics."""
    kg = TCMKnowledgeGraph()

    kg.add_entity("s1", "Symptom", "頭痛", "Headache")
    kg.add_entity("h1", "Herb", "川芎", "Chuanxiong")
    kg.add_entity("h2", "Herb", "白芷", "Baizhi")
    kg.add_relationship("h1", "s1", "TREATS")
    kg.add_relationship("h2", "s1", "TREATS")

    stats = kg.get_statistics()

    assert stats["total_nodes"] == 3
    assert stats["total_edges"] == 2
    assert stats["nodes_by_type"]["Symptom"] == 1
    assert stats["nodes_by_type"]["Herb"] == 2
    assert stats["edges_by_type"]["TREATS"] == 2

    print("✅ test_statistics passed")


def test_with_project_data():
    """Test with actual project data file."""
    project_data_path = Path(__file__).parent.parent / "data" / "graph" / "entities.json"

    if not project_data_path.exists():
        print("⚠️ test_with_project_data skipped (no data file)")
        return

    kg = create_graph_from_json(str(project_data_path))
    stats = kg.get_statistics()

    print(f"Project graph: {stats['total_nodes']} entities, {stats['total_edges']} relationships")
    print(f"  Entities by type: {stats['nodes_by_type']}")
    print(f"  Relationships by type: {stats['edges_by_type']}")

    # Test traversal for 頭痛
    headache_id = kg.find_entity_by_name("頭痛")
    if headache_id:
        related = kg.get_related_entities(headache_id, max_depth=1)
        print(f"  Related to '頭痛': {len(related)} entities")

    print("✅ test_with_project_data passed")


if __name__ == "__main__":
    print("Running graph_builder tests...\n")

    test_add_entities()
    test_add_relationships()
    test_find_entity_by_name()
    test_get_related_entities()
    test_load_from_json()
    test_search_by_name()
    test_statistics()
    test_with_project_data()

    print("\n✅ All tests passed!")
