"""Diagnose graph search issue."""
import sys
sys.path.insert(0, 'src')

from graph_builder import create_graph_from_json

graph = create_graph_from_json('data/graph/entities_partial.json')
print(f"Graph stats: {graph.get_statistics()}")

# Check what entity names look like
print("\n=== Sample Entity Nodes ===")
for i, (node_id, attrs) in enumerate(graph.graph.nodes(data=True)):
    if i < 5:
        print(f"  ID: {node_id}")
        print(f"  Attrs: {attrs}")
        print()

# Test search
print("\n=== Testing search_by_name ===")
test_queries = ["川芎", "营", "手太阴", "肺"]
for q in test_queries:
    matches = graph.search_by_name(q)
    print(f"Query '{q}': {len(matches)} matches -> {matches[:3]}")
