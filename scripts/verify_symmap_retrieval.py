#!/usr/bin/env python3
"""
Verify SymMap-shaped graph data supports the same graph traversal used in hybrid retrieval.

Run from project root: python scripts/verify_symmap_retrieval.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import GRAPH_DATA_PATH  # noqa: E402
from src.graph_builder import create_graph_from_json  # noqa: E402
from src.ui_backend import _search_graph_documents  # noqa: E402


def main() -> None:
    if not GRAPH_DATA_PATH.exists():
        raise SystemExit(f"Missing graph file: {GRAPH_DATA_PATH}")

    kg = create_graph_from_json(str(GRAPH_DATA_PATH))
    matching_ids = kg.search_by_name("頭痛")
    assert matching_ids, f"expected at least one entity match for 頭痛 in {GRAPH_DATA_PATH}"
    eid = matching_ids[0]

    related = kg.get_related_entities(eid, max_depth=2)
    assert related, "expected graph neighbors for 頭痛"

    types_found = {item["relationship"]["type"] for item in related}
    print(f"OK: SymMap sample graph — 頭痛 ({eid}) has {len(related)} related edges")
    print(f"    Relationship types: {sorted(types_found)}")

    graph_path = str(GRAPH_DATA_PATH.resolve())
    docs = _search_graph_documents("頭痛", graph_path, depth=2)
    assert docs, "expected _search_graph_documents to return SymMap facts for 頭痛"
    assert any(
        "頭痛" in d.page_content or d.metadata.get("source_type") == "graph" for d in docs
    ), "graph facts should mention entity or carry graph metadata"
    print(f"OK: ui_backend._search_graph_documents — {len(docs)} graph Document(s) for query 頭痛")


if __name__ == "__main__":
    main()
