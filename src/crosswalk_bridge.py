"""
Query-time crosswalk bridge for mapping RAG terms to SymMap node IDs.

This module intentionally uses the approved CSV as the only bridge input.
It does not depend on the legacy Neijing extracted KG graph.
"""

from __future__ import annotations

import csv
import os
import unicodedata
from functools import lru_cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_APPROVED_CROSSWALK = PROJECT_ROOT / "data" / "graph" / "crosswalk" / "seed_crosswalk_approved.csv"


def _normalize_name(raw: str) -> str:
    text = unicodedata.normalize("NFKC", (raw or "").strip()).lower()
    cleaned: list[str] = []
    for ch in text:
        if ch.isalnum():
            cleaned.append(ch)
            continue
        code = ord(ch)
        if 0x4E00 <= code <= 0x9FFF:
            cleaned.append(ch)
    return "".join(cleaned)


def _get_crosswalk_path() -> Path:
    raw = os.getenv("CROSSWALK_APPROVED_PATH", str(DEFAULT_APPROVED_CROSSWALK))
    path = Path(raw)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


@lru_cache(maxsize=1)
def _load_crosswalk_rows() -> list[dict[str, str]]:
    path = _get_crosswalk_path()
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        out: list[dict[str, str]] = []
        for row in reader:
            canonical_id = (row.get("canonical_symmap_id") or "").strip()
            neijing_name = (row.get("neijing_name") or "").strip()
            normalized_name = (row.get("normalized_name") or "").strip()
            if not canonical_id:
                continue
            if not normalized_name and neijing_name:
                normalized_name = _normalize_name(neijing_name)
            out.append(
                {
                    "canonical_symmap_id": canonical_id,
                    "neijing_name": neijing_name,
                    "normalized_name": normalized_name,
                }
            )
    return out


def resolve_query_to_symmap_ids(query: str) -> set[str]:
    """
    Resolve candidate SymMap IDs from approved crosswalk using query-time matching.

    Matching strategy:
    - direct substring against `neijing_name`
    - normalized substring against `normalized_name`
    """
    if not query:
        return set()

    rows = _load_crosswalk_rows()
    if not rows:
        return set()

    query_normalized = _normalize_name(query)
    hits: set[str] = set()
    for row in rows:
        neijing_name = row["neijing_name"]
        normalized_name = row["normalized_name"]
        if neijing_name and neijing_name in query:
            hits.add(row["canonical_symmap_id"])
            continue
        if normalized_name and normalized_name in query_normalized:
            hits.add(row["canonical_symmap_id"])
    return hits
