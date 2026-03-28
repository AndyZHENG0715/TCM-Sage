#!/usr/bin/env python3
"""
Import SymMap-style CSV/TSV exports into TCM-Sage `entities.json` graph format.

See `.planning/phases/02-standard-kg-integration/SYMMAP_MAPPING.md` for file layout.

Examples:
  python scripts/import_symmap_kg.py --sample -o data/graph/symmap_entities.json
  python scripts/import_symmap_kg.py --input-dir path/to/symmap_csv -o data/graph/symmap_entities.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def _norm_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def _read_table(path: Path) -> list[dict[str, str]]:
    """Load a CSV/TSV file as list of dict rows (string values)."""
    text = path.read_text(encoding="utf-8", errors="replace")
    delimiter = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        delimiter = dialect.delimiter
    except csv.Error:
        pass

    lines = text.splitlines()
    if not lines:
        return []
    reader = csv.DictReader(lines, delimiter=delimiter)
    return [{k: (v or "").strip() for k, v in row.items()} for row in reader]


def _pick(row: dict[str, str], *candidates: str) -> str:
    keys = {k.lower(): k for k in row}
    for cand in candidates:
        for variant in (cand, cand.lower(), cand.upper()):
            if variant in row and row[variant]:
                return row[variant]
        lk = cand.lower()
        if lk in keys and row[keys[lk]]:
            return row[keys[lk]]
    return ""


def _detect_prefix(entity_id: str) -> str | None:
    if not entity_id:
        return None
    m = re.match(r"^(SM[A-Z]{2})", entity_id.upper())
    return m.group(1) if m else None


def parse_entity_rows(rows: list[dict[str, str]], source_file: str) -> list[dict[str, Any]]:
    """Map arbitrary SymMap-like rows to entity dicts."""
    entities: list[dict[str, Any]] = []
    for row in rows:
        eid = _pick(row, "SymMap_ID", "symmap_id", "ID", "id")
        if not eid:
            continue
        prefix = _detect_prefix(eid) or ""
        name = _pick(row, "Name_CN", "Chinese", "name_cn", "Name", "name", "Herb_Name")
        name_en = _pick(row, "Name_EN", "English", "name_en", "Name_en")
        pinyin = _pick(row, "Pinyin", "pinyin", "PINYIN")
        desc = _pick(row, "Description", "description", "Function", "function")

        type_map = {
            "SMTS": "Symptom",
            "SMMS": "Symptom",
            "SMHB": "Herb",
            "SMIT": "Ingredient",
            "SMTT": "Target",
            "SMDE": "Disease",
            "SMYS": "Syndrome",
        }
        etype = type_map.get(prefix, "Symptom")
        symmap_component = prefix or None

        if not name and name_en:
            name = name_en

        ent: dict[str, Any] = {
            "id": eid,
            "type": etype,
            "name": name or eid,
            "name_en": name_en,
            "pinyin": pinyin,
            "description": desc,
            "source_ref": source_file,
        }
        if symmap_component:
            ent["symmap_component"] = symmap_component
        if prefix == "SMMS":
            ent["symmap_component"] = "SMMS"
        entities.append(ent)
    return entities


def parse_relationship_rows(
    rows: list[dict[str, str]],
    source_file: str,
    default_type: str = "ASSOCIATED_WITH",
) -> list[dict[str, Any]]:
    rels: list[dict[str, Any]] = []
    for row in rows:
        src = _pick(row, "Source", "source", "Head", "head", "ID1", "Herb_ID", "SMTS_ID")
        tgt = _pick(row, "Target", "target", "Tail", "tail", "ID2", "Symptom_ID", "SMDE_ID")
        if not src or not tgt:
            src = _pick(row, "SMHB_ID", "Herb_ID")
            tgt = _pick(row, "SMTS_ID", "Symptom_ID")
        rtype = _pick(row, "Type", "type", "Relation", "relation") or default_type
        desc = _pick(row, "Description", "description", "Evidence", "evidence")
        if not src or not tgt:
            continue
        rels.append(
            {
                "source": src,
                "target": tgt,
                "type": rtype,
                "description": desc,
                "source_ref": source_file,
            }
        )
    return rels


def load_directory(input_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    entities: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []

    for path in sorted(input_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".csv", ".tsv", ".tab", ".txt"}:
            continue
        name_u = path.name.upper()
        rows = _read_table(path)
        if not rows:
            continue

        if "REL" in name_u or "PAIR" in name_u or "ASSOC" in name_u:
            relationships.extend(parse_relationship_rows(rows, path.name))
        elif "SMTS" in name_u and "SMHB" in name_u:
            relationships.extend(parse_relationship_rows(rows, path.name, default_type="TREATS"))
        else:
            entities.extend(parse_entity_rows(rows, path.name))

    # Deduplicate entities by id
    by_id: dict[str, dict[str, Any]] = {}
    for e in entities:
        by_id[e["id"]] = e
    return list(by_id.values()), relationships


def build_sample_graph() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Synthetic SymMap-shaped graph for CI / demos (no external download)."""
    entities: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []

    symptoms = [
        ("SMTS000001", "頭痛", "headache"),
        ("SMTS000002", "眩暈", "vertigo"),
        ("SMTS000003", "失眠", "insomnia"),
    ]
    for i in range(4, 36):
        symptoms.append((f"SMTS{i:06d}", f"示例症狀{i}", f"symptom_{i}"))

    for sid, zh, en in symptoms:
        entities.append(
            {
                "id": sid,
                "type": "Symptom",
                "name": zh,
                "name_en": en,
                "symmap_component": "SMTS",
                "source_ref": "import_symmap_kg.sample",
            }
        )

    for i in range(1, 36):
        hid = f"SMHB{i:06d}"
        entities.append(
            {
                "id": hid,
                "type": "Herb",
                "name": f"示例藥材{i}",
                "name_en": f"herb_{i}",
                "symmap_component": "SMHB",
                "source_ref": "import_symmap_kg.sample",
            }
        )

    for i in range(1, 36):
        iid = f"SMIT{i:06d}"
        entities.append(
            {
                "id": iid,
                "type": "Ingredient",
                "name": f"成分{i}",
                "name_en": f"ingredient_{i}",
                "symmap_component": "SMIT",
                "source_ref": "import_symmap_kg.sample",
            }
        )

    for i in range(1, 26):
        tid = f"SMTT{i:06d}"
        entities.append(
            {
                "id": tid,
                "type": "Target",
                "name": f"GENE{i}",
                "name_en": f"protein_target_{i}",
                "symmap_component": "SMTT",
                "source_ref": "import_symmap_kg.sample",
            }
        )

    for i in range(1, 26):
        did = f"SMDE{i:06d}"
        entities.append(
            {
                "id": did,
                "type": "Disease",
                "name": f"疾病{i}",
                "name_en": f"disease_{i}",
                "symmap_component": "SMDE",
                "source_ref": "import_symmap_kg.sample",
            }
        )

    for i in range(1, 11):
        entities.append(
            {
                "id": f"SMYS{i:06d}",
                "type": "Syndrome",
                "name": f"證候{i}",
                "name_en": f"syndrome_{i}",
                "symmap_component": "SMYS",
                "source_ref": "import_symmap_kg.sample",
            }
        )

    # Herb -> Symptom TREATS
    for i in range(1, 36):
        hid = f"SMHB{i:06d}"
        sid = f"SMTS{(i % 35) + 1:06d}"
        if sid == "SMTS000036":
            sid = "SMTS000001"
        relationships.append(
            {
                "source": hid,
                "target": sid,
                "type": "TREATS",
                "description": "sample TREATS",
                "source_ref": "import_symmap_kg.sample",
            }
        )

    # Herb -> Ingredient CONTAINS
    for i in range(1, 36):
        relationships.append(
            {
                "source": f"SMHB{i:06d}",
                "target": f"SMIT{i:06d}",
                "type": "CONTAINS",
                "description": "sample CONTAINS",
                "source_ref": "import_symmap_kg.sample",
            }
        )

    # Ingredient -> Target TARGETS
    for i in range(1, 26):
        relationships.append(
            {
                "source": f"SMIT{i:06d}",
                "target": f"SMTT{i:06d}",
                "type": "TARGETS",
                "description": "sample TARGETS",
                "source_ref": "import_symmap_kg.sample",
            }
        )

    # Target -> Disease ASSOCIATED_WITH
    for i in range(1, 26):
        relationships.append(
            {
                "source": f"SMTT{i:06d}",
                "target": f"SMDE{i:06d}",
                "type": "ASSOCIATED_WITH",
                "description": "sample ASSOCIATED_WITH",
                "source_ref": "import_symmap_kg.sample",
            }
        )

    # Symptom -> Disease CORRELATES_WITH
    for i in range(1, 26):
        relationships.append(
            {
                "source": f"SMTS{(i % 3) + 1:06d}",
                "target": f"SMDE{i:06d}",
                "type": "CORRELATES_WITH",
                "description": "sample CORRELATES_WITH",
                "source_ref": "import_symmap_kg.sample",
            }
        )

    # Extra edges for density
    for i in range(1, 16):
        relationships.append(
            {
                "source": f"SMMS{i:06d}",
                "target": f"SMTS{i:06d}",
                "type": "MAPS_TO",
                "description": "sample MM->TCM symptom MAPS_TO",
                "source_ref": "import_symmap_kg.sample",
            }
        )

    # Add SMMS nodes referenced above
    for i in range(1, 16):
        entities.append(
            {
                "id": f"SMMS{i:06d}",
                "type": "Symptom",
                "name": f"MM症狀{i}",
                "name_en": f"mm_symptom_{i}",
                "symmap_component": "SMMS",
                "source_ref": "import_symmap_kg.sample",
            }
        )

    return entities, relationships


def main() -> None:
    parser = argparse.ArgumentParser(description="Import SymMap CSV/TSV into graph JSON.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing SymMap export tables (.csv/.tsv)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("data/graph/symmap_entities.json"),
        help="Output JSON path",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Generate synthetic SymMap-shaped data (no input files required)",
    )
    args = parser.parse_args()

    if args.sample:
        entities, relationships = build_sample_graph()
    elif args.input_dir:
        entities, relationships = load_directory(args.input_dir)
        if not entities and not relationships:
            raise SystemExit(f"No rows parsed from {args.input_dir}")
    else:
        parser.error("Provide --input-dir or --sample")

    payload = {"entities": entities, "relationships": relationships}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(entities)} entities, {len(relationships)} relationships -> {args.output}")


if __name__ == "__main__":
    main()
