#!/usr/bin/env python3
"""
Import SymMap-style CSV/TSV exports into TCM-Sage `entities.json` graph format.

See `.planning/phases/02-standard-kg-integration/SYMMAP_MAPPING.md` for file layout.

Examples:
  python scripts/import_symmap_kg.py --sample -o data/graph/symmap/symmap_entities.json
  python scripts/import_symmap_kg.py --input-dir data/graph/symmap/raw -o data/graph/symmap/symmap_entities.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


def _read_xlsx(path: Path) -> list[dict[str, str]]:
    """Load first sheet of an Excel file as list of dict rows (string values)."""
    try:
        import openpyxl
    except ImportError as e:
        raise SystemExit(
            "Reading .xlsx requires openpyxl. Install with: venv\\Scripts\\python.exe -m pip install openpyxl"
        ) from e

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    rows_iter = ws.iter_rows(values_only=True)
    try:
        header_row = next(rows_iter)
    except StopIteration:
        return []
    headers: list[str] = []
    for h in header_row:
        headers.append(str(h).strip() if h is not None else "")
    out: list[dict[str, str]] = []
    for row in rows_iter:
        d: dict[str, str] = {}
        for i, key in enumerate(headers):
            if not key:
                continue
            val = row[i] if i < len(row) else None
            if val is None:
                d[key] = ""
            elif isinstance(val, float) and val.is_integer():
                d[key] = str(int(val))
            else:
                d[key] = str(val).strip()
        out.append(d)
    return out


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
    return [{k: (v or "").strip() for k, v in row.items() if k is not None} for row in reader]


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


def _symmap_prefix(entity_id: str) -> str | None:
    """SymMap 2.0 style: SMTSxxxxx, SMHBxxxxx, etc."""
    if not entity_id:
        return None
    m = re.match(r"^(SM[A-Z]{2})", entity_id.upper())
    return m.group(1) if m else None


def _legacy_layer(entity_id: str) -> str | None:
    """
    Legacy export IDs: SM00001 (symptom), HM00001 (herb), IM/TM/MM.
    Returns a short token: SMTS, SMHB, SMIT, SMTT, SMDE, SMMS.
    """
    if not entity_id:
        return None
    u = entity_id.upper()
    if re.match(r"^SM\d", u):
        return "SMTS"
    if re.match(r"^HM\d", u):
        return "SMHB"
    if re.match(r"^IM\d", u):
        return "SMIT"
    if re.match(r"^TM\d", u):
        return "SMTT"
    if re.match(r"^MM\d", u):
        return "SMDE"
    return None


def _normalize_symmap_id(entity_id: str) -> str:
    """
    Normalize SymMap IDs to stable zero-padded forms when possible.

    - v2 prefixes: SMHB/SMTS/SMMS/SMIT/SMTT/SMDE/SMSY/SMYS -> 6 digits
    - legacy prefixes: SM/HM/IM/TM/MM -> keep current shape
    """
    if not entity_id:
        return entity_id
    raw = entity_id.strip().upper()
    m_v2 = re.match(r"^(SM[A-Z]{2})(\d+)$", raw)
    if m_v2:
        return f"{m_v2.group(1)}{int(m_v2.group(2)):06d}"
    return raw


PREFIX_TO_ENTITY_TYPE: dict[str, str] = {
    "SMTS": "Symptom",
    "SMMS": "Symptom",
    "SMHB": "Herb",
    "SMIT": "Ingredient",
    "SMTT": "Target",
    "SMDE": "Disease",
    "SMYS": "Syndrome",
    "SMSY": "Syndrome",
}


def _entity_id_from_row(row: dict[str, str]) -> str:
    """First non-empty value from common SymMap / legacy ID columns."""
    return _pick(
        row,
        "SymMap_ID",
        "symmap_id",
        "SMTS_ID",
        "SMMS_ID",
        "SMHB_ID",
        "SMIT_ID",
        "SMTT_ID",
        "SMDE_ID",
        "SMYS_ID",
        "SM_ID",
        "HM_ID",
        "IM_ID",
        "TM_ID",
        "MM_ID",
        "Herb_id",
        "TCM_symptom_id",
        "MM_symptom_id",
        "Ingredient_id",
        "Target_id",
        "Disease_id",
        "Syndrome_id",
        "ID",
        "id",
    )


def entity_hint_from_filename(path: Path) -> tuple[str, str] | None:
    """
    (entity_type, symmap_component) from filename when IDs lack SMxx prefix.
    """
    n = path.stem.lower()
    # SymMap v2.0 bulk filenames like "SymMap v2.0, SMHB file"
    if "smhb" in n:
        return ("Herb", "SMHB")
    if "smts" in n:
        return ("Symptom", "SMTS")
    if "smms" in n:
        return ("Symptom", "SMMS")
    if "smit" in n:
        return ("Ingredient", "SMIT")
    if "smtt" in n:
        return ("Target", "SMTT")
    if "smde" in n:
        return ("Disease", "SMDE")
    if "smsy" in n:
        return ("Syndrome", "SMSY")
    if "symptom" in n and "syndrome" not in n:
        return ("Symptom", "SMTS")
    if "herb" in n:
        return ("Herb", "SMHB")
    if "ingredient" in n:
        return ("Ingredient", "SMIT")
    if "target" in n:
        return ("Target", "SMTT")
    if "disease" in n:
        return ("Disease", "SMDE")
    if "syndrome" in n:
        return ("Syndrome", "SMSY")
    return None


def parse_entity_rows(
    rows: list[dict[str, str]],
    source_file: str,
    type_hint: tuple[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Map arbitrary SymMap-like rows to entity dicts."""
    entities: list[dict[str, Any]] = []
    for row in rows:
        if _pick(row, "Suppress", "suppress").lower() in {"1", "true", "yes"}:
            continue
        eid = _normalize_symmap_id(_entity_id_from_row(row))
        if not eid:
            continue

        prefix = _symmap_prefix(eid) or _legacy_layer(eid)
        if prefix:
            etype = PREFIX_TO_ENTITY_TYPE.get(prefix, "Symptom")
            component = prefix
        elif type_hint:
            etype, component = type_hint
            if eid.isdigit():
                eid = f"{component}{int(eid):06d}"
                prefix = component
        else:
            etype, component = "Symptom", "SMTS"

        name = _pick(
            row,
            "SMTS_Chinese_Name",
            "SMHB_Chinese_Name",
            "SMYS_Chinese_Name",
            "Chinese_name",
            "TCM_symptom_name",
            "MM_symptom_name",
            "Molecule_name",
            "Syndrome_name",
            "Disease_name",
            "Gene_symbol",
            "Name_CN",
            "Chinese",
            "SM_Name",
            "HM_Name",
            "IM_Name",
            "TM_Name",
            "MM_Name",
            "Name",
            "name",
            "Herb_Name",
            "compound_name",
            "SMDE_Name",
        )
        name_en = _pick(
            row,
            "SMTS_English_Name",
            "SMHB_English_Name",
            "SMYS_English_Name",
            "Name_EN",
            "English",
            "name_en",
            "Name_en",
        )
        pinyin = _pick(
            row,
            "SMTS_Pinyin",
            "SMHB_Pinyin",
            "SM_Pinyin",
            "HM_Pinyin",
            "Symptom_pinYin",
            "Symptom_pinyin",
            "Syndrome_Pinyin",
            "Pinyin_name",
            "Pinyin",
            "pinyin",
            "PINYIN",
        )
        desc = _pick(
            row,
            "Description",
            "description",
            "Function",
            "function",
            "Symptom_definition",
            "MM_symptom_definition",
            "Syndrome_definition",
            "Disease_definition",
            "SMMS_Name",
        )

        if not name and name_en:
            name = name_en
        if not name:
            name = eid

        ent: dict[str, Any] = {
            "id": eid,
            "type": etype,
            "name": name,
            "name_en": name_en,
            "pinyin": pinyin,
            "description": desc,
            "source_ref": source_file,
            "symmap_component": component,
        }
        if prefix == "SMMS" or (component == "SMMS"):
            ent["symmap_component"] = "SMMS"
        if component == "SMSY":
            ent["symmap_component"] = "SMSY"
        entities.append(ent)
    return entities


def parse_relationship_rows(
    rows: list[dict[str, str]],
    source_file: str,
    default_type: str = "ASSOCIATED_WITH",
    source_key: str | None = None,
    target_key: str | None = None,
) -> list[dict[str, Any]]:
    rels: list[dict[str, Any]] = []
    for row in rows:
        if source_key and target_key:
            src = _normalize_symmap_id(_pick(row, source_key))
            tgt = _normalize_symmap_id(_pick(row, target_key))
        else:
            src = _normalize_symmap_id(_pick(
                row,
                "Source",
                "source",
                "Head",
                "head",
                "ID1",
                "Herb_ID",
                "SMHB_ID",
                "HM_ID",
                "IM_ID",
                "TM_ID",
                "SMTS_ID",
            ))
            tgt = _normalize_symmap_id(_pick(
                row,
                "Target",
                "target",
                "Tail",
                "tail",
                "ID2",
                "Symptom_ID",
                "SMDE_ID",
                "SM_ID",
                "MM_ID",
                "SMTT_ID",
            ))
            if not src or not tgt:
                src = _normalize_symmap_id(_pick(row, "SMHB_ID", "Herb_ID", "HM_ID", "IM_ID", "TM_ID"))
                tgt = _normalize_symmap_id(
                    _pick(row, "SMTS_ID", "Symptom_ID", "SM_ID", "MM_ID", "SMTT_ID", "SMIT_ID")
                )
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


def parse_relationship_file(path: Path, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """
    Apply SYMMAP_MAPPING edge types and endpoint order per relationship file name.
    Herb treats TCM symptom: TREATS Herb -> Symptom.
    """
    stem = path.stem.lower()
    name = path.name

    if "rel_sm_hm" in stem or ("sm" in stem and "hm" in stem and "rel" in stem):
        return parse_relationship_rows(rows, name, "TREATS", "HM_ID", "SM_ID")

    if "rel_hm_im" in stem or ("hm" in stem and "im" in stem and "rel" in stem):
        return parse_relationship_rows(rows, name, "CONTAINS", "HM_ID", "IM_ID")

    if "rel_im_tm" in stem or ("im" in stem and "tm" in stem and "rel" in stem):
        return parse_relationship_rows(rows, name, "TARGETS", "IM_ID", "TM_ID")

    if "rel_tm_mm" in stem or ("tm" in stem and "mm" in stem and "rel" in stem):
        return parse_relationship_rows(rows, name, "ASSOCIATED_WITH", "TM_ID", "MM_ID")

    if "rel_sm_mm" in stem or ("sm" in stem and "mm" in stem and "rel" in stem):
        return parse_relationship_rows(rows, name, "CORRELATES_WITH", "SM_ID", "MM_ID")

    # SymMap 2.0 herb → TCM symptom (e.g. rel_smhb_smts.tsv from scripts/fetch_symmap_v2.py)
    if "rel_smhb_smts" in stem or ("smhb" in stem and "smts" in stem and "rel" in stem):
        return parse_relationship_rows(rows, name, "TREATS", "SMHB_ID", "SMTS_ID")
    if "smt" in stem and "smhb" in stem:
        return parse_relationship_rows(rows, name, "TREATS", "SMHB_ID", "SMTS_ID")
    if "smhb" in stem and "smit" in stem:
        return parse_relationship_rows(rows, name, "CONTAINS", "SMHB_ID", "SMIT_ID")
    if "smit" in stem and "smtt" in stem:
        return parse_relationship_rows(rows, name, "TARGETS", "SMIT_ID", "SMTT_ID")
    if "smtt" in stem and "smde" in stem:
        return parse_relationship_rows(rows, name, "ASSOCIATED_WITH", "SMTT_ID", "SMDE_ID")

    return parse_relationship_rows(rows, name)


def _is_relationship_file(path: Path) -> bool:
    name_u = path.name.upper()
    stem_l = path.stem.lower()
    if "REL" in name_u or "PAIR" in name_u or "ASSOC" in name_u:
        return True
    if stem_l.startswith("rel_"):
        return True
    return False


def load_directory(input_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    entities: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []

    for path in sorted(input_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() == ".xlsx":
            stem_l = path.stem.lower()
            if "key" in stem_l and "file" in stem_l:
                continue
            rows = _read_xlsx(path)
        elif path.suffix.lower() in {".csv", ".tsv", ".tab", ".txt"}:
            rows = _read_table(path)
        else:
            continue
        if not rows:
            continue

        if _is_relationship_file(path):
            relationships.extend(parse_relationship_file(path, rows))
        else:
            hint = entity_hint_from_filename(path)
            entities.extend(parse_entity_rows(rows, path.name, type_hint=hint))

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
        default=Path("data/graph/symmap/symmap_entities.json"),
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
