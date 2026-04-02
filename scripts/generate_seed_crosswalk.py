#!/usr/bin/env python3
"""
Generate a review-first seed crosswalk between Neijing KG and SymMap KG.

This script does NOT mutate graph merge logic. It only writes:
1) pending candidate rows for human review
2) an empty approved table template
3) a markdown preview for fast inspection
"""

from __future__ import annotations

import argparse
import csv
import json
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


TYPE_MAP = {
    "symptom": "Symptom",
    "sym,ptom": "Symptom",
    "¶symptom": "Symptom",
    "herb": "Herb",
}


@dataclass
class Entity:
    entity_id: str
    entity_type: str
    name: str
    normalized_name: str


def normalize_name(raw: str) -> str:
    text = unicodedata.normalize("NFKC", (raw or "").strip()).lower()
    # Keep alnum and CJK only; remove punctuation/spacing noise.
    cleaned = []
    for ch in text:
        if ch.isalnum():
            cleaned.append(ch)
            continue
        code = ord(ch)
        if 0x4E00 <= code <= 0x9FFF:
            cleaned.append(ch)
    return "".join(cleaned)


def normalize_type(raw: str) -> str | None:
    if not raw:
        return None
    return TYPE_MAP.get(raw.strip().lower())


def load_neijing_entities(path: Path) -> list[Entity]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entities: list[Entity] = []
    for row in payload.get("entities", []):
        entity_type = normalize_type(str(row.get("type", "")))
        if entity_type not in {"Symptom", "Herb"}:
            continue
        name = str(row.get("mention") or row.get("name") or "").strip()
        if not name:
            continue
        normalized_name = normalize_name(name)
        if not normalized_name:
            continue
        entities.append(
            Entity(
                entity_id=str(row.get("id", "")),
                entity_type=entity_type,
                name=name,
                normalized_name=normalized_name,
            )
        )
    return entities


def load_symmap_entities(path: Path) -> list[Entity]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entities: list[Entity] = []
    for row in payload.get("entities", []):
        entity_type = normalize_type(str(row.get("type", "")))
        if entity_type not in {"Symptom", "Herb"}:
            continue
        name = str(row.get("name") or row.get("mention") or "").strip()
        if not name:
            continue
        normalized_name = normalize_name(name)
        if not normalized_name:
            continue
        entities.append(
            Entity(
                entity_id=str(row.get("id", "")),
                entity_type=entity_type,
                name=name,
                normalized_name=normalized_name,
            )
        )
    return entities


def group_by_name(entities: Iterable[Entity]) -> dict[tuple[str, str], list[Entity]]:
    grouped: dict[tuple[str, str], list[Entity]] = defaultdict(list)
    for e in entities:
        grouped[(e.entity_type, e.normalized_name)].append(e)
    return grouped


def write_pending_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "candidate_id",
        "entity_type",
        "normalized_name",
        "canonical_symmap_id",
        "symmap_name",
        "neijing_entity_id",
        "neijing_name",
        "neijing_frequency",
        "match_rule",
        "confidence",
        "status",
        "reviewer",
        "reviewed_at",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_approved_template(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "candidate_id",
        "entity_type",
        "normalized_name",
        "canonical_symmap_id",
        "symmap_name",
        "neijing_entity_id",
        "neijing_name",
        "match_rule",
        "confidence",
        "approved_by",
        "approved_at",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def write_review_preview(rows: list[dict[str, str]], path: Path) -> None:
    symptom_rows = [r for r in rows if r["entity_type"] == "Symptom"]
    herb_rows = [r for r in rows if r["entity_type"] == "Herb"]

    lines = [
        "# Seed crosswalk review preview",
        "",
        "Status: pending human review (do not ingest as L0 until approved)",
        "",
        f"Total candidates: {len(rows)}",
        f"- Symptom: {len(symptom_rows)}",
        f"- Herb: {len(herb_rows)}",
        "",
        "## Symptom candidates",
        "",
        "| candidate_id | neijing_name | symmap_name | neijing_frequency |",
        "|---|---|---|---:|",
    ]
    for r in symptom_rows:
        lines.append(
            f"| {r['candidate_id']} | {r['neijing_name']} | {r['symmap_name']} | {r['neijing_frequency']} |"
        )

    lines.extend(
        [
            "",
            "## Herb candidates",
            "",
            "| candidate_id | neijing_name | symmap_name | neijing_frequency |",
            "|---|---|---|---:|",
        ]
    )
    for r in herb_rows:
        lines.append(
            f"| {r['candidate_id']} | {r['neijing_name']} | {r['symmap_name']} | {r['neijing_frequency']} |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manual_source_lists(
    neijing_entities: list[Entity],
    symmap_entities: list[Entity],
    out_dir: Path,
    top_symptom: int,
    top_herb: int,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    neijing_freq = Counter((e.entity_type, e.normalized_name, e.name, e.entity_id) for e in neijing_entities)
    symmap_freq = Counter((e.entity_type, e.normalized_name, e.name, e.entity_id) for e in symmap_entities)

    def pick_rows(counter: Counter, target_type: str, limit: int) -> list[tuple]:
        rows = [(*k, v) for k, v in counter.items() if k[0] == target_type]
        rows.sort(key=lambda x: x[-1], reverse=True)
        return rows[:limit]

    neijing_rows = pick_rows(neijing_freq, "Symptom", top_symptom) + pick_rows(neijing_freq, "Herb", top_herb)
    symmap_rows = pick_rows(symmap_freq, "Symptom", top_symptom) + pick_rows(symmap_freq, "Herb", top_herb)

    neijing_path = out_dir / "seed_crosswalk_manual_neijing.csv"
    with neijing_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["entity_type", "neijing_entity_id", "neijing_name", "normalized_name", "frequency"],
        )
        writer.writeheader()
        for entity_type, normalized_name, name, entity_id, freq in neijing_rows:
            writer.writerow(
                {
                    "entity_type": entity_type,
                    "neijing_entity_id": entity_id,
                    "neijing_name": name,
                    "normalized_name": normalized_name,
                    "frequency": freq,
                }
            )

    symmap_path = out_dir / "seed_crosswalk_manual_symmap.csv"
    with symmap_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["entity_type", "canonical_symmap_id", "symmap_name", "normalized_name", "frequency"],
        )
        writer.writeheader()
        for entity_type, normalized_name, name, entity_id, freq in symmap_rows:
            writer.writerow(
                {
                    "entity_type": entity_type,
                    "canonical_symmap_id": entity_id,
                    "symmap_name": name,
                    "normalized_name": normalized_name,
                    "frequency": freq,
                }
            )
    return neijing_path, symmap_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate pending seed crosswalk candidates.")
    parser.add_argument(
        "--neijing",
        default="data/graph/entities_partial.json",
        help="Path to Neijing extracted KG JSON",
    )
    parser.add_argument(
        "--symmap",
        default="data/graph/symmap/symmap_entities.json",
        help="Path to SymMap KG JSON",
    )
    parser.add_argument(
        "--out-dir",
        default="data/graph/crosswalk",
        help="Output directory for seed crosswalk artifacts",
    )
    parser.add_argument("--top-symptom", type=int, default=30, help="Top symptom candidates by Neijing frequency")
    parser.add_argument("--top-herb", type=int, default=20, help="Top herb candidates by Neijing frequency")
    args = parser.parse_args()

    neijing_entities = load_neijing_entities(Path(args.neijing))
    symmap_entities = load_symmap_entities(Path(args.symmap))
    neijing_freq = Counter((e.entity_type, e.normalized_name) for e in neijing_entities)
    neijing_grouped = group_by_name(neijing_entities)
    symmap_grouped = group_by_name(symmap_entities)

    by_type_rows: dict[str, list[dict[str, str]]] = {"Symptom": [], "Herb": []}
    for key, freq in neijing_freq.items():
        entity_type, normalized_name = key
        if key not in symmap_grouped:
            continue

        # Representative picks: first SymMap entity as canonical, first Neijing entity as representative.
        symmap_rep = symmap_grouped[key][0]
        neijing_rep = neijing_grouped[key][0]
        candidate_id = f"{entity_type.lower()}::{normalized_name}"
        by_type_rows[entity_type].append(
            {
                "candidate_id": candidate_id,
                "entity_type": entity_type,
                "normalized_name": normalized_name,
                "canonical_symmap_id": symmap_rep.entity_id,
                "symmap_name": symmap_rep.name,
                "neijing_entity_id": neijing_rep.entity_id,
                "neijing_name": neijing_rep.name,
                "neijing_frequency": str(freq),
                "match_rule": "L1_exact_normalized_name_same_type",
                "confidence": "medium",
                "status": "pending",
                "reviewer": "",
                "reviewed_at": "",
                "notes": "",
            }
        )

    by_type_rows["Symptom"].sort(key=lambda r: int(r["neijing_frequency"]), reverse=True)
    by_type_rows["Herb"].sort(key=lambda r: int(r["neijing_frequency"]), reverse=True)

    pending_rows = by_type_rows["Symptom"][: args.top_symptom] + by_type_rows["Herb"][: args.top_herb]

    out_dir = Path(args.out_dir)
    pending_path = out_dir / "seed_crosswalk_pending.csv"
    approved_path = out_dir / "seed_crosswalk_approved.csv"
    preview_path = out_dir / "seed_crosswalk_review.md"

    write_pending_csv(pending_rows, pending_path)
    write_approved_template(approved_path)

    manual_neijing_path = None
    manual_symmap_path = None
    if pending_rows:
        write_review_preview(pending_rows, preview_path)
    else:
        manual_neijing_path, manual_symmap_path = write_manual_source_lists(
            neijing_entities,
            symmap_entities,
            out_dir=out_dir,
            top_symptom=args.top_symptom,
            top_herb=args.top_herb,
        )
        preview_lines = [
            "# Seed crosswalk review preview",
            "",
            "Status: BLOCKED_FOR_MANUAL_REVIEW",
            "",
            "No L1 exact-name candidates found between current Neijing and SymMap datasets.",
            "Likely cause: current SymMap graph is sample/synthetic naming (e.g., 示例*).",
            "",
            "Manual seed files generated:",
            f"- `{manual_neijing_path}`",
            f"- `{manual_symmap_path}`",
            "",
            "Action required:",
            "1. Review and author a small approved seed table manually (`seed_crosswalk_approved.csv`).",
            "2. Prefer real SymMap export before large-scale L0 construction.",
            "",
        ]
        preview_path.write_text("\n".join(preview_lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "pending_path": str(pending_path),
                "approved_path": str(approved_path),
                "preview_path": str(preview_path),
                "manual_neijing_path": str(manual_neijing_path) if manual_neijing_path else "",
                "manual_symmap_path": str(manual_symmap_path) if manual_symmap_path else "",
                "pending_count": len(pending_rows),
                "symptom_count": len([r for r in pending_rows if r["entity_type"] == "Symptom"]),
                "herb_count": len([r for r in pending_rows if r["entity_type"] == "Herb"]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
