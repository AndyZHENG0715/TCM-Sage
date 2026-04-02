#!/usr/bin/env python3
"""
Download SymMap v2.0 component tables from symmap.org and fetch curated herb→TCM_symptom
pairwise rows via the same HTTP API used by the website (/related_components/).

Outputs under data/graph/symmap/raw/:
  - Official .xlsx (entity description tables)
  - rel_smhb_smts.tsv — edges suitable for import_symmap_kg.py (TREATS: SMHB → SMTS)

Respect the remote service: small delay between API calls. Re-run is safe (idempotent files).
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote

import requests

V20_BASE = "http://www.symmap.org/static/download/V2.0/"
RELATED_URL = "http://www.symmap.org/related_components/"

# (URL filename fragment after V20_BASE, local filename)
V20_ENTITY_XLSX: tuple[tuple[str, str], ...] = (
    ("SymMap%20v2.0%2C%20SMHB%20file.xlsx", "SymMap_v2.0_SMHB_file.xlsx"),
    ("SymMap%20v2.0%2C%20SMTS%20file.xlsx", "SymMap_v2.0_SMTS_file.xlsx"),
    ("SymMap%20v2.0%2C%20SMMS%20file.xlsx", "SymMap_v2.0_SMMS_file.xlsx"),
    ("SymMap%20v2.0%2C%20SMIT%20file.xlsx", "SymMap_v2.0_SMIT_file.xlsx"),
    ("SymMap%20v2.0%2C%20SMTT%20file.xlsx", "SymMap_v2.0_SMTT_file.xlsx"),
    ("SymMap%20v2.0%2C%20SMDE%20file.xlsx", "SymMap_v2.0_SMDE_file.xlsx"),
    ("SymMap%20v2.0%2C%20SMSY%20file.xlsx", "SymMap_v2.0_SMSY_file.xlsx"),
)


def _read_herb_ids(raw_dir: Path) -> list[str]:
    try:
        import openpyxl
    except ImportError as e:
        raise SystemExit("openpyxl required. pip install openpyxl") from e

    smhb = raw_dir / "SymMap_v2.0_SMHB_file.xlsx"
    if not smhb.exists():
        raise SystemExit(f"Missing {smhb}; run with --download-xlsx first")

    wb = openpyxl.load_workbook(smhb, read_only=True, data_only=True)
    ws = wb.active
    rows = ws.iter_rows(values_only=True)
    header = next(rows)
    idx = {str(h).strip(): i for i, h in enumerate(header) if h is not None}
    hid = idx.get("Herb_id")
    sup = idx.get("Suppress")
    if hid is None:
        raise SystemExit("SMHB xlsx missing Herb_id column")
    out: list[str] = []
    for row in rows:
        if sup is not None and sup < len(row) and row[sup] not in (None, "", 0, "0"):
            continue
        raw_id = row[hid] if hid < len(row) else None
        if raw_id is None or raw_id == "":
            continue
        n = int(float(raw_id))
        out.append(f"SMHB{n:06d}")
    return out


def download_xlsx(raw_dir: Path, timeout: float = 120.0) -> list[Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for part, local_name in V20_ENTITY_XLSX:
        url = f"{V20_BASE}{part}"
        dest = raw_dir / local_name
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        dest.write_bytes(r.content)
        written.append(dest)
        print(f"Wrote {dest.name} ({len(r.content)} bytes) <- {unquote(part)}")
    return written


def fetch_herb_symptom_edges(
    raw_dir: Path,
    herb_ids: Iterable[str],
    delay_s: float,
    timeout: float = 60.0,
) -> list[tuple[str, str]]:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "TCM-Sage/1.0 (SymMap bulk fetch; +https://github.com/)",
            "Accept": "application/json,text/plain,*/*",
        }
    )
    edges: set[tuple[str, str]] = set()
    for i, hid in enumerate(herb_ids):
        resp = session.post(
            RELATED_URL,
            data={"rrid": hid, "table_name": "TCM_symptom", "filter": "0"},
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = json.loads(resp.text)
        for row in payload.get("data") or []:
            sid = row.get("TCM_symptom_id") or row.get("tcm_symptom_id")
            if not sid:
                continue
            edges.add((hid, str(sid)))
        if delay_s > 0:
            time.sleep(delay_s)
        if (i + 1) % 50 == 0:
            print(f"  ... herbs processed: {i + 1}, edges so far: {len(edges)}")
    return sorted(edges)


def write_rel_tsv(path: Path, edges: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["SMHB_ID", "SMTS_ID"], delimiter="\t")
        w.writeheader()
        for h, s in edges:
            w.writerow({"SMHB_ID": h, "SMTS_ID": s})
    print(f"Wrote {len(edges)} TREATS edges -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SymMap v2.0 data and fetch herb→symptom pairs.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/graph/symmap/raw"),
        help="Directory for xlsx + rel_smhb_smts.tsv",
    )
    parser.add_argument(
        "--download-xlsx",
        action="store_true",
        help="Download official v2.0 component xlsx files from symmap.org",
    )
    parser.add_argument(
        "--fetch-edges",
        action="store_true",
        help="POST /related_components/ for each herb and write rel_smhb_smts.tsv",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.25,
        help="Seconds between API calls (default 0.25)",
    )
    parser.add_argument(
        "--max-herbs",
        type=int,
        default=0,
        help="Limit herbs for testing (0 = all)",
    )
    args = parser.parse_args()
    raw_dir = args.raw_dir.resolve()

    if args.download_xlsx:
        download_xlsx(raw_dir)

    if args.fetch_edges:
        herbs = _read_herb_ids(raw_dir)
        if args.max_herbs:
            herbs = herbs[: args.max_herbs]
        print(f"Fetching TCM_symptom neighbors for {len(herbs)} herbs ...")
        edges = fetch_herb_symptom_edges(raw_dir, herbs, delay_s=args.delay)
        write_rel_tsv(raw_dir / "rel_smhb_smts.tsv", edges)

    if not args.download_xlsx and not args.fetch_edges:
        parser.error("Specify --download-xlsx and/or --fetch-edges")


if __name__ == "__main__":
    main()
