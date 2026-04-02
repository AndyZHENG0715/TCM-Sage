# SymMap knowledge graph (TCM-Sage)

This folder holds **SymMap-oriented** graph assets: raw bulk exports and the merged JSON consumed by hybrid retrieval.

## Layout

| Path | Purpose |
|------|---------|
| `raw/` | Original SymMap download (TSV/CSV). Do not edit; replace when you refresh the dataset. |
| `symmap_entities.json` | Merged graph: `{"entities": [...], "relationships": [...]}` — produced by `scripts/import_symmap_kg.py`. |

### Official SymMap v2.0 (symmap.org)

The [download page](http://www.symmap.org/download/) ships **component description tables as `.xlsx` only**; curated **herb→TCM_symptom** pairwise rows are exposed through the site API (`POST /related_components/`), not as a separate static TSV.

One-shot: download xlsx + fetch edges + build JSON:

```text
venv\Scripts\python.exe scripts\fetch_symmap_v2.py --download-xlsx --fetch-edges --raw-dir data\graph\symmap\raw
venv\Scripts\python.exe scripts\import_symmap_kg.py --input-dir data\graph\symmap\raw -o data\graph\symmap\symmap_entities.json
```

`--fetch-edges` issues ~700 polite POSTs (see `--delay`). For a quick test: `--fetch-edges --max-herbs 20`.

### Import only (you already have files in `raw/`)

```text
venv\Scripts\python.exe scripts\import_symmap_kg.py --input-dir data\graph\symmap\raw -o data\graph\symmap\symmap_entities.json
```

Synthetic demo (no download):

```text
venv\Scripts\python.exe scripts\import_symmap_kg.py --sample -o data\graph\symmap\symmap_entities.json
```

## Provenance (fill in when you ingest real data)

- **SymMap release / version:**
- **Download date:**
- **Files included:** (list the tables you placed in `raw/`)
- **License / citation:** Wu Y, Zhang F, Yang K, et al. *SymMap: an integrative database of traditional Chinese medicine enhanced by symptom mapping.* Nucleic Acids Research. 2019;47(D1):D1110-D1117. [NAR](https://academic.oup.com/nar/article/47/D1/D1110/5150228)

## Note

Quick local fixtures remain under `data/symmap_sample/` for adapter tests; production-like runs should use `raw/` + import as above.
