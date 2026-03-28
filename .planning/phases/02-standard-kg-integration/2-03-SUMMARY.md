---
phase: 2-standard-kg-integration
plan: "2-03"
subsystem: retrieval-config
tags: [symmap, hybrid-retrieval, graph, verification]

requires:
  - phase: 2-standard-kg-integration
    provides: "symmap_entities.json and graph_builder compatibility"
provides:
  - "GRAPH_DATA_PATH default aligned with SymMap JSON"
  - "scripts/verify_symmap_retrieval.py hybrid graph path evidence"
affects:
  - "Streamlit/API backend graph resolution and HybridRetriever defaults"

tech-stack:
  added: []
  patterns:
    - "GRAPH_DATA_PATH env overrides GRAPH_DATA_DEFAULT_RELATIVE"
    - "_search_graph_documents uses same graph file as create_hybrid_retriever default"

key-files:
  created:
    - "scripts/verify_symmap_retrieval.py"
  modified:
    - "src/config.py"
    - "src/ui_backend.py"
    - "src/retriever.py"
    - "src/test_hybrid_retriever.py"
    - ".env.example"

key-decisions:
  - "Canonical constant is GRAPH_DATA_PATH (replaces GRAPH_DATA_FILE); retriever and tests import GRAPH_DATA_PATH"
  - "Entity-resolution L0: no auto-merge logic added; full SymMap export and human-reviewed seed crosswalk remain prerequisites for trustable merge (see ENTITY_RESOLUTION.md)"

requirements-completed: ["Task 2.3", "Task 2.4"]

duration: 25min
completed: 2026-03-28
---

# Phase 2 Plan 2-03: Pipeline Integration & Verification Summary

**SymMap-shaped KG is the default graph file (`GRAPH_DATA_PATH`), hybrid retrieval exercises `_search_graph_documents` for 頭痛 with 14 graph facts verified by script output.**

## Performance

- **Duration:** ~25 min
- **Completed:** 2026-03-28
- **Tasks:** 2 automated + 1 human-verify checkpoint (pending user)

## Changes by file (plan focus)

### `src/config.py`

- **`GRAPH_DATA_PATH = GRAPH_DIR / "symmap_entities.json"`** — single canonical absolute default for the SymMap JSON (replaces `GRAPH_DATA_FILE`).
- **`GRAPH_DATA_DEFAULT_RELATIVE`** remains `"data/graph/symmap_entities.json"` for `.env` / string defaults.

### `src/ui_backend.py`

- **`GRAPH_DATA_PATH` env** (or **`GRAPH_DATA_DEFAULT_RELATIVE`** when unset) resolves through `_resolve_path` in `_get_default_pipeline_config`; comment clarifies default targets SymMap KG.
- **Hybrid path:** `_retrieve_documents` → `_search_graph_documents` uses `config.graph_data_path` (SymMap file when present).
- Note: Working tree had a full hybrid-capable backend; commit `b3bd240` brought the repo in line with that implementation plus SymMap default wiring.

### `.env.example`

- Comment added that **`GRAPH_DATA_PATH`** default matches **`src/config.py`** (`GRAPH_DATA_PATH` / `GRAPH_DATA_DEFAULT_RELATIVE`).

### Also updated (rename fallout)

- `src/retriever.py` — `create_hybrid_retriever` default `os.getenv("GRAPH_DATA_PATH", str(GRAPH_DATA_PATH))`.
- `src/test_hybrid_retriever.py` — integration test uses `GRAPH_DATA_PATH` when present.

## Verification evidence (SymMap retrieval path)

Command (project venv):

```text
venv\Scripts\python.exe scripts\verify_symmap_retrieval.py
```

Observed output (2026-03-28):

```text
OK: SymMap sample graph — 頭痛 (SM00001) has 14 related edges
    Relationship types: ['ASSOCIATED_WITH', 'CONTAINS', 'CORRELATES_WITH', 'TREATS']
OK: ui_backend._search_graph_documents — 14 graph Document(s) for query 頭痛
```

This confirms: **query "頭痛"** → **`search_by_name`** on SymMap sample → **`get_related_entities`** → facts formatted as graph `Document`s, matching the hybrid path used by `run_query` / `run_query_stream`.

## Human checkpoint (UI) — **not completed by automation**

**Resume signal (per plan):** `approved`

1. Start the UI (Streamlit `src/ui_app.py` and/or Next.js per your setup).
2. Ask a TCM symptom/herb question (e.g. mentions **頭痛** or related herbs).
3. Open **citations / KG viewer** in the chat UI (`CitationPanel` graph region).
4. Confirm nodes/edges reflect **SymMap** structure (relationship types such as TREATS, ASSOCIATED_WITH, etc.) and not only legacy `entities.json` fallbacks.

Reply **`approved`** after visual confirmation (or note defects).

## Blockers / pending for L0 trustable merge

- **Full official SymMap 2.0 dataset** (not only in-repo sample) is still pending for production-accurate demos.
- **Human-reviewed seed crosswalk** (per `ENTITY_RESOLUTION.md`) is **required before** any automated entity merge; **no auto-merge logic was implemented in this plan.**

## Deviations from Plan

### Auto-fixed issues

1. **[Rule 1 — Bug]** `verify_symmap_retrieval.py` asserted legacy id `SMTS000001`; sample graph uses **`SM00001`** for 頭痛 — assertion updated to match `data/graph/symmap_entities.json`.

2. **[Rule 3 — Blocking]** Unused `huggingface_hub.constants` import removed from `ui_backend.py` after hybrid backend landed.

### GSD automation

- `gsd-tools state advance-plan` returned a parse error against current `STATE.md` format; **ROADMAP / STATE updated manually** for this execution.

## Task commits

| Task | Commit | Message |
|------|--------|---------|
| 2.3.1 | `b3bd240` | feat(2-03): SymMap GRAPH_DATA_PATH default and KG wiring |
| 2.3.2 | `02891e2` | test(2-03): verify SymMap graph facts via _search_graph_documents |
| Fix | `8b787fe` | refactor(2-03): remove unused huggingface_hub import in ui_backend |

## Known stubs

None introduced for SymMap default path or verification script.

---

## Self-Check: PASSED

- `2-03-SUMMARY.md` present at `.planning/phases/02-standard-kg-integration/2-03-SUMMARY.md`.
- Commits `b3bd240`, `02891e2`, `8b787fe` on branch `feature/premium-ui`.
- Verification script exit code 0 with expected printed evidence.
