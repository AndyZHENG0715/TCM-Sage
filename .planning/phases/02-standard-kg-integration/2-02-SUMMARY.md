---
phase: 2-standard-kg-integration
plan: "2-02"
subsystem: data
tags: [symmap, knowledge-graph, tsv, networkx]

requires:
  - phase: 2-standard-kg-integration
    provides: "SYMMAP_MAPPING.md and graph_builder ENTITY_TYPES aligned with SymMap"
provides:
  - "scripts/import_symmap_kg.py SymMap/legacy TSV adapter"
  - "data/graph/symmap_entities.json sample-backed graph export"
affects:
  - "Phase 2 KG consumption in retriever/UI"

tech-stack:
  added: []
  patterns:
    - "Filename hints for entity kind when IDs are legacy (SM/HM/IM/TM/MM)"
    - "rel_* files mapped to edge types per SYMMAP_MAPPING.md"

key-files:
  created: []
  modified:
    - "scripts/import_symmap_kg.py"
    - "data/graph/symmap_entities.json"

key-decisions:
  - "Prefer repo sample data at data/symmap_sample over --sample when present for realistic column coverage"
  - "Legacy HM_ID/SM_ID relationship files emit TREATS as Herb→Symptom per mapping table"

patterns-established:
  - "Relationship file stem drives default edge type and endpoint column pairs"

requirements-completed: ["Task 2.2"]

duration: 12min
completed: 2026-03-28
---

# Phase 2 Plan 2-02: SymMap KG adapter Summary

**SymMap-style TSV/CSV import into `entities`/`relationships` JSON, validated by loading 160 nodes and 230 edges with `TCMKnowledgeGraph.load_from_json`.**

## Performance

- **Duration:** ~12 min
- **Started:** 2026-03-28 (execution window)
- **Completed:** 2026-03-28
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Extended `import_symmap_kg.py` to parse legacy SymMap-style IDs (`SM_ID`, `HM_ID`, …) and map `rel_*` tables to `TREATS`, `CONTAINS`, `TARGETS`, `ASSOCIATED_WITH`, and `CORRELATES_WITH` per `SYMMAP_MAPPING.md`.
- Generated `data/graph/symmap_entities.json` from `data/symmap_sample/` (160 entities, 230 relationships; all entity types present: Symptom, Herb, Ingredient, Target, Disease).
- Confirmed `TCMKnowledgeGraph.load_from_json` loads the file without error and statistics match raw JSON counts.

## Task Commits

1. **Task 2.2.1: Implement SymMap Data Adapter script** - `0ca1145` (feat)
2. **Task 2.2.2: Generate symmap_entities.json** - `2939bb5` (chore)

**Plan metadata:** _(pending — docs commit hash below)_

## Files Created/Modified

- `scripts/import_symmap_kg.py` — Parses entity tables and `rel_*` pairwise files; `--sample` remains for synthetic graphs; `--input-dir` for bulk exports.
- `data/graph/symmap_entities.json` — Graph payload with `entities` and `relationships` keys.

## Decisions Made

- Used `data/symmap_sample/` as the generation source because it ships in-repo and exercises legacy column names; full SymMap 2.0 bulk downloads can use the same script with `--input-dir`.
- `MM_*` rows in the sample disease table are exported as `Disease` / `SMDE` component (modern disease layer), consistent with the mapping doc’s SMDE column examples.

## Deviations from Plan

None - plan executed as written. The prior importer failed on sample files (unrecognized `SM_ID`/`HM_ID`); fixing that was required to satisfy the plan’s verification, not a scope change.

## Issues Encountered

- Initial `load_directory` run produced “No rows parsed” because entity IDs and relationship endpoints did not match the earlier `_pick` candidate lists; resolved by legacy ID detection and explicit `rel_*` routing.

## User Setup Required

None. Optional: place official SymMap export TSVs in a directory and run `venv\Scripts\python.exe scripts/import_symmap_kg.py --input-dir <dir> -o data/graph/symmap_entities.json`.

## Next Phase Readiness

- JSON shape matches `load_from_json` expectations; downstream work can wire retrieval or UI against `symmap_entities.json` or merge with existing graph assets.

## Known Stubs

None for this plan’s deliverables.

---

*Phase: 2-standard-kg-integration*  
*Completed: 2026-03-28*

## Self-Check

- **Files:** `scripts/import_symmap_kg.py` — FOUND; `data/graph/symmap_entities.json` — FOUND; `2-02-SUMMARY.md` — FOUND.
- **Commits:** `0ca1145`, `2939bb5` — present on branch `feature/premium-ui`.
- **Load test:** `TCMKnowledgeGraph.load_from_json` → 160 nodes, 230 edges — PASSED.

## Self-Check: PASSED
