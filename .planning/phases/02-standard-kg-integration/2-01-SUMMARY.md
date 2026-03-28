---
phase: 02-standard-kg-integration
plan: "2-01"
subsystem: kg
tags: [symmap, networkx, tcm, mapping, knowledge-graph]

requires:
  - phase: 01-stabilization-bug-fixes
    provides: Stable UI and pipeline for later KG swap
provides:
  - SymMap 2.0 dataset structure and download-format documentation
  - Entity, relationship, and attribute mapping to TCMKnowledgeGraph / entities.json shape
  - Confirmed compatibility of graph_builder ENTITY_TYPES and RELATIONSHIP_TYPES with SymMap-oriented types
affects:
  - 02-standard-kg-integration plan 2-02 (adapter script)
  - 02-standard-kg-integration plan 2-03 (pipeline integration)

tech-stack:
  added: []
  patterns:
    - "SymMap prefix IDs (SMTS, SMHB, SMIT, SMTT, SMDE, SMYS, SMMS) as canonical entity keys"
    - "Legacy SM/HM/IM/TM/MM shorthand documented alongside SymMap 2.0 names"

key-files:
  created: []
  modified:
    - .planning/phases/02-standard-kg-integration/SYMMAP_MAPPING.md
    - src/graph_builder.py

key-decisions:
  - "No change to ENTITY_TYPES or RELATIONSHIP_TYPES — they already included Disease, Ingredient, Target, Syndrome, MAPS_TO, TARGETS, ASSOCIATED_WITH, and CORRELATES_WITH required by SYMMAP_MAPPING.md"
  - "Module docstring for INDICATES/CORRELATES_WITH updated to match SYMMAP_MAPPING.md (symptom–syndrome vs symptom–disease layers)"

patterns-established:
  - "Mapping doc is single source of truth for adapter column aliases and edge directions"

requirements-completed: [Task 2.1]

duration: 15 min
completed: 2026-03-28
---

# Phase 02 Plan 01: SymMap 2.0 research and mapping summary

**SymMap 2.0 bulk structure, legacy SM/HM/IM/TM/MM column examples, and full entity/relationship mapping to NetworkX `TCMKnowledgeGraph` types — with graph_builder constants verified as already sufficient.**

## Performance

- **Duration:** 15 min
- **Started:** 2026-03-28T00:00:00Z
- **Completed:** 2026-03-28T00:15:00Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- **Dataset Structure:** `SYMMAP_MAPPING.md` already contained download sources, six entity layers + syndrome, relationship tables, and legacy shorthand; Task 2.1.1 added an explicit **example header patterns** table (`SMTS_ID`, `SMHB_Pinyin`, etc.) and TSV/CSV/Excel notes aligned with Wu et al. 2019 / symmap.org bulk exports.
- **Schema Mapping:** Confirmed **Schema Mapping** section documents entity types (Symptom/Herb/Ingredient/Target/Disease/Syndrome), edge types (TREATS, CONTAINS, TARGETS, ASSOCIATED_WITH, CORRELATES_WITH, MAPS_TO, INDICATES), node attributes, and JSON export shape for Wave 2 adapter work.
- **graph_builder.py:** `ENTITY_TYPES` and `RELATIONSHIP_TYPES` **already matched** the mapping document; only the top-of-file **relationship docstrings** were aligned so `INDICATES` and `CORRELATES_WITH` describe the same SymMap layers as `SYMMAP_MAPPING.md`.

## Task commits

1. **Task 2.1.1: Research SymMap 2.0 file structures** — `276f6e2` (docs)
2. **Task 2.1.2: Map SymMap to TCMKnowledgeGraph schema** — `480207b` (docs)

**Plan metadata:** Single commit titled `docs(2-01): complete SymMap research and mapping plan` (SUMMARY, STATE, ROADMAP); see `git log -1 --oneline` on branch tip after pull.

_Note: Per-task scope was documentation and comment accuracy; no runtime graph logic changed._

## Files created/modified

- `.planning/phases/02-standard-kg-integration/SYMMAP_MAPPING.md` — Dataset Structure enrichment (example ID/name column patterns; format notes).
- `src/graph_builder.py` — Relationship type descriptions aligned with SymMap mapping (no constant set changes).

## Decisions made

- Treat **SYMMAP_MAPPING.md** as the adapter contract; `load_from_json` remains permissive for forward compatibility.
- **No new** `ENTITY_TYPES` / `RELATIONSHIP_TYPES` entries — existing sets already cover SymMap-oriented ingestion.

## Deviations from plan

None — plan executed as written. Verification noted pre-existing comprehensive mapping sections; incremental edits strengthened column-level detail and doc consistency only.

## Issues encountered

- `gsd-tools` `state advance-plan`, `update-progress`, `record-metric`, `add-decision`, and `record-session` reported missing sections in `STATE.md` (non–GSD-shaped file); **ROADMAP** `update-plan-progress` succeeded. `requirements mark-complete "Task 2.1"` split the token and did not match `REQUIREMENTS.md` rows — **manual traceability:** plan requirement **Task 2.1** is recorded in this summary frontmatter only.

## User setup required

None — no external service configuration required.

## Next phase readiness

- Mapping document is suitable input for **2-02** (SymMap data adapter).
- Adapter should normalize alias headers and respect directed edge semantics in the Schema Mapping table.

## Self-check: PASSED

- `SYMMAP_MAPPING.md` exists and includes **Dataset Structure** and **Schema Mapping** sections (verified by read).
- Task commits `276f6e2` and `480207b` verified; metadata commit present with message `docs(2-01): complete SymMap research and mapping plan`.
- `2-01-SUMMARY.md` written at `.planning/phases/02-standard-kg-integration/2-01-SUMMARY.md`.

---
*Phase: 02-standard-kg-integration*  
*Completed: 2026-03-28*
