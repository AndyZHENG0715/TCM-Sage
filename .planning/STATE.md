# Project State

## Active Milestone
- **Name**: FYP Stabilization & KG Pivot
- **Deadline**: 2026-04-13
- **Status**: IN_PROGRESS (Context-safe handoff updated)

## Current Phase
- **Phase**: 1.6 (Closing Verification Gaps)
- **Progress**: 96%

## Parked / deferred (do not block other work)

| Item | Status | Resume trigger | Artifact |
|------|--------|----------------|----------|
| TCM expert prompt & answer contract | **Parked** — awaiting friend / domain feedback | Expert returns structure, tone, safety, few-shots | `.planning/todos/parked-001-tcm-expert-prompt-spec.md` |

**Safe to proceed in parallel:** Phase 2 KG migration (SymMap), adding more text sources, UI/UX improvements per roadmap.

## Completed Phases
- [x] Phase 1: Core UI Fixes (Completed 2026-03-23)
- [x] Phase 1.5: Alibaba Fix & Layout Refactor (Completed 2026-03-23)

## Active Verification Gaps
Remaining issues after recent fixes:
1. **Prompt/Answer Quality Regression**: Responses may still include undesired source-like endings in edge cases and overall answer quality needs prompt redesign.
2. **Mobile Background Streaming Limitation**: Android/iOS can drop active streaming when device locks or tab is backgrounded (browser/OS behavior; defer unless architecture change is approved).

## Recent Activity
- Completed **Phase 02 plan 2-03** (pipeline integration): `2-03-SUMMARY.md`, `GRAPH_DATA_PATH` default to `data/graph/symmap_entities.json`, `scripts/verify_symmap_retrieval.py` confirms `_search_graph_documents` for 頭痛; **UI human-verify checkpoint** pending (resume signal `approved`). Full SymMap export + human-reviewed seed crosswalk still required before L0 trustable entity merge (no auto-merge in this plan).
- Completed **Phase 02 plan 2-02** (SymMap KG adapter): `2-02-SUMMARY.md`, `scripts/import_symmap_kg.py` (legacy SM/HM + `rel_*` edge typing), `data/graph/symmap_entities.json` from `data/symmap_sample/` (160 nodes, 230 edges; `TCMKnowledgeGraph.load_from_json` verified).
- Completed **Phase 02 plan 2-01** (SymMap 2.0 research & mapping): `2-01-SUMMARY.md`, mapping doc column examples, `graph_builder` docstring alignment with `SYMMAP_MAPPING.md`.
- Fixed full-context citation chain end-to-end:
  - canonical chunk ID reconstruction in `src/main.py`
  - robust `/source/{chunk_id}/context` lookup normalization
  - `/books/{book_name}` filename + encoding fallback handling
- Added same-origin backend proxy in Next.js: `web/app/api/backend/[...path]/route.ts`
  - resolved remote/mobile client failures caused by client-side `localhost` backend URL usage
- Updated source-page back behavior for popup/tab flow:
  - opened-with-chat path uses close-tab-first UX
- Added context-reset-safe handoff document:
  - `/.planning/phases/01-stabilization-bug-fixes/1.6-HANDOFF-APIS-PROMPT.md`
- Added standard pause/resume artifacts for session handoff:
  - `/.planning/HANDOFF.json` (machine-readable)
  - `/.planning/phases/01-stabilization-bug-fixes/.continue-here.md` (human-readable)

## Resume Instructions
- Primary command: `/gsd-resume-work`
- If routing is unclear in a fresh session, use: `/gsd-do` with a natural-language request.
