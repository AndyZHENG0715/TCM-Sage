# Project State

## Active Milestone
- **Name**: FYP Stabilization & KG Pivot
- **Deadline**: 2026-04-13
- **Status**: IN_PROGRESS (Context-safe handoff updated)

## Current Phase
- **Phase**: 3.0 (Presentation & E2E Polish)
- **Progress**: 10%

## Parked / deferred (do not block other work)

| Item | Status | Resume trigger | Artifact |
|------|--------|----------------|----------|
| TCM expert prompt & answer contract | **Partially resolved** — friend's sample Q&As received, cite-then-explain incorporated | Expert returns structure, tone, safety, few-shots | `.planning/todos/parked-001-tcm-expert-prompt-spec.md` |

**Safe to proceed in parallel:** Phase 2 KG migration (SymMap), adding more text sources, UI/UX improvements per roadmap.

## Completed Phases
- [x] Phase 1: Core UI Fixes (Completed 2026-03-23)
- [x] Phase 1.5: Alibaba Fix & Layout Refactor (Completed 2026-03-23)
- [x] Phase 2.5: Arena Blind Evaluation (Completed on `feature/premium-ui`, 11 commits)
- [x] Phase 2.7: Corpus Expansion & Embedding Upgrade (Completed 2026-04-02)
- [x] Phase 2.8: System Prompt Redesign (Completed 2026-04-02)

## Active Verification Gaps
Remaining issues after recent fixes:
1. **E2E Testing**: Full pipeline test with expanded corpus and new embedding model pending.
2. **KG Subgraph Visualization**: KGViewer shows simple 2-node view; subgraph exploration not yet implemented.

## Recent Activity
- Completed **Corpus Expansion**: 17 classical TCM texts (3.72M chars, 11,522 chunks) with text-embedding-v4.
- Completed **System Prompt Redesign**: Chinese 辨证论治 prompt, cite-then-explain, .env-configurable.
- Fixed Issues 1-6: duplicate /config, citation rendering, KG matching, arena UX.
- Cleaned 12 legacy scripts referencing deprecated entities_partial.json.
- Completed **Phase 02 plan 2-03** (pipeline integration): `2-03-SUMMARY.md`, `GRAPH_DATA_PATH` default to `data/graph/symmap/symmap_entities.json`, `scripts/verify_symmap_retrieval.py` confirms `_search_graph_documents` for 頭痛; UI human-verify completed with real SymMap data and crosswalk review.
- Completed **Phase 02 plan 2-02** (SymMap KG adapter): `2-02-SUMMARY.md`, `scripts/import_symmap_kg.py` (legacy SM/HM + `rel_*` edge typing), now validated against real SymMap v2.0 export.
- Architecture update: SymMap-only KG is now the runtime source of truth; legacy `entities_partial.json` is treated as archival.
- Bridge update: query-time crosswalk lookup is enabled (`src/crosswalk_bridge.py`) so approved mappings can resolve RAG terms to SymMap node IDs during graph retrieval.
- Completed **Phase 02 plan 2-01** (SymMap 2.0 research & mapping): `2-01-SUMMARY.md`, mapping doc column examples, `graph_builder` docstring alignment with `SYMMAP_MAPPING.md`.
- Completed **Arena blind evaluation feature set** on `feature/premium-ui` (11 commits): backend `arena.py`, frontend arena page, KG `max_results`, arena model config, clickable citations, verification badge; architecture docs synced via map-codebase.
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
