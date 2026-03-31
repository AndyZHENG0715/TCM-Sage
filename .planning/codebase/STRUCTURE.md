# Codebase Structure

**Analysis Date:** 2026-04-01

## Directory Layout

```text
TCM-Sage/
├── src/                     # Python RAG core (ingest, retriever, graph, API, CLI, arena)
├── web/                     # Next.js 16 App Router frontend + backend proxy route
├── scripts/                 # Operational scripts (SymMap import, verification, diagnostics)
├── data/
│   ├── source/              # Raw corpus .txt files for ingestion
│   ├── processed/           # Generated `chunks.json`
│   ├── graph/               # Knowledge graph assets (`symmap/` default + legacy graph files)
│   ├── feedback/            # Arena vote output (`arena_votes.jsonl`)
│   └── ...                  # Additional research/sample assets
├── vectorstore/             # Generated Chroma persistence directory (`chroma/`)
├── docs/                    # Project documentation (for setup/config/reference)
├── .planning/               # GSD planning artifacts, phases, and codebase maps
├── plan/                    # Separate planning docs (`plan/sdp.md`)
├── requirements.txt         # Python dependency lock-style list
├── README.md                # Project overview/runbook
└── AGENTS.md                # Agent conventions and workflow guardrails
```

## Directory Purposes

**`src`:**
- Purpose: Owns runtime backend behavior and shared RAG logic.
- Contains: `src/main.py`, `src/api.py`, `src/ui_backend.py`, `src/retriever.py`, `src/graph_builder.py`, `src/arena.py`, `src/config.py`, `src/citation_types.py`, `src/ingest.py`, plus `src/test_*.py`.
- Key files: `src/main.py`, `src/ui_backend.py`, `src/api.py`.

**`web`:**
- Purpose: Owns user-facing UI and browser-side transport.
- Contains: `web/app` routes, `web/components`, `web/hooks`, `web/lib`.
- Key files: `web/app/page.tsx`, `web/lib/api.ts`, `web/app/api/backend/[...path]/route.ts`, `web/app/source/[chunkId]/page.tsx`, `web/app/arena/page.tsx`.

**`scripts`:**
- Purpose: Owns one-shot data import/verification and diagnostics.
- Contains: `scripts/import_symmap_kg.py`, `scripts/verify_symmap_retrieval.py`, `scripts/e2e_test.py`, and other checks.
- Key files: `scripts/import_symmap_kg.py`, `scripts/verify_symmap_retrieval.py`.

**`data`:**
- Purpose: Owns corpus and graph data artifacts consumed by backend.
- Contains: `data/source`, `data/processed/chunks.json`, `data/graph/symmap/symmap_entities.json`, feedback and sample datasets.
- Key files: `data/processed/chunks.json`, `data/graph/symmap/symmap_entities.json`.

**`vectorstore`:**
- Purpose: Owns persisted vector index generated from ingestion.
- Contains: `vectorstore/chroma`.
- Key files: runtime-loaded by `src/main.py` and `src/ui_backend.py`.

**`.planning`:**
- Purpose: Owns planning and mapping references used by GSD workflow.
- Contains: `.planning/codebase/*.md`, `.planning/phases/*`, roadmap and requirements docs.

## Key File Locations

**Entry Points:**
- `src/main.py`: CLI entrypoint for interactive RAG.
- `src/api.py`: FastAPI app + SSE + arena endpoints.
- `src/ingest.py`: Build chunk metadata and vector index.
- `web/app/page.tsx`: main chat page.
- `web/app/arena/page.tsx`: arena A/B experience.

**Configuration:**
- `src/config.py`: central path and defaults constants.
- `docs/CONFIG.md`: environment and runtime configuration reference.
- `.env.example`: sample env keys (values provided at runtime in `.env`).
- `web/next.config.ts`: frontend build/runtime config.

**Core Logic:**
- `src/retriever.py`: `HybridRetriever` implementation.
- `src/graph_builder.py`: graph model, loading, traversal, matching.
- `src/ui_backend.py`: API execution pipeline and cached resources.
- `src/citation_types.py`: backend citation contract.
- `web/lib/types.ts`: frontend mirrored citation/message/settings types.

**Testing:**
- `src/test_citations.py`, `src/test_graph.py`, `src/test_hybrid_retriever.py`: script-style backend tests.
- `scripts/verify_symmap_retrieval.py`, `scripts/e2e_test.py`: integration/diagnostic checks.

## Naming Conventions

**Files:**
- Python modules: `snake_case.py` in `src/` and `scripts/`.
- React components: `PascalCase.tsx` in `web/components`.
- Hooks/utils/types: `camelCase.ts` in `web/hooks` and `web/lib`.
- App Router dynamic route segments: bracketed names like `web/app/source/[chunkId]/page.tsx`.

**Directories:**
- Frontend route ownership in `web/app/<route>/`.
- Planning phases in `.planning/phases/<numeric-prefix>-<slug>/`.
- Data organization by lifecycle stage (`source` -> `processed` -> `vectorstore`).

## Where to Add New Code

**New Feature:**
- Primary backend logic: `src/` (prefer extending existing modules before creating new top-level abstractions).
- API exposure: `src/api.py`.
- Frontend integration: `web/lib/api.ts` + route/component under `web/app`/`web/components`.
- Tests: `src/test_<feature>.py` and optional script-level verification under `scripts/`.

**New Component/Module:**
- UI component: `web/components/<Name>.tsx`.
- UI hook: `web/hooks/use<Feature>.ts`.
- Backend helper module: `src/<feature>.py` with imports from `src/config.py` for paths/defaults.

**Utilities:**
- Shared frontend helpers: `web/lib/*.ts`.
- Shared backend constants/config: `src/config.py`.
- One-time migration/import logic: `scripts/*.py`.

## Special Directories

**`data/graph/symmap`:**
- Purpose: SymMap-shaped KG files and source materials.
- Generated: Yes, via scripts such as `scripts/import_symmap_kg.py`.
- Committed: Yes, currently used as default graph source.

**`vectorstore/chroma`:**
- Purpose: Chroma database files generated by ingestion.
- Generated: Yes.
- Committed: Not required for source-level development.

**`web/app/api/backend/[...path]`:**
- Purpose: Internal proxy boundary between browser and FastAPI service.
- Generated: No.
- Committed: Yes.

**`.planning/codebase`:**
- Purpose: Implementation-planning references (`ARCHITECTURE.md`, `STRUCTURE.md`, etc.).
- Generated: Yes (by mapping workflow).
- Committed: Yes.

## Module Boundaries and Ownership

**Backend runtime boundary:**
- `src/api.py` should own HTTP concerns only; heavy retrieval/generation logic stays in `src/ui_backend.py` and `src/main.py` helpers.

**Retrieval boundary:**
- Retrieval algorithms belong in `src/retriever.py` and graph traversal in `src/graph_builder.py`; avoid embedding retrieval logic into route handlers.

**Frontend transport boundary:**
- Network contracts belong in `web/lib/api.ts`; UI hooks/components consume typed functions instead of raw fetch calls.

**Data pipeline boundary:**
- Data transformation/import belongs in `src/ingest.py` and `scripts/`; runtime query code should treat `data/` and `vectorstore/` as read-only inputs.

---

*Structure analysis: 2026-04-01*
