# Architecture

**Analysis Date:** 2026-04-01

## Pattern Overview

**Overall:** Modular RAG system with shared backend primitives and two serving surfaces: CLI (`src/main.py`) and web (`src/api.py` + `web/`). Retrieval is vector-first (Chroma) with optional KG augmentation (NetworkX graph loaded from JSON). The web stack adds SSE token streaming, structured citation metadata, and source-context drill-down.

**Key Characteristics:**
- Shared core functions in `src/main.py` (`create_llm`, prompt construction, query classification, verification, citation formatting).
- Runtime configuration centralized in `src/ui_backend.py` via `PipelineConfig` (env defaults + per-request overrides).
- Data persistence split by role: chunk metadata in `data/processed/chunks.json`, embeddings in `vectorstore/chroma`, KG in `data/graph/symmap/symmap_entities.json` by default.

## Layers

**Ingestion Layer:**
- Purpose: Convert raw corpus into chunked metadata + vector index.
- Location: `src/ingest.py`
- Contains: `SentenceAwareChineseTextSplitter`, chapter parsing with offsets, Chroma write path.
- Depends on: `data/source/*.txt`, `langchain_community` embeddings/vectorstore.
- Used by: CLI/API query paths and citation context reconstruction endpoint.

**Retrieval Layer:**
- Purpose: Produce retrieval documents for generation.
- Location: `src/retriever.py`, `src/ui_backend.py`
- Contains: `HybridRetriever` (CLI path), `_retrieve_documents` (API/UI path), `vector_search_with_scores`.
- Depends on: `vectorstore/chroma`, graph loader in `src/graph_builder.py`.
- Used by: `src/main.py` and `src/ui_backend.py`.

**Knowledge Graph Layer:**
- Purpose: Entity/relationship graph loading and traversal.
- Location: `src/graph_builder.py`
- Contains: `TCMKnowledgeGraph`, `search_by_name`, `get_related_entities`, schema-tolerant JSON parsing.
- Depends on: JSON graph files (SymMap and legacy-compatible shapes).
- Used by: `src/retriever.py`, `src/ui_backend.py`, verification scripts such as `scripts/verify_symmap_retrieval.py`.

**Generation + Safety Layer:**
- Purpose: Route queries, generate answers, and verify faithfulness.
- Location: `src/main.py`, consumed from `src/ui_backend.py`
- Contains: `get_query_severity`, `build_prompt_template`, `create_llm`, `verify_answer`.
- Depends on: provider env settings and keys, retrieval context string.
- Used by: CLI interactive loop and streamed API execution.

**API Layer:**
- Purpose: HTTP/SSE interface for UI and tools.
- Location: `src/api.py`
- Contains: `/query` (SSE), `/config`, `/health`, `/source/{chunk_id}/context`, `/books/{book_name}`, arena endpoints.
- Depends on: `run_query_stream`, `get_runtime_config`, shared vectorstore/cache functions in `src/ui_backend.py`.
- Used by: Next.js backend proxy route and test scripts.

**Frontend Layer:**
- Purpose: Chat UX, citation visualization, settings, arena comparison mode.
- Location: `web/app`, `web/hooks`, `web/components`, `web/lib`
- Contains: Chat state in `web/hooks/useChat.ts`, API transport in `web/lib/api.ts`, source page in `web/app/source/[chunkId]/page.tsx`.
- Depends on: `web/app/api/backend/[...path]/route.ts` proxy to FastAPI.
- Used by: User-facing web app at `web/app/page.tsx` and `web/app/arena/page.tsx`.

## Data Flow

**Ingestion to index:**
1. `src/ingest.py` reads `data/source/*.txt`.
2. Content is split into chapter-aware chunks with `book/source/chunk_index/char_start/char_end`.
3. Chunks saved to `data/processed/chunks.json`.
4. Embeddings generated and persisted to `vectorstore/chroma`.

**Retrieval + KG for answering (CLI):**
1. `src/main.py` loads Chroma and optionally builds `HybridRetriever`.
2. Query classified by `get_query_severity`.
3. Retriever returns vector docs and optional graph docs.
4. Context assembled via `format_docs`.
5. Selected LLM generates answer; verifier LLM checks support status.

**Retrieval + KG for answering (API/UI):**
1. `web/lib/api.ts` sends POST to `/api/backend/query` (proxied by `web/app/api/backend/[...path]/route.ts`).
2. `src/api.py` forwards to `run_query_stream` in `src/ui_backend.py`.
3. `run_query_stream` resolves runtime config, retrieves docs (`_retrieve_documents`), formats numbered context (`format_docs_with_citations`), streams LLM chunks.
4. Final SSE `metadata` event includes citations, severity, verification payload.
5. `web/hooks/useChat.ts` merges streamed text and metadata into assistant message state.

**Citation/source drill-down:**
1. User clicks citation in `web/components/CitationPanel.tsx`.
2. UI fetches `/source/{chunk_id}/context` for chapter reconstruction/highlight mapping.
3. `src/api.py` reconstructs full chapter by deduplicating overlap from `data/processed/chunks.json`.
4. Source page `web/app/source/[chunkId]/page.tsx` can additionally load full book content via `/books/{book_name}`.

**Arena flow:**
1. `web/app/arena/page.tsx` + `web/hooks/useArena.ts` send `/arena/query`.
2. `src/api.py` multiplexes two async streams from `src/arena.py` (RAG vs plain model, blind assignment).
3. Votes posted to `/arena/vote` and appended to `data/feedback/arena_votes.jsonl`.

**State Management:**
- Client state: React hooks (`useChat`, `useHistory`, `useSettings`, `useArena`).
- Backend runtime state: process-local caches in `src/ui_backend.py` (`@lru_cache` for embeddings/vectorstore/graph/config).
- Session persistence: local UI session storage (`useHistory`) and append-only arena votes file.

## Key Abstractions

**PipelineConfig:**
- Purpose: Canonical runtime settings model.
- Examples: `src/ui_backend.py`
- Pattern: immutable dataclass resolved from env + request overrides.

**HybridRetriever:**
- Purpose: Encapsulate vector + KG retrieval for CLI path.
- Examples: `src/retriever.py`
- Pattern: vector results first, graph facts second.

**Citation Types:**
- Purpose: Strongly typed metadata contract between backend and UI.
- Examples: `src/citation_types.py`, `web/lib/types.ts`
- Pattern: `TextCitation` + `GraphCitation` union aligned to numbered context.

**Backend Proxy Seam:**
- Purpose: Decouple browser from direct backend host/port.
- Examples: `web/app/api/backend/[...path]/route.ts`
- Pattern: transparent pass-through for GET/POST and streaming bodies.

## Entry Points

**CLI RAG:**
- Location: `src/main.py`
- Triggers: `venv\Scripts\python.exe src/main.py`
- Responsibilities: interactive question loop using shared retrieval/generation pipeline.

**FastAPI Server:**
- Location: `src/api.py`
- Triggers: `venv\Scripts\python.exe src/api.py`
- Responsibilities: streaming query endpoint, source lookup endpoints, arena endpoints.

**Ingestion Job:**
- Location: `src/ingest.py`
- Triggers: `venv\Scripts\python.exe src/ingest.py`
- Responsibilities: rebuild chunk metadata and vector store.

**Next.js App:**
- Location: `web/app`
- Triggers: `npm run dev` from `web/`
- Responsibilities: chat UI, citation panel, source reader, arena UI.

## Error Handling

**Strategy:** Prefer graceful degradation to vector-only retrieval when graph/hybrid steps fail; return explicit HTTP errors for invalid requests; emit SSE `error` events for stream failures.

**Patterns:**
- Hybrid fallback in `src/main.py` when retriever initialization fails.
- Graph retrieval fallback in `src/ui_backend.py` `_retrieve_documents`.
- API input validation through Pydantic models in `src/api.py`.

## Cross-Cutting Concerns

**Logging:** Print/debug logging in Python modules; API traceback passthrough for unexpected errors.
**Validation:** Pydantic request models and typed citation payload structures.
**Authentication:** Not detected in `src/api.py`; API is open by default with CORS controlled by `ALLOWED_ORIGINS`.

## Notable Coupling and Extension Seams

**Coupling Points:**
- `src/ui_backend.py` imports core orchestration functions directly from `src/main.py`.
- Frontend stream parser in `web/lib/api.ts` is coupled to exact SSE event names emitted by `src/api.py`.
- Citation UX relies on chunk metadata shape produced by `src/ingest.py` and consumed in `/source/{chunk_id}/context`.

**Extension Seams:**
- Add providers/models by extending `create_llm` in `src/main.py`.
- Add retrieval strategies in `src/retriever.py` and/or `_retrieve_documents` in `src/ui_backend.py`.
- Add new API capabilities in `src/api.py`, then map in `web/lib/api.ts`.
- Expand graph ingestion via `scripts/import_symmap_kg.py` while preserving loader compatibility in `src/graph_builder.py`.

## Divergence: `src/main.py` vs `src/ui_backend.py`

| Concern | `src/main.py` (CLI) | `src/ui_backend.py` (API/UI) |
||--------|---------------------|------------------------------|
| Hybrid retrieval | Uses `HybridRetriever` from `src/retriever.py` | Reimplements hybrid via `vector_search_with_scores` + `_search_graph_documents` (parallel logic, not shared call) |
| Graph file fallback | Env default only | Additional fallback to `entities.json` / `entities_partial.json` |
| Context string for LLM | `format_docs` (debug sections) in chain | `format_docs_with_citations` (numbered sources, UI-safe) |
| Streaming | Not used | `run_query_stream` + `create_llm(..., streaming=True)` |
| Embeddings | HuggingFace model id directly | Optional local snapshot + `HF_LOCAL_FILES_ONLY` for air-gapped loads |
| Citation bound check | `verify_citation_bounds` exists but is **not** wired into CLI or UI pipeline | Not invoked |
| Chat history | N/A | Prepended to context only in streaming path |

**Prescription for new work:** Prefer extending `HybridRetriever` (or a thin shared retrieval module) so CLI and API stay behavior-identical; call `verify_citation_bounds` after generation if UI integrity for `[n]` markers is required.
## Current Branch Realities

- Default graph path now points to SymMap export (`src/config.py` -> `data/graph/symmap/symmap_entities.json`), while `src/ui_backend.py` still includes fallback to legacy `data/graph/entities*.json`.
- Arena A/B workflow is active across backend (`src/arena.py`, arena routes in `src/api.py`) and frontend (`web/app/arena/page.tsx`, `web/hooks/useArena.ts`).
- Both CLI and API paths remain in active use, with similar but not identical retrieval orchestration.

---

*Architecture analysis: 2026-04-01*
