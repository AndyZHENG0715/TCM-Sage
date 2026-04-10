**Generated:** 2026-04-08 | **See also:** root AGENTS.md for GSD workflow

# src/ — Python RAG Core

## Overview

RAG pipeline, FastAPI server, knowledge graph, and shared helpers. All production Python lives here.

## Key Files

| File | Purpose |
|------|---------|
| `main.py` | CLI entry + LLM factory, prompts, classification, verification |
| `api.py` | FastAPI server (SSE streaming, CORS, health, source context) |
| `ui_backend.py` | Cached resources, `run_query_stream`, `PipelineConfig` |
| `retriever.py` | `HybridRetriever` — vector + graph ensemble |
| `graph_builder.py` | `TCMKnowledgeGraph` — NetworkX loader, traversal |
| `citation_types.py` | `TextCitation`, `GraphCitation` TypedDicts |
| `config.py` | Central paths, defaults (`CHUNKS_PATH`, `GRAPH_DATA_DEFAULT_RELATIVE`) |
| `ingest.py` | Build vector index + `chunks.json` (run once) |
| `arena.py` | Arena blind A/B evaluation: raw LLM path, RAG wrapper, vote JSONL storage |

## Data Flow

**CLI path (`main.py`):**
1. Load Chroma + HybridRetriever
2. Classify query → select temperature
3. Retrieve (vector + graph)
4. Generate → Verify → Print

**API path (`api.py` → `ui_backend.py`):**
1. SSE `/query` → `run_query_stream`
2. `PipelineConfig` resolves env + overrides
3. Stream LLM chunks + final `metadata` event (citations, verification)

## Conventions

- **Imports:** `sys.path.append/insert` before local imports (not a package)
- **Naming:** `snake_case.py`, `snake_case` functions
- **Types:** Type hints on public functions; Pydantic for API models
- **Env:** Read via `src/config.py` or `os.getenv` with defaults

## Where to Add Code

| New Feature | Location |
|-------------|----------|
| Retrieval/ranking | Extend `retriever.py` or `HybridRetriever` |
| API routes | `api.py` (mirror in `web/lib/api.ts`) |
| Graph schema/import | `graph_builder.py` loader; scripts in `scripts/` |
| Tests | `test_<feature>.py` colocated here |
| Arena endpoints | `api.py` (arena routes appended at bottom) + `arena.py` |
|| Test patterns | `test_*.py` colocated in src/ — no pytest, run directly with venv python |

## Tests

No pytest. Run directly:

```bash
venv\Scripts\python.exe src/test_citations.py
venv\Scripts\python.exe src/test_graph.py
venv\Scripts\python.exe src/test_hybrid_retriever.py
```

## Anti-Patterns

- **Type suppression:** Never `as any` equivalent — fix types properly
- **Duplicate retrieval:** Use `HybridRetriever` from both CLI and `ui_backend`, not parallel implementations
- **LLM response formatting:** DO NOT include "Sources:" sections — UI strips them
- **Verification bounds:** `verify_citation_bounds` exists but not wired; invoke after generation if `[n]` markers need validation
