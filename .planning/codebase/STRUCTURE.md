# Codebase Structure

**Analysis Date:** 2026-03-23

## Directory Layout

```
TCM-Sage/
├── data/               # Persistent data storage
│   ├── source/         # Raw source texts (.txt)
│   ├── processed/      # Chunked and processed data (.json)
│   └── graph/          # Knowledge Graph extraction results
├── src/                # Backend source code (Python)
├── web/                # Frontend source code (Next.js/TypeScript)
├── scripts/            # Maintenance and audit scripts
├── vectorstore/        # Vector database persistence (ChromaDB)
├── docs/               # Architecture and project documentation
├── research/           # Research papers and notes
├── walkthrough/        # Pipeline guides and documentation
├── openspec/           # Feature and evolution specifications
└── plan/               # Implementation and roadmap plans
```

## Directory Purposes

**src/:**
- Purpose: Contains the core RAG pipeline, API server, and retrieval logic.
- Contains: Python modules for retrieval, generation, and extraction.
- Key files: `api.py` (API server), `main.py` (Core pipeline), `ui_backend.py` (Shared logic), `retriever.py` (Hybrid search), `kg_extractor.py` (KG extraction).

**web/:**
- Purpose: Modern Next.js application for the user interface.
- Contains: React components, hooks, and API client.
- Key files: `app/page.tsx` (Main UI), `lib/api.ts` (API client), `hooks/useChat.ts` (State management).

**data/:**
- Purpose: Storage for raw and intermediate data throughout the pipeline.
- Contains: `.txt` sources, `.json` chunks, and `.json` graph facts.
- Key files: `source/` (Huangdi Neijing texts), `processed/chunks.json` (Deduplicated chunks for UI).

**scripts/:**
- Purpose: Utility scripts for maintenance, testing, and auditing.
- Contains: Single-purpose Python scripts for health checks and data inspection.
- Key files: `check_health.py`, `quality_check.py`, `e2e_test.py`.

**vectorstore/chroma/:**
- Purpose: Persistent storage for vector embeddings.
- Contains: SQLite and Parquet files managed by ChromaDB.

**openspec/specs/:**
- Purpose: Detailed technical specifications for past and future architecture changes.
- Contains: Markdown files describing feature designs (e.g., `retrieval-graph`).

## Key File Locations

**Entry Points:**
- `src/api.py`: Backend REST API entry point.
- `src/main.py`: Core logic and CLI entry point.
- `web/app/page.tsx`: Frontend main application page.

**Configuration:**
- `.env.example`: Template for environment variables.
- `src/config.py`: Centralized backend configuration loader.
- `web/next.config.ts`: Next.js build and runtime configuration.
- `web/tailwind.config.ts`: Frontend styling configuration.

**Core Logic:**
- `src/retriever.py`: Hybrid retrieval implementation.
- `src/ui_backend.py`: Shared backend helpers for UI-specific features.
- `web/lib/api.ts`: Frontend client for communicating with the backend.

**Testing:**
- `src/test_retriever.py`: Unit tests for retrieval logic.
- `src/test_citations.py`: Tests for provenance tracking.
- `scripts/e2e_test.py`: End-to-end integration tests.

## Naming Conventions

**Files:**
- **Python**: `snake_case.py` (e.g., `graph_builder.py`).
- **TypeScript (React)**: `PascalCase.tsx` for components (e.g., `ChatArea.tsx`), `camelCase.ts` for hooks and libs (e.g., `useChat.ts`).
- **Data**: `snake_case.json` or `snake_case.txt`.

**Directories:**
- **General**: `snake_case` (e.g., `vectorstore`, `source`).
- **Web App**: Next.js standard `app/` structure.

## Where to Add New Code

**New Feature (Backend):**
- **Logic**: Primary implementation in `src/` (e.g., `src/new_feature.py`).
- **API**: Add endpoints to `src/api.py`.
- **Integration**: Update `src/ui_backend.py` if it affects the main RAG flow.

**New UI Component:**
- **Implementation**: Place in `web/components/`.
- **State**: Use existing hooks in `web/hooks/` or add new ones if shared.

**New Utility:**
- **Shared Helpers**: `src/utils.py` (if added) or specific module.
- **Maintenance Scripts**: `scripts/`.

**New Data Source:**
- **Input**: Place raw text in `data/source/`.
- **Processing**: Run `python src/ingest.py` to update the vector store.

## Special Directories

**vectorstore/chroma/:**
- Purpose: Persistent vector database.
- Generated: Yes (by `src/ingest.py`).
- Committed: No (managed locally).

**web/.next/:**
- Purpose: Next.js build output.
- Generated: Yes.
- Committed: No.

**venv/:**
- Purpose: Python virtual environment.
- Generated: Yes.
- Committed: No.

---

*Structure analysis: 2026-03-23*
