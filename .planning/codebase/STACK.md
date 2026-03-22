# Technology Stack

**Analysis Date:** 2026-03-23

## Languages

**Primary:**
- **Python 3.10+**: Used for backend API, RAG pipeline, and data processing scripts (`src/api.py`, `src/graph_builder.py`, `src/ingest.py`, etc.).
- **TypeScript**: Used for the frontend application (`web/app/page.tsx`, `web/lib/api.ts`).

**Secondary:**
- **Markdown**: Used for system prompts, documentation, and LLM output rendering.
- **CSS (Tailwind)**: Used for UI styling in the frontend.

## Runtime

**Environment:**
- **Node.js 19+**: Frontend runtime (using Next.js 16).
- **Python 3.10+**: Backend runtime.

**Package Manager:**
- **npm**: Frontend package management (`web/package.json`).
- **pip**: Backend package management (`requirements.txt`).
- **Lockfile**: `web/package-lock.json` (present), `requirements.txt` (present).

## Frameworks

**Core:**
- **Next.js 16 (React 19)**: Frontend framework for building the chat interface and visualizations (`web/next.config.ts`).
- **FastAPI**: Backend framework for exposing the RAG pipeline as a REST API (`src/api.py`).

**Testing:**
- **Pytest/Script-based**: Backend logic is tested via various scripts in `src/` (e.g., `src/test_retriever.py`) and specialized audit scripts in `scripts/` (e.g., `scripts/quality_check.py`).
- **ESLint**: Frontend code quality and linting (`web/eslint.config.mjs`).

**Build/Dev:**
- **Tailwind CSS v4**: Utility-first CSS framework for frontend styling.
- **Uvicorn**: ASGI server for running the FastAPI backend (`src/api.py`).
- **PostCSS**: CSS transformation tool used by Tailwind.

## Key Dependencies

**Critical:**
- **ChromaDB**: Local vector database for semantic search and RAG (`src/config.py`).
- **LangChain**: LLM orchestration and retrieval components (`requirements.txt`).
- **@xyflow/react**: Used for Knowledge Graph visualization in the frontend (`web/package.json`).
- **Pydantic**: Data validation and settings management (`src/api.py`, `src/config.py`).

**Infrastructure:**
- **python-dotenv**: Environment variable management (`src/api.py`).
- **Lucide React**: Icon library for the frontend.
- **React Markdown**: Rendering LLM responses with citations.

## Configuration

**Environment:**
- **.env**: Centralized backend configuration for LLM providers, retrieval settings, and API keys.
- **web/.env.local**: Frontend-specific configuration (e.g., `NEXT_PUBLIC_BACKEND_URL`).

**Build:**
- **web/next.config.ts**: Next.js build configuration.
- **web/tsconfig.json**: TypeScript configuration.
- **src/config.py**: Python-based centralized backend configuration.

## Platform Requirements

**Development:**
- **Local Machine**: Capable of running ChromaDB and potentially local LLMs (Ollama/LM Studio).
- **Internet Connection**: Required for external LLM API providers (Alibaba, OpenAI, etc.).

**Production:**
- **Docker/Cloud**: Suitable for hosting FastAPI and Next.js applications.

---

*Stack analysis: 2026-03-23*
