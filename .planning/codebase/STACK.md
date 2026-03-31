# Technology Stack

**Analysis Date:** 2026-04-01

## Languages

**Primary:**
- Python (3.10+ expected from project docs) - backend API, retrieval pipeline, ingestion, utility scripts (`src/`, `scripts/`, `AGENTS.md`).
- TypeScript - frontend app and API proxy (`web/app/`, `web/lib/`, `web/components/`).

**Secondary:**
- Markdown - docs and planning (`README.md`, `docs/`, `.planning/`).
- Plain text corpora - source medical texts (`data/source/*.txt`).

## Runtime

**Environment:**
- Python via project virtual environment - required workflow is `venv\Scripts\python.exe` on Windows (`AGENTS.md`).
- Node.js runtime for Next.js frontend (`web/package.json` scripts).

**Package Manager:**
- Python: `pip` + pinned `requirements.txt`.
- Frontend: `npm` + lockfile present at `web/package-lock.json`.

## Frameworks

**Core backend:**
- FastAPI ecosystem (imported in `src/api.py`) - REST + SSE API.
- Uvicorn (`uvicorn==0.37.0`) - ASGI server entrypoint for `src/api.py`.
- LangChain stack (`langchain`, `langchain-core`, `langchain-community`, `langchain-text-splitters`) - prompt/chains/retrieval composition in `src/main.py`, `src/retriever.py`, `src/ui_backend.py`.
- Chroma (`chromadb==1.1.1` + `langchain_community.vectorstores.Chroma`) - vector persistence under `vectorstore/chroma` (`src/config.py`, `src/ingest.py`).
- NetworkX (`networkx==3.5`) - SymMap graph operations (`src/graph_builder.py`, `src/retriever.py`).

**LLM provider adapters:**
- `langchain-openai` - OpenAI plus OpenAI-compatible endpoints (DashScope, Ollama, LM Studio) in `src/main.py`.
- `langchain-google-genai` - Gemini support in `src/main.py`.
- `langchain-anthropic` - Claude support in `src/main.py`.
- `langchain-community` - OpenRouter and Together adapters in `src/main.py`.
- `dashscope==1.24.0` is installed for Alibaba ecosystem support (`requirements.txt`).

**Frontend:**
- Next.js `16.1.6` + React `19.2.3` (`web/package.json`).
- Tailwind CSS v4 + PostCSS plugin (`@tailwindcss/postcss`) (`web/package.json`).
- ESLint v9 + Next config (`web/eslint.config.mjs`).
- Notable UI libs: `@xyflow/react`, `react-markdown`, `remark-gfm`, `lucide-react`, `clsx`, `tailwind-merge` (`web/package.json`).

**Optional prototype UI:**
- Streamlit (`streamlit==1.39.0`) for lightweight demo app in `src/ui_app.py`.

## Key Dependencies

**Critical:**
- `python-dotenv` - environment loading at startup (`src/main.py`, `src/api.py`, `src/ui_backend.py`, `scripts/test_tongyi.py`).
- `pydantic` + `pydantic-settings` - request/response schemas and settings handling (`src/api.py`, deps list).
- `sentence-transformers`, `transformers`, `torch`, `huggingface-hub` - embedding/model runtime used by `HuggingFaceEmbeddings` in `src/main.py`, `src/retriever.py`, `src/ingest.py`.
- `openpyxl` - SymMap XLSX processing (`scripts/fetch_symmap_v2.py`).

**Infrastructure and transport:**
- `requests` - SymMap remote download and edge fetch (`scripts/fetch_symmap_v2.py`).
- `httpx`/`httpx-sse` and websocket libs are present for HTTP streaming ecosystem support (`requirements.txt`).

## Configuration

**Environment surface:**
- Core model/retrieval settings are env-driven (`docs/CONFIG.md`, `src/main.py`, `src/ui_backend.py`, `src/api.py`).
- Central file/path defaults live in `src/config.py` (`VECTORSTORE_DIR`, `CHUNKS_PATH`, `GRAPH_DATA_DEFAULT_RELATIVE`).
- Frontend backend target is env-driven in `web/app/api/backend/[...path]/route.ts` via `BACKEND_URL` / `NEXT_PUBLIC_BACKEND_URL`.

**Build and lint configs:**
- Next.js config: `web/next.config.ts` (explicit Turbopack root).
- ESLint config: `web/eslint.config.mjs`.
- No root `pyproject.toml` or frontend monorepo build orchestrator detected.

## Commands In Active Use

**Backend/dev (documented):**
- `venv\Scripts\python.exe src/ingest.py` - build vector index (`AGENTS.md`).
- `venv\Scripts\python.exe src/api.py` - run FastAPI backend (`AGENTS.md`).
- `venv\Scripts\python.exe src/main.py` - run CLI RAG flow (`AGENTS.md`).

**Frontend/dev:**
- `cd web && npm install && npm run dev` - Next.js dev server (`AGENTS.md`, `web/package.json`).
- `cd web && npm run lint` - ESLint (`AGENTS.md`, `web/package.json`).
- `cd web && npm run build && npm run start` - production Next build/start (`web/package.json`).

**Script-style testing:**
- `venv\Scripts\python.exe src/test_citations.py`
- `venv\Scripts\python.exe src/test_graph.py`
- `venv\Scripts\python.exe scripts/verify_symmap_retrieval.py`

## Platform Requirements

**Development:**
- Windows and Unix are supported with venv conventions in `AGENTS.md`.
- Network access is required for cloud LLMs and first-time Hugging Face model pulls.
- Local-first option exists with Ollama/LM Studio endpoints configured in env (`src/main.py`).

**Production:**
- Backend serves via Uvicorn on `0.0.0.0:${PORT}` (`src/api.py`).
- Frontend deploys as a standard Next.js app (`web/package.json` scripts).
- Deployment platform is not prescribed in repository files.

---

*Stack analysis: 2026-04-01*
