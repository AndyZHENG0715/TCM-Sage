<!-- generated-by: gsd-doc-writer -->
# Development Guide

This guide covers local setup, project commands, code conventions, and contribution workflow for TCM-Sage.

---

## Local Setup

TCM-Sage has two separate runtimes — a Python backend and a Node.js frontend — that must both be running for full functionality.

### 1. Clone and enter the repo

```bash
git clone https://github.com/AndyZHENG0715/TCM-Sage.git
cd TCM-Sage
```

### 2. Create and activate the Python virtual environment

```bash
# Create venv (one-time)
python -m venv venv
```

**Always use the project venv.** Never use the system `python` or bare `pip`.

```bash
# Windows
venv\Scripts\python.exe -m pip install -r requirements.txt

# macOS / Linux
venv/bin/python -m pip install -r requirements.txt
```

### 3. Configure environment variables

Copy the example env file and fill in your credentials:

```bash
cp .env.example .env        # root-level: backend config
cp web/.env.local.example web/.env.local   # frontend: backend URL override
```

Minimum viable `.env`:
```env
LLM_PROVIDER=alibaba
DASHSCOPE_API_KEY=your-api-key-here
```

See [`docs/CONFIGURATION.md`](CONFIGURATION.md) for the full variable reference.

### 4. Install frontend dependencies

```bash
cd web
npm install
cd ..
```

### 5. Build the vector index (first run only)

```bash
# Windows
venv\Scripts\python.exe src/ingest.py

# macOS / Linux
venv/bin/python src/ingest.py
```

Ingestion supports checkpoint/resume — safe to interrupt and re-run. To force a full rebuild first clear the checkpoint with `venv\Scripts\python.exe scripts/reset_tracking.py`.

---

## Running for Development

Start both processes in separate terminals:

**Terminal 1 — Backend (FastAPI at `http://127.0.0.1:8000`)**
```bash
# Windows
venv\Scripts\python.exe src/api.py

# macOS / Linux
venv/bin/python src/api.py
```

**Terminal 2 — Frontend (Next.js at `http://localhost:3000`)**
```bash
cd web
npm run dev
```

The frontend proxies all API calls through `/api/backend/...` — direct calls to `localhost:8000` from the browser are not used.

---

## Build Commands

### Backend (Python)

| Command | Description |
|---------|-------------|
| `venv\Scripts\python.exe src/api.py` | Start FastAPI dev server (port 8000) |
| `venv\Scripts\python.exe src/main.py` | Run CLI RAG pipeline interactively |
| `venv\Scripts\python.exe src/ingest.py` | Build / refresh the ChromaDB vector index |

### Frontend (Next.js)

All frontend commands run from the `web/` directory:

| Command | Description |
|---------|-------------|
| `npm run dev` | Start Next.js dev server with Turbopack (port 3000) |
| `npm run build` | Production build |
| `npm run start` | Start production server (requires prior build) |
| `npm run lint` | Run ESLint (eslint-config-next, TypeScript rules) |

### Presentation (optional)

```bash
cd presentation
npx slidev --port 3030   # Dev server
npx slidev export        # Export to PDF
```

---

## Code Style

### Python (backend)

- **Naming:** `snake_case.py` files, `snake_case` functions and variables.
- **Types:** Type hints required on all public functions. Pydantic models for FastAPI request/response bodies.
- **Imports:** `src/` is **not** a Python package. New scripts must bootstrap the path before importing local modules:

  ```python
  import sys, os
  sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
  ```

- **Type checker:** Pyright (configured in `pyrightconfig.json`). Python version: `3.13`. Never suppress types with `Any` casts — fix them properly.
- **Env vars:** Read through `src/config.py` constants or `os.getenv()` with defaults. Never hardcode paths.

### Frontend (TypeScript/React)

- **Files:** `PascalCase.tsx` for components, `camelCase.ts` for utilities and hooks.
- **Imports:** Use `@/` path aliases (`@/lib/...`, `@/components/...`). Order: external → `@/` aliases → relative.
- **Types:** Export shared types from `web/lib/types.ts`. Keep in sync with `src/citation_types.py`.
- **Strict mode:** TypeScript strict mode is enabled — no implicit `any`.
- **Linting:** ESLint 9 with `eslint-config-next` (Core Web Vitals + TypeScript rules). Config: `web/eslint.config.mjs`.
- **No barrel files:** Import from concrete paths, not `index.ts` re-exports.

---

## Critical Development Patterns

These patterns are enforced by the codebase and must be followed to avoid runtime regressions.

### Prompt template structure (Python)

Always use `ChatPromptTemplate.from_messages()` with **separate** `SystemMessage` and user message objects. Merging them into a single template string causes the LLM to lose output formatting:

```python
# ✅ Correct
from langchain.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}\n\nContext:\n{context}"),
])

# ❌ Wrong — merges system + user, breaks formatting
prompt = ChatPromptTemplate.from_template("{system}\n\n{question}\n\n{context}")
```

### Context formatting (Pattern Priming)

RAG context chunks sent to the LLM **must** use markdown formatting (headers, blockquotes). Plain text context causes the LLM to output plain text responses (Pattern Priming effect):

```python
# ✅ Correct — markdown context
context = "### 伤寒论·第12条\n> 太阳中风，阳浮而阴弱..."

# ❌ Wrong — plain text context
context = "伤寒论第12条 太阳中风，阳浮而阴弱..."
```

### LLM response output

Do **not** instruct the LLM to append "Sources:", "References:", or similar lists to its output. The frontend UI strips these patterns, and their presence creates visual artifacts in the citation panel.

### Knowledge graph context labels

Do **not** use the strings `"KG"` or `"Knowledge Graph"` in any context passed to the LLM. These labels reveal the RAG system identity and invalidate Arena blind evaluation results.

### DashScope batch limits

The DashScope embedding API (`text-embedding-v4`) has a **batch limit of 10 texts**. The `DashScopeEmbeddings` class in `src/embeddings.py` handles embedding batching automatically during ingestion. The `qwen3-rerank` endpoint supports up to 500 documents per request, but keep request sizes bounded to the configured retrieval window.

### Frontend API calls

All backend calls from the frontend must go through the `/api/backend/` Next.js proxy route — never call `http://127.0.0.1:8000` directly from browser-side code.

---

## Where to Add Code

### Adding a new Python feature

| Feature type | Location |
|---|---|
| New retrieval / ranking logic | Extend `src/retriever.py` (`HybridRetriever`) |
| New FastAPI endpoint | Add the route wrapper in `src/api.py`; keep reusable logic in a focused helper module; mirror client call in `web/lib/api.ts` |
| Source/book route behavior | `src/source_context.py` |
| New graph schema or import | `src/graph_builder.py` + utility script in `scripts/` |
| New arena endpoint | `src/api.py` route wrapper + `src/arena.py` generation + `src/arena_stream.py` streaming + `src/arena_stats.py` statistics |
| New standalone test | `src/test_<feature>.py` (co-located) |

### Adding a new frontend feature

| Feature type | Location |
|---|---|
| New page / route | `web/app/` (Next.js App Router) |
| New UI component | `web/components/PascalCase.tsx` |
| New React hook | `web/hooks/useX.ts` |
| New API client function | `web/lib/api.ts` |
| Shared TypeScript types | `web/lib/types.ts` |

---

## Testing

There is no pytest. Tests are standalone scripts run directly with the project venv. See [`docs/TESTING.md`](TESTING.md) if generated, or the test commands below.

### Backend test scripts

```bash
# Citation formatting and reconstruction
venv\Scripts\python.exe src/test_citations.py

# SymMap knowledge graph retrieval
venv\Scripts\python.exe src/test_graph.py

# Hybrid retriever (vector + graph ensemble)
venv\Scripts\python.exe src/test_hybrid_retriever.py

# Arena blind evaluation logic
venv\Scripts\python.exe src/test_arena.py

# KG entity extraction
venv\Scripts\python.exe src/test_kg_extraction.py

# Retriever unit checks
venv\Scripts\python.exe src/test_retriever.py
```

### RAG quality scripts (`scripts/`)

```bash
# Compare RAG vs plain LLM answers
venv\Scripts\python.exe scripts/test_llm_vs_rag.py

# Measure answer structure and length quality
venv\Scripts\python.exe scripts/test_answer_length.py

# Validate context formatting impact (Pattern Priming)
venv\Scripts\python.exe scripts/test_context_format.py

# SymMap KG retrieval sanity check
venv\Scripts\python.exe scripts/verify_symmap_retrieval.py

# Verify /context API endpoint response shape
venv\Scripts\python.exe scripts/verify_context_endpoint.py

# Arena vote write concurrency stress test
venv\Scripts\python.exe scripts/test_arena_concurrency.py
```

### Frontend lint

```bash
cd web
npm run lint
```

---

## Branch Conventions

No formal convention is documented. Follow descriptive branch names prefixed by type, e.g.:

- `feat/arena-statistics` — new feature
- `fix/sse-disconnect` — bug fix
- `refactor/retriever-cleanup` — code cleanup
- `docs/development-guide` — documentation only

The default branch is `main`.

---

## PR Process

1. Branch from `main` with a descriptive name.
2. Keep changes focused — one logical concern per PR.
3. Verify backend changes with relevant `test_*.py` scripts before opening a PR.
4. Run `npm run lint` in `web/` and resolve any ESLint errors.
5. Reference the relevant issue or feature description in the PR body.
6. Request review; address all comments before merging.

There is no automated CI pipeline. All verification is manual.

---

## Useful References

| Document | Purpose |
|----------|---------|
| [`docs/CONFIGURATION.md`](CONFIGURATION.md) | Full environment variable reference |
| [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) | System design and component overview |
| `src/AGENTS.md` | Python backend conventions and data flow |
| `web/AGENTS.md` | Next.js frontend conventions |
| `scripts/AGENTS.md` | Utility script catalogue |
