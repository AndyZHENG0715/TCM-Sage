<!-- generated-by: gsd-doc-writer -->
# Getting Started with TCM-Sage

This guide walks first-time users through every step needed to run TCM-Sage locally — from installing prerequisites to asking your first query in the web interface.

---

## Prerequisites

Before cloning the repository, ensure the following tools are installed and on your `PATH`.

| Tool | Required version | Notes |
|------|-----------------|-------|
| **Python** | `3.13.x` | Confirmed with `python --version`. Earlier versions are untested. |
| **Node.js** | `>= 22.x` | Confirmed with `node --version`. Needed for the Next.js frontend. |
| **npm** | `>= 10.x` | Ships with Node 22. |
| **Git** | Any recent | For cloning the repository. |

> **Windows note:** The commands in this guide use the Windows venv path (`venv\Scripts\python.exe`). On macOS/Linux substitute `venv/bin/python` everywhere.

---

## Step 1 — Clone the repository

```bash
git clone https://github.com/AndyZHENG0715/TCM-Sage.git
cd TCM-Sage
```

---

## Step 2 — Create and activate the Python virtual environment

TCM-Sage requires its own isolated virtual environment. **Never use a bare `python` or `pip` command** — always invoke the venv interpreter directly.

```bash
# Create the venv (run once)
python -m venv venv
```

Install all Python dependencies into the venv:

```bash
# Windows
venv\Scripts\python.exe -m pip install -r requirements.txt

# macOS / Linux
venv/bin/python -m pip install -r requirements.txt
```

This installs FastAPI, LangChain, ChromaDB, DashScope SDK, and all other backend dependencies listed in `requirements.txt`.

---

## Step 3 — Install frontend dependencies

```bash
cd web
npm install
cd ..
```

---

## Step 4 — Configure environment variables

Copy the example file and fill in your credentials:

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Open `.env` in your editor. The minimum required settings are:

```bash
LLM_PROVIDER=alibaba
DASHSCOPE_API_KEY=your-api-key-here
```

### Choosing a provider

| Provider | Variable | Notes |
|----------|----------|-------|
| **Alibaba DashScope** (recommended) | `DASHSCOPE_API_KEY` | Free-tier tokens available. Embeddings (`text-embedding-v4`) and reranker (`qwen3-rerank`) both require DashScope regardless of the main LLM provider. |
| OpenAI | `OPENAI_API_KEY` | Set `LLM_PROVIDER=openai` |
| Google Gemini | `GOOGLE_API_KEY` | Set `LLM_PROVIDER=google` |
| Anthropic | `ANTHROPIC_API_KEY` | Set `LLM_PROVIDER=anthropic` |
| OpenRouter | `OPENROUTER_API_KEY` | Set `LLM_PROVIDER=openrouter` |
| Together AI | `TOGETHER_API_KEY` | Set `LLM_PROVIDER=together` |
| **Ollama** (local, free) | *(none)* | Set `LLM_PROVIDER=ollama`. Requires Ollama running at `http://localhost:11434`. |
| **LM Studio** (local, free) | *(none)* | Set `LLM_PROVIDER=lmstudio`. Requires LM Studio running at `http://localhost:1234`. |

> **Important:** The DashScope API key is always needed for embeddings and reranking, even when using a different LLM provider. Only Ollama and LM Studio can run fully offline without any API keys.

The frontend also reads its own environment file. Copy the example:

```bash
# Windows
copy web\.env.local.example web\.env.local

# macOS / Linux
cp web/.env.local.example web/.env.local
```

The default `NEXT_PUBLIC_BACKEND_URL=http://localhost:8000` works for local development with no changes required.

See `docs/CONFIGURATION.md` for the full list of environment variables.

---

## Step 5 — Build the vector knowledge base

This step ingests the 17 classical TCM texts from `data/source/`, generates embeddings, and writes the ChromaDB vectorstore to `vectorstore/chroma/`. It takes several minutes on first run but supports checkpoint/resume — you can safely interrupt and re-run.

```bash
# Windows
venv\Scripts\python.exe src/ingest.py

# macOS / Linux
venv/bin/python src/ingest.py
```

**Skip this step** if the `vectorstore/chroma/` directory is already populated (e.g., you received a pre-built vectorstore).

> **Heads up:** Ingestion calls the DashScope embedding API in batches of 10 chunks at a time. A free-tier DashScope account has generous limits but very large re-ingestion runs may hit rate limits. The checkpoint file at `data/processed/ingest_checkpoint.json` ensures re-runs resume from where they left off.

---

## Step 6 — Start the backend API

```bash
# Windows
venv\Scripts\python.exe src/api.py

# macOS / Linux
venv/bin/python src/api.py
```

The FastAPI server starts on **`http://127.0.0.1:8000`**. You should see Uvicorn startup output like:

```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

Leave this terminal open and open a second terminal for the next step.

---

## Step 7 — Start the frontend dev server

```bash
cd web
npm run dev
```

The Next.js development server starts on **`http://localhost:3000`**. Open that URL in your browser.

---

## Step 8 — Verify it works

With both servers running, open [http://localhost:3000](http://localhost:3000) and try asking a TCM question in Chinese or English, for example:

> 风寒感冒的主要症状和治疗原则是什么？

You should see:
- Streaming text appearing word-by-word in the chat window
- A citation panel on the right listing the source chapter(s)
- A verification badge indicating answer faithfulness

You can also check the backend health endpoint directly:

```bash
curl http://127.0.0.1:8000/health
# Expected: {"status":"ok"}
```

---

## Common First-Run Issues

### `DASHSCOPE_API_KEY` not set

**Symptom:** Backend starts but queries fail with an authentication error, or the ingestion script exits immediately with a key error.

**Fix:** Ensure `.env` exists in the project root (not inside `src/`) and contains `DASHSCOPE_API_KEY=<your-key>`. The `.env` file is loaded from the working directory at startup.

---

### Wrong Python interpreter / `ModuleNotFoundError`

**Symptom:** Running `python src/api.py` raises `ModuleNotFoundError: No module named 'fastapi'`.

**Fix:** You are using the system Python instead of the venv. Always use the full path:

```bash
# Correct — uses venv Python
venv\Scripts\python.exe src/api.py

# Incorrect — uses system Python
python src/api.py
```

---

### Frontend cannot reach the backend (`fetch failed`)

**Symptom:** The chat UI shows a network error or spinner that never resolves.

**Fix 1:** Confirm the backend is running and listening on port 8000.

**Fix 2:** Confirm `web/.env.local` contains `NEXT_PUBLIC_BACKEND_URL=http://localhost:8000`. If the file is missing, copy it from `web/.env.local.example`.

**Fix 3:** Restart the Next.js dev server after changing `.env.local` — Next.js reads environment files only at startup.

---

### Port 3000 or 8000 already in use

**Symptom:** `EADDRINUSE` (Node) or `[Errno 10048]` (Windows) / `[Errno 98]` (Linux) on startup.

**Fix (backend):** Pass a different port via Uvicorn arguments or kill the process occupying port 8000.

**Fix (frontend):** Next.js will automatically try port 3001, 3002, etc. if 3000 is taken — check the terminal output for the actual URL.

---

### Ingestion hangs or rate-limit error

**Symptom:** `ingest.py` stalls at a particular chunk batch or raises a DashScope rate-limit exception.

**Fix:** Press `Ctrl+C` to interrupt and re-run the same command. The checkpoint at `data/processed/ingest_checkpoint.json` records completed chunks; re-runs skip already-embedded batches automatically.

---

### `chunks.json` not found after ingestion

**Symptom:** The backend raises a `FileNotFoundError` for `data/processed/chunks.json` on startup.

**Fix:** Run `src/ingest.py` to completion before starting `src/api.py`. The processed chunk file is a required artefact written by the ingestion pipeline.

---

## Next Steps

| Goal | Where to look |
|------|---------------|
| Full environment variable reference | `docs/CONFIGURATION.md` |
| System architecture and component overview | `docs/ARCHITECTURE.md` |
| Arena blind A/B evaluation | Navigate to `/arena` in the running web UI |
| Knowledge Graph explorer | Navigate to `/kg/<entityId>` (links appear in citation panel) |
| CLI mode (no browser required) | `venv\Scripts\python.exe src/main.py` |
| Presentation slides | `cd presentation && npx slidev --port 3030` |
