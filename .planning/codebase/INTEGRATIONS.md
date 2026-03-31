# External Integrations

**Analysis Date:** 2026-04-01

## APIs & External Services

**LLM providers (runtime selectable):**
- OpenAI via `langchain_openai.ChatOpenAI` in `src/main.py`
  - SDK/Client: `langchain-openai`
  - Auth: `OPENAI_API_KEY`
- Google Gemini via `langchain_google_genai.ChatGoogleGenerativeAI` in `src/main.py`
  - SDK/Client: `langchain-google-genai`
  - Auth: `GOOGLE_API_KEY`
- Anthropic via `langchain_anthropic.ChatAnthropic` in `src/main.py`
  - SDK/Client: `langchain-anthropic`
  - Auth: `ANTHROPIC_API_KEY`
- OpenRouter via `langchain_community.llms.OpenRouter` in `src/main.py`
  - SDK/Client: `langchain-community`
  - Auth: `OPENROUTER_API_KEY`
- Together via `langchain_community.llms.Together` in `src/main.py`
  - SDK/Client: `langchain-community`
  - Auth: `TOGETHER_API_KEY`
- Alibaba DashScope (OpenAI-compatible endpoint) in `src/main.py`
  - SDK/Client: `langchain-openai` (`ChatOpenAI` with `base_url=https://dashscope-intl.aliyuncs.com/compatible-mode/v1`)
  - Auth: `DASHSCOPE_API_KEY`
- Ollama local endpoint in `src/main.py`
  - SDK/Client: `langchain-openai`
  - Auth: no real key required (uses fixed placeholder), endpoint from `OLLAMA_BASE_URL`
- LM Studio local endpoint in `src/main.py`
  - SDK/Client: `langchain-openai`
  - Auth: no real key required (uses fixed placeholder), endpoint from `LMSTUDIO_BASE_URL`

**Knowledge source integration:**
- SymMap public dataset and related API consumed by `scripts/fetch_symmap_v2.py`
  - Endpoints: `http://www.symmap.org/static/download/V2.0/` and `http://www.symmap.org/related_components/`
  - Client: `requests`
  - Auth: none

**Frontend-backend bridge:**
- Next.js catch-all proxy at `web/app/api/backend/[...path]/route.ts`
  - Forwards to `BACKEND_URL` or `NEXT_PUBLIC_BACKEND_URL` (fallback `http://127.0.0.1:8000`)
  - Preserves query strings and supports streaming request bodies (`duplex: "half"`)

## Data Storage

**Databases:**
- Local embedded vector database (Chroma)
  - Connection: filesystem path `vectorstore/chroma` (`src/config.py` `VECTORSTORE_DIR`)
  - Client: `langchain_community.vectorstores.Chroma`

**File Storage:**
- Local filesystem only
  - Corpus: `data/source/`
  - Chunk metadata/content: `data/processed/chunks.json`
  - Graph JSON: `data/graph/symmap/symmap_entities.json` (default via `src/config.py`, override with `GRAPH_DATA_PATH`)
  - SymMap raw exports: `data/graph/symmap/raw/`

**Caching:**
- In-process cache via `functools.lru_cache` in `src/api.py` and `src/ui_backend.py`
- Hugging Face model cache via `HF_HOME`/default cache path (used in `src/ui_backend.py`)

## Authentication & Identity

**Auth Provider:**
- Custom/no user identity layer in `src/api.py` (no JWT/session middleware)
  - Implementation: API-key based provider access only for outbound LLM calls

## Monitoring & Observability

**Error Tracking:**
- None detected as an active external service integration in runtime code paths

**Logs:**
- Application-level print/exception logs in Python modules (`src/api.py`, `src/ui_backend.py`, `src/main.py`)
- Server logs from Uvicorn when running `src/api.py`

## CI/CD & Deployment

**Hosting:**
- Not fixed in repo; backend is deployable as Uvicorn service (`src/api.py`), frontend as standard Next.js app (`web/package.json`)

**CI Pipeline:**
- Not detected (`.github/workflows/` not present)

## Environment Configuration

**Required env vars (integration-relevant):**
- Provider selection and model routing: `LLM_PROVIDER`, `LLM_MODEL`, `CLASSIFIER_LLM_PROVIDER`, `CLASSIFIER_LLM_MODEL`, `VERIFIER_LLM_PROVIDER`, `VERIFIER_LLM_MODEL`
- Provider auth (choose by provider): `DASHSCOPE_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, `TOGETHER_API_KEY`
- Local inference endpoints: `OLLAMA_BASE_URL`, `LMSTUDIO_BASE_URL`
- Retrieval graph plumbing: `HYBRID_RETRIEVAL_ENABLED`, `GRAPH_DATA_PATH`, `GRAPH_DEPTH`, `RETRIEVAL_K`
- API/network surface: `PORT`, `ALLOWED_ORIGINS`, `BACKEND_URL`, `NEXT_PUBLIC_BACKEND_URL`
- Optional Hugging Face cache controls: `HF_HOME`, `XDG_CACHE_HOME`, `HF_LOCAL_FILES_ONLY`
- Optional UI integration: `FEEDBACK_FORM_URL` (used by `src/ui_app.py`)

**Secrets location:**
- `.env` in project root (present); `.env.example` as template

## Webhooks & Callbacks

**Incoming:**
- None detected

**Outgoing:**
- Outbound HTTP calls to selected LLM provider endpoints from `src/main.py`
- Outbound SymMap download/API calls from `scripts/fetch_symmap_v2.py`

---

*Integration audit: 2026-04-01*
