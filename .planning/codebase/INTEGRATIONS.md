# External Integrations

**Analysis Date:** 2026-03-23

## APIs & External Services

**LLM Providers:**
- **Alibaba Cloud DashScope**: Used for Qwen models.
  - SDK/Client: `dashscope` (Python SDK)
  - Auth: `DASHSCOPE_API_KEY`
- **OpenAI API**: Used for GPT-4o, etc.
  - SDK/Client: `langchain-openai`
  - Auth: `OPENAI_API_KEY`
- **Google Generative AI**: Used for Gemini models.
  - SDK/Client: `langchain-google-genai`
  - Auth: `GOOGLE_API_KEY`
- **Anthropic Claude**: Used for Claude models.
  - SDK/Client: `langchain-anthropic`
  - Auth: `ANTHROPIC_API_KEY`
- **OpenRouter/Together AI**: Meta-providers for various open-source models.
  - SDK/Client: `langchain-openai` (compatible)
  - Auth: `OPENROUTER_API_KEY`, `TOGETHER_API_KEY`

## Data Storage

**Databases:**
- **ChromaDB**: Local, persistent vector database.
  - Connection: `VECTORSTORE_DIR` (from `src/config.py`)
  - Client: `chromadb` (Python library)

**File Storage:**
- **Local Filesystem**: Used for storing processed document chunks (`data/processed/chunks.json`) and Knowledge Graph entity data (`data/graph/entities.json`).

**Caching:**
- **Python lru_cache**: Used in `src/api.py` for `load_chunks_data()`.
- **Frontend State**: `web/hooks/useHistory.ts` manages chat history persistence in the browser session.

## Authentication & Identity

**Auth Provider:**
- **None/Custom**: The application is currently designed for internal use and does not implement a formal auth provider like Auth0 or NextAuth.js.

## Monitoring & Observability

**Error Tracking:**
- **None**: No external service like Sentry or LogRocket is integrated.

**Logs:**
- **Standard Console Logs**: Backend uses standard logging and `coloredlogs` for console output.

## CI/CD & Deployment

**Hosting:**
- **Local/Self-hosted**: The codebase is designed to run locally for development and testing.

**CI Pipeline:**
- **GitHub Actions**: Workflows in `.github/workflows/` handle automated triage, review, and potentially deployment tasks.

## Environment Configuration

**Required env vars:**
- `LLM_PROVIDER`: The primary LLM provider to use (e.g., `alibaba`).
- `DASHSCOPE_API_KEY` / `OPENAI_API_KEY`: API keys for the chosen provider.
- `NEXT_PUBLIC_BACKEND_URL`: Used by the frontend to communicate with the FastAPI backend.

**Secrets location:**
- `.env` (Backend, ignored by git)
- `web/.env.local` (Frontend, ignored by git)

## Webhooks & Callbacks

**Incoming:**
- **SSE Endpoints**: `/query` on the backend provides a Server-Sent Events stream for real-time LLM responses.

**Outgoing:**
- **None detected**.

---

*Integration audit: 2026-03-23*
