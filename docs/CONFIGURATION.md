<!-- generated-by: gsd-doc-writer -->
# TCM-Sage: Configuration Guide

This document covers every configurable aspect of the TCM-Sage system — LLM providers, retrieval tuning, knowledge graph, arena evaluation, frontend integration, and operational settings. Create a `.env` file in the project root and set values as described below.

## Quick Start

```bash
# 1. Copy the example configuration
cp .env.example .env

# 2. Edit .env with your provider and API key (minimum viable config)
LLM_PROVIDER=alibaba
DASHSCOPE_API_KEY=your-actual-api-key-here

# 3. Install dependencies (use project venv)
venv\Scripts\python.exe -m pip install -r requirements.txt   # Windows
venv/bin/python -m pip install -r requirements.txt            # macOS/Linux

# 4. Build the vector index (run once)
venv\Scripts\python.exe src/ingest.py

# 5. Start the backend API
venv\Scripts\python.exe src/api.py

# 6. Start the frontend (separate terminal)
cd web && npm install && npm run dev
```

## Environment Variables — Complete Reference

### Core LLM Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LLM_PROVIDER` | Yes | `alibaba` | LLM provider ID. One of: `alibaba`, `openai`, `google`, `anthropic`, `openrouter`, `together`, `ollama`, `lmstudio` |
| `LLM_MODEL` | No | Per-provider (see below) | Override the default model for the selected provider |
| `LLM_TEMPERATURE` | No | `0.1` | Temperature for informational queries. Range: 0.0–1.0 |
| `PRESCRIPTIVE_TEMPERATURE` | No | `0.0` | Temperature for prescriptive/diagnostic queries. Keep at 0.0 for medical accuracy |
| `SYSTEM_PROMPT_OVERRIDE` | No | — | Overrides the built-in Chinese TCM clinical reference system prompt in `src/main.py` |

### Classifier LLM Configuration

A separate lightweight LLM classifies each query as _informational_ or _prescriptive_ to select the appropriate temperature. When no classifier-specific variables are set, the classifier follows the main provider/model.

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `CLASSIFIER_LLM_PROVIDER` | No | Same as `LLM_PROVIDER` | Provider for the classification model |
| `CLASSIFIER_LLM_MODEL` | No | Same as `LLM_MODEL` | Lightweight model for fast classification (recommended: `qwen-flash`, `qwen3-0.6b`) |
| `CLASSIFIER_LLM_TEMPERATURE` | No | `0.0` | Keep at 0.0 for consistent classification results |

### Verifier LLM Configuration

After answer generation, a verifier LLM performs a self-critique check for faithfulness against retrieved context. When no verifier-specific variables are set, it follows the main provider/model.

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `VERIFIER_LLM_PROVIDER` | No | Same as `LLM_PROVIDER` | Provider for the verification model |
| `VERIFIER_LLM_MODEL` | No | Same as `LLM_MODEL` | Model for answer-faithfulness checking (recommended: `qwen-flash`) |
| `VERIFIER_LLM_TEMPERATURE` | No | `0.0` | Keep at 0.0 for deterministic verification |
| `VERIFICATION_PROMPT` | No | Built-in prompt | Custom verification prompt template. Must contain `{context}`, `{question}`, and `{answer}` placeholders |

### Retrieval Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `RETRIEVAL_K` | No | `5` | Number of document chunks to retrieve per query. Range: 1–20 recommended |
| `HYBRID_RETRIEVAL_ENABLED` | No | `true` | Enable hybrid vector + knowledge graph retrieval |
| `GRAPH_DATA_PATH` | No | `data/graph/symmap/symmap_entities.json` | Path to the SymMap 2.0 knowledge graph JSON file |
| `GRAPH_DEPTH` | No | `1` (CLI) / `1` (API) | Max traversal depth for graph search. 1–2 recommended |
| `GRAPH_MAX_RESULTS` | No | `20` | Maximum graph entity results per query |

### Embedding & Reranker Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DASHSCOPE_API_KEY` | Yes (for embeddings & reranker) | — | API key for DashScope text-embedding-v4 and qwen3-rerank |
| `DASHSCOPE_EMBEDDING_API_URL` | No | `https://dashscope-intl.aliyuncs.com/api/v1` | DashScope API base URL for embeddings and reranker |

### Provider API Keys

Only set the key for the provider you are using.

| Variable | Provider | Required |
|----------|----------|----------|
| `DASHSCOPE_API_KEY` | Alibaba Cloud (DashScope) | Yes if `LLM_PROVIDER=alibaba` or for embeddings/reranker |
| `OPENAI_API_KEY` | OpenAI | Yes if `LLM_PROVIDER=openai` |
| `GOOGLE_API_KEY` | Google AI Studio | Yes if `LLM_PROVIDER=google` |
| `ANTHROPIC_API_KEY` | Anthropic | Yes if `LLM_PROVIDER=anthropic` |
| `OPENROUTER_API_KEY` | OpenRouter | Yes if `LLM_PROVIDER=openrouter` |
| `TOGETHER_API_KEY` | Together AI | Yes if `LLM_PROVIDER=together` |

### Local LLM Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OLLAMA_BASE_URL` | No | `http://localhost:11434/v1` | Ollama local server OpenAI-compatible endpoint |
| `LMSTUDIO_BASE_URL` | No | `http://localhost:1234/v1` | LM Studio local server OpenAI-compatible endpoint |

### Arena Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ARENA_MODELS` | No | `{"flash":"qwen-flash","plus":"qwen-plus","max":"qwen-max"}` | JSON string overriding the arena tier-to-model mapping |
| `ARENA_STREAM_TIMEOUT_SECONDS` | No | `60` | Timeout in seconds for each arena SSE panel stream |

### Web Frontend / API Integration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ALLOWED_ORIGINS` | No | `*` | Comma-separated CORS whitelist for the FastAPI backend |
| `PORT` | No | `8000` | Port the FastAPI server listens on |
| `BACKEND_URL` | No | `http://127.0.0.1:8000` | Server-side backend URL for the Next.js proxy route (set in `web/.env.local`) |
| `NEXT_PUBLIC_BACKEND_URL` | No | `http://127.0.0.1:8000` | Client-visible fallback backend URL (set in `web/.env.local`) |
| `FEEDBACK_FORM_URL` | No | — | URL for the Google Form feedback link shown in the UI |

### Crosswalk Bridge Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `CROSSWALK_APPROVED_PATH` | No | `data/graph/crosswalk/seed_crosswalk_approved.csv` | Path to the approved RAG↔SymMap entity crosswalk CSV |

### Output Format Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OUTPUT_FORMAT` | No | `detailed` | Response format: `detailed`, `concise`, `academic` |
| `CITATION_STYLE` | No | `chapter` | Citation display format: `chapter`, `page`, `section` |

## Supported LLM Providers

### 1. Alibaba Cloud Model Studio (Recommended)

**Default Provider** — Cost-effective, excellent Chinese language support.

- **Provider ID**: `alibaba`
- **Default Model**: `qwen-plus`
- **API Key**: `DASHSCOPE_API_KEY`
- **Base URL**: `https://dashscope-intl.aliyuncs.com/compatible-mode/v1` (international endpoint)
- **Setup**:
  1. Sign up at [Alibaba Cloud Model Studio](https://dashscope.aliyuncs.com/)
  2. Create an API key in the DashScope console
  3. Set `LLM_PROVIDER=alibaba` and your `DASHSCOPE_API_KEY` in `.env`

> **Note:** The same `DASHSCOPE_API_KEY` is used for embeddings (text-embedding-v4) and reranking (qwen3-rerank), so this key is effectively required for all providers.

### 2. OpenAI

- **Provider ID**: `openai`
- **Default Model**: `gpt-5-4`
- **API Key**: `OPENAI_API_KEY`
- **Required Package**: `langchain-openai`
- **Setup**: Get API key from [OpenAI Platform](https://platform.openai.com/)

### 3. Google AI Studio

- **Provider ID**: `google`
- **Default Model**: `gemini-3-1-pro`
- **API Key**: `GOOGLE_API_KEY`
- **Required Package**: `langchain-google-genai`
- **Setup**: Get API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

### 4. Anthropic (Claude)

- **Provider ID**: `anthropic`
- **Default Model**: `claude-4-6-sonnet`
- **API Key**: `ANTHROPIC_API_KEY`
- **Required Package**: `langchain-anthropic`
- **Setup**: Get API key from [Anthropic Console](https://console.anthropic.com/)

### 5. OpenRouter

- **Provider ID**: `openrouter`
- **Default Model**: `openai/gpt-5-4`
- **API Key**: `OPENROUTER_API_KEY`
- **Required Package**: `langchain-community`
- **Setup**: Get API key from [OpenRouter](https://openrouter.ai/)

### 6. Together AI

- **Provider ID**: `together`
- **Default Model**: `meta-llama/Llama-3.1-8B-Instruct-Turbo`
- **API Key**: `TOGETHER_API_KEY`
- **Required Package**: `langchain-community`
- **Setup**: Get API key from [Together AI](https://together.ai/)

### 7. Ollama (Local)

- **Provider ID**: `ollama`
- **Default Model**: `qwen3:8b`
- **API Key**: Not required (uses dummy key internally)
- **Base URL**: `OLLAMA_BASE_URL` (default: `http://localhost:11434/v1`)
- **Required Package**: `langchain-openai` (uses OpenAI-compatible mode)
- **Setup**:
  1. Install Ollama from [ollama.ai](https://ollama.ai/)
  2. Pull a model: `ollama pull qwen3:8b`
  3. Set `LLM_PROVIDER=ollama` in `.env`
  4. Optionally set `LLM_MODEL` to any model you've pulled

### 8. LM Studio (Local)

- **Provider ID**: `lmstudio`
- **Default Model**: `qwen3-8b`
- **API Key**: Not required (uses dummy key internally)
- **Base URL**: `LMSTUDIO_BASE_URL` (default: `http://localhost:1234/v1`)
- **Required Package**: `langchain-openai` (uses OpenAI-compatible mode)
- **Setup**:
  1. Download LM Studio from [lmstudio.ai](https://lmstudio.ai/)
  2. Load a model in the LM Studio UI
  3. Start the local server (LM Studio → Local Server tab)
  4. Set `LLM_PROVIDER=lmstudio` in `.env`
  5. Set `LLM_MODEL` to match the loaded model name

### Default Models by Provider (Summary)

| Provider | Default Model | Notes |
|----------|---------------|-------|
| `alibaba` | `qwen-plus` | Best Chinese language support, cost-effective |
| `openai` | `gpt-5-4` | High-performance general model |
| `google` | `gemini-3-1-pro` | Advanced reasoning capabilities |
| `anthropic` | `claude-4-6-sonnet` | Strong analytical capabilities |
| `openrouter` | `openai/gpt-5-4` | Access to multiple models via single API |
| `together` | `meta-llama/Llama-3.1-8B-Instruct-Turbo` | Open-source model, competitive pricing |
| `ollama` | `qwen3:8b` | Local inference, good CJK support |
| `lmstudio` | `qwen3-8b` | Local inference via GUI |

To override the default model for any provider:

```bash
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o    # Override default
```

## Query Classification and Routing

TCM-Sage includes an intelligent query classification system that adjusts response generation based on clinical severity.

### How It Works

1. **Query Classification**: A lightweight classifier model analyzes each user query to determine if it is:
   - **Informational**: General knowledge questions, definitions, or explanations (e.g., "陰陽是什麼？")
   - **Prescriptive**: Questions asking for diagnoses, treatments, formulas, or medical advice (e.g., "頭痛應該用什麼方劑？")

2. **Dynamic Temperature Adjustment**: Based on the classification:
   - **Informational queries**: Use `LLM_TEMPERATURE` (default 0.1)
   - **Prescriptive queries**: Use `PRESCRIPTIVE_TEMPERATURE` (default 0.0 for maximum accuracy)

3. **Self-Critique Verification**: After generation, a verifier LLM checks whether the answer is faithful to the retrieved context, returning `SUPPORTED` or `UNSUPPORTED`.

### Best Practices

- **For informational queries**: You can increase `LLM_TEMPERATURE` to 0.3–0.7 for more creative explanations
- **For prescriptive queries**: Always keep `PRESCRIPTIVE_TEMPERATURE` at 0.0 to ensure medical accuracy
- **Classifier model**: Use a small, fast model (e.g., `qwen-flash`, `qwen3-0.6b`) to minimize latency and cost
- **Verifier model**: Use a fast model for quick verification; accuracy matters less than speed here

## Hybrid Retrieval (Knowledge Graph)

TCM-Sage combines vector search with the SymMap 2.0 Knowledge Graph for improved TCM terminology resolution. The system retrieves from 17 classical TCM texts (3.72M characters, 12,204+ chunks) with clause-level chunking for 伤寒论 and 金匮要略.

### Configuration

```bash
HYBRID_RETRIEVAL_ENABLED=true
GRAPH_DATA_PATH=data/graph/symmap/symmap_entities.json
GRAPH_DEPTH=2
GRAPH_MAX_RESULTS=20
```

### How It Works

1. **Vector Search**: Retrieves semantically similar text passages from the 17-text classical TCM corpus using DashScope text-embedding-v4 (1024 dimensions) with TCM domain-specific prefixes
2. **Reranking**: Retrieved passages are re-scored using DashScope qwen3-rerank for improved relevance ordering
3. **Source Authority Boost**: Canonical texts (黄帝内经, 伤寒论, 金匮要略, etc.) receive a gentle distance-score boost to rank higher when semantic scores are close
4. **Graph Search**: Traverses the SymMap 2.0 knowledge graph (18,450 entities, 21,476 relationships) to find related TCM entities (symptoms, herbs, formulas) via NetworkX
5. **Crosswalk Bridge**: Maps RAG query terms to SymMap node IDs using the approved crosswalk CSV (`data/graph/crosswalk/seed_crosswalk_approved.csv`) with jieba-enhanced entity matching
6. **Ensemble Context**: Vector results and graph facts are combined as distinct sections in the LLM prompt

### Knowledge Graph Schema

- **Entities**: `Symptom`, `Herb`, `Formula`, and more from SymMap 2.0
- **Relationships**: `TREATS`, `CONTAINS`, `ASSOCIATED_WITH`, and other clinical relationships
- **Provenance**: Each relationship includes optional `source_ref` for traceability

Example: Query "頭痛" returns:

- Vector passages about headaches from classical texts
- Graph facts: "川芎 --TREATS--> 頭痛", "天麻 --TREATS--> 頭痛", "川芎茶調散 --TREATS--> 頭痛"

### Extending the Graph

Edit `data/graph/symmap/symmap_entities.json` to add new entities and relationships. See `data/graph/symmap/README.md` for the JSON schema. Use `scripts/import_symmap_kg.py` for bulk imports from SymMap raw data.

## Embedding & Reranker Details

TCM-Sage uses **DashScope text-embedding-v4** with domain-specific prefixes for optimal TCM retrieval:

- **Ingestion prefix**: `为这段中医古籍文本生成语义表示用于检索：` (prepended to each document chunk)
- **Query prefix**: `为这个中医临床问题生成语义表示以检索相关古籍段落：` (prepended to each search query)
- **Dimensions**: 1024
- **Batch limit**: 10 texts per API call (DashScope limit)
- **Reranker**: `qwen3-rerank` supports up to 500 docs per request

The embedding model and reranker both require `DASHSCOPE_API_KEY`, even when using a different LLM provider.

## Arena Configuration

The Arena is a blind A/B evaluation system where TCM practitioners compare RAG-enhanced responses against plain LLM responses (with web search grounding) without knowing which is which.

### Model Tiers

The `ARENA_MODELS` environment variable maps tier names to model identifiers:

```bash
ARENA_MODELS='{"flash":"qwen-flash","plus":"qwen-plus","max":"qwen-max"}'
```

Default tiers:

| Tier | Default Model | Use Case |
|------|---------------|----------|
| `flash` | `qwen-flash` | Fast, lightweight comparisons |
| `plus` | `qwen-plus` | Balanced quality/speed |
| `max` | `qwen-max` | Maximum quality evaluation |

### Arena Behavior

- **RAG side**: Uses the full TCM-Sage pipeline (retrieval + reranking + system prompt + verification)
- **Plain side**: Uses a generic assistant prompt with DuckDuckGo web search results for grounding via the `ddgs` Python package
- **Position randomization**: Which side is A vs B is randomized per query
- **Vote storage**: Votes are stored as JSONL at `data/feedback/arena_votes.jsonl`
- **Statistics**: T-Test analysis available at `/arena/stats` with downloadable charts

### Stream Timeout

```bash
ARENA_STREAM_TIMEOUT_SECONDS=60  # Per-panel timeout for SSE streaming
```

## Web Frontend / API Integration

The Next.js frontend communicates with the FastAPI backend through an API proxy route at `web/app/api/backend/[...path]/route.ts`.

### Backend Configuration (root `.env`)

```bash
ALLOWED_ORIGINS=http://localhost:3000,https://your-production-domain.com
PORT=8000
FEEDBACK_FORM_URL=https://forms.gle/your-feedback-form
```

- `ALLOWED_ORIGINS`: Comma-separated CORS whitelist. Defaults to `*` (allow all) if not set. Set explicitly in production.
- `PORT`: The port FastAPI listens on (default `8000`).
- `FEEDBACK_FORM_URL`: Optional Google Form link shown in the frontend UI.

### Frontend Configuration (`web/.env.local`)

```bash
BACKEND_URL=http://127.0.0.1:8000
NEXT_PUBLIC_BACKEND_URL=http://127.0.0.1:8000
```

- `BACKEND_URL`: Used server-side by the Next.js proxy route to forward requests to FastAPI.
- `NEXT_PUBLIC_BACKEND_URL`: Client-side fallback backend URL. The proxy route checks `BACKEND_URL` first, then falls back to `NEXT_PUBLIC_BACKEND_URL`, then defaults to `http://127.0.0.1:8000`.

Copy `web/.env.local.example` to `web/.env.local` for local development.

## Retrieval Tuning Guide

### RETRIEVAL_K Parameter

The `RETRIEVAL_K` parameter controls how many document chunks are retrieved for each query:

| Range | Performance | Best For |
|-------|-------------|----------|
| 3–5 | Fast responses | Simple questions, definitions |
| 5–8 | Balanced (default: 5) | Most use cases |
| 8–15 | Slower, more comprehensive | Complex multi-concept queries |
| 15–20 | Slowest, maximum context | Research-grade deep analysis |

**Impact**: Higher values increase response time, API costs, and LLM context usage, but provide more comprehensive answers.

### GRAPH_DEPTH Parameter

Controls how many relationship hops the graph search traverses:

| Depth | Behavior |
|-------|----------|
| 1 | Direct relationships only (recommended for most queries) |
| 2 | Includes entities one hop away from direct matches |
| 3 | Maximum depth (capped in API; may return noisy results) |

### GRAPH_MAX_RESULTS Parameter

Limits the number of graph entity results per query (default: 20). Increase for comprehensive entity exploration; decrease for faster responses.

## Temperature Configuration

The temperature parameter controls the randomness of model responses:

| Value | Behavior | Recommended For |
|-------|----------|-----------------|
| 0.0 | Most deterministic, factual | Prescriptive/clinical queries |
| 0.1 | Slightly creative but mostly factual | Default informational queries |
| 0.3–0.7 | Balanced creativity and accuracy | Educational explanations |
| 1.0 | Most creative/varied responses | Creative writing, brainstorming |

The dual temperature strategy ensures medical accuracy for prescriptive queries while allowing appropriate creativity for educational content.

## System Prompt Configuration

The built-in system prompt is a comprehensive Chinese TCM clinical reference prompt (defined in `src/main.py` as `DEFAULT_SYSTEM_PROMPT`). It includes:

- TCM diagnostic framework (辨证论治)
- Cite-then-explain methodology
- Historical dosage conversion tables (东汉/隋唐/明清 weight systems)
- Source-tracing guidance for multi-text results
- Drug comparison table formatting
- Prescription analysis with 君臣佐使 framework

### Customizing the System Prompt

Use `SYSTEM_PROMPT_OVERRIDE` in `.env` to replace the default prompt without modifying code:

```bash
SYSTEM_PROMPT_OVERRIDE=你的自定义提示词...
```

> **Note:** Legacy `SYSTEM_PROMPT` env var is also read but `SYSTEM_PROMPT_OVERRIDE` takes priority. Any legacy "Sources:" directives in custom prompts are automatically stripped to prevent UI rendering conflicts.

## Required vs Optional Settings

### Settings That Cause Startup Failure If Missing

| Setting | Error Message | Fix |
|---------|--------------|-----|
| `DASHSCOPE_API_KEY` (when `LLM_PROVIDER=alibaba`) | "Alibaba API key (DASHSCOPE_API_KEY) not found" | Set in `.env` |
| `DASHSCOPE_API_KEY` (for embeddings) | "DASHSCOPE_API_KEY not set in environment" | Required for all providers (embeddings) |
| `OPENAI_API_KEY` (when `LLM_PROVIDER=openai`) | "OpenAI API key not found" | Set in `.env` |
| `GOOGLE_API_KEY` (when `LLM_PROVIDER=google`) | "Google API key not found" | Set in `.env` |
| `ANTHROPIC_API_KEY` (when `LLM_PROVIDER=anthropic`) | "Anthropic API key not found" | Set in `.env` |
| `OPENROUTER_API_KEY` (when `LLM_PROVIDER=openrouter`) | "OpenRouter API key not found" | Set in `.env` |
| `TOGETHER_API_KEY` (when `LLM_PROVIDER=together`) | "Together API key not found" | Set in `.env` |
| Vector store directory (`vectorstore/chroma/`) | "Vector store not found" | Run `python src/ingest.py` |

### Settings with Defaults (Optional)

All other settings have sensible defaults and are optional. The minimum viable `.env` is:

```bash
LLM_PROVIDER=alibaba
DASHSCOPE_API_KEY=your-key-here
```

## Troubleshooting

### Common Issues

1. **"Configuration Error: API key not found"**
   - Ensure your `.env` file is in the **project root** (not `src/` or `web/`)
   - Verify the API key variable name matches your provider
   - Check that the API key value is not the placeholder (e.g., `your-alibaba-api-key-here`)
   - Confirm the API key has sufficient credits/quota

2. **"Unsupported provider"**
   - Verify `LLM_PROVIDER` is one of: `alibaba`, `openai`, `google`, `anthropic`, `openrouter`, `together`, `ollama`, `lmstudio`
   - Check for typos or capitalization issues (value is case-insensitive)

3. **"Vector store not found"**
   - Run `venv\Scripts\python.exe src/ingest.py` to build the vector index
   - Ensure the `vectorstore/chroma/` directory was created successfully

4. **Import errors**
   - Run `venv\Scripts\python.exe -m pip install -r requirements.txt`
   - Provider-specific packages: `langchain-openai` (OpenAI/Alibaba/Ollama/LMStudio), `langchain-google-genai` (Google), `langchain-anthropic` (Anthropic), `langchain-community` (OpenRouter/Together)

5. **API connection errors**
   - Verify your API key is valid and has credits
   - Check internet connectivity
   - For DashScope: the international endpoint (`dashscope-intl.aliyuncs.com`) is used by default

6. **Reranker fails silently**
   - The reranker requires `DASHSCOPE_API_KEY` even when using a non-Alibaba LLM provider
   - If reranking fails, the system gracefully falls back to the original retrieval order

7. **CORS errors in browser**
   - Set `ALLOWED_ORIGINS` to include your frontend URL (e.g., `http://localhost:3000`)
   - Default `*` allows all origins (not recommended for production)

8. **Arena streams timing out**
   - Increase `ARENA_STREAM_TIMEOUT_SECONDS` (default 60s)
   - Use a faster model tier (e.g., `flash` instead of `max`)

### Provider-Specific Notes

- **Alibaba Cloud**: Uses the international Singapore-region endpoint (`dashscope-intl.aliyuncs.com`). Override with `DASHSCOPE_EMBEDDING_API_URL` if needed.
- **OpenAI**: Requires a paid account with sufficient credits.
- **Google AI Studio**: Free tier available with usage limits.
- **Anthropic**: Requires an API key from the Anthropic Console.
- **OpenRouter**: Pay-per-use pricing; supports many third-party models.
- **Together AI**: Competitive pricing for open-source models.
- **Ollama**: Ensure Ollama is running (`ollama serve`). Verify with `curl http://localhost:11434/v1/models`.
- **LM Studio**: Ensure the local server is started and a model is loaded. Check the LM Studio Server tab for the port.

## Cost Optimization

For cost-effective development and testing:

1. **Use Ollama or LM Studio** for free local inference (GPU recommended for acceptable speed)
2. **Start with Alibaba Cloud Model Studio** — competitive pricing for Qwen models
3. **Use smaller/faster models** for classifier and verifier (e.g., `qwen-flash` instead of `qwen-plus`)
4. **Use `qwen-flash`** for arena testing instead of `qwen-max`
5. **Lower `RETRIEVAL_K`** to reduce context size and token consumption
6. **Monitor usage** through provider dashboards — all cloud providers offer usage tracking

### Approximate Token Usage Per Query

| Component | Typical Tokens |
|-----------|---------------|
| System prompt | ~1,500 |
| Retrieved context (K=5) | ~2,000–4,000 |
| Classification | ~100 |
| Verification | ~3,000–5,000 |
| Generated answer | ~500–2,000 |

Total per query: ~7,000–12,000 tokens (varies with `RETRIEVAL_K` and answer length).

## Security Notes

- Never commit your `.env` file to version control (it is in `.gitignore`)
- Use environment variables or secret managers in production deployments
- Rotate API keys regularly
- Monitor API usage for unexpected charges
- Set `ALLOWED_ORIGINS` explicitly in production — do not use `*`
- The `.env.example` file contains only placeholder values and is safe to commit
