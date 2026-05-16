<!-- generated-by: gsd-doc-writer -->
# TCM-Sage API Reference

The TCM-Sage backend exposes a REST API built with FastAPI, running by default on `http://127.0.0.1:8000`. The Next.js frontend proxies all API calls through `/api/backend/...` to avoid CORS issues — direct calls to `http://127.0.0.1:8000` work equally well from any HTTP client.

---

## Authentication

No authentication is required. The API is designed for local or internal deployment. CORS origins are controlled by the `ALLOWED_ORIGINS` environment variable (default: `*`).

> **Note:** If you deploy TCM-Sage publicly, set `ALLOWED_ORIGINS` to your frontend domain and add your own auth layer (e.g., a reverse-proxy API key check). No auth mechanism is built in.

---

## Base URL

| Environment | URL |
|-------------|-----|
| Local development | `http://127.0.0.1:8000` |
| Via Next.js proxy | `http://localhost:3000/api/backend` |
| Custom deployment | <!-- VERIFY: production base URL --> |

---

## Endpoints Overview

| Method | Path | Description | Streaming |
|--------|------|-------------|-----------|
| `GET` | `/health` | Health check | No |
| `GET` | `/config` | Current pipeline configuration | No |
| `POST` | `/query` | RAG query with SSE streaming | **Yes** |
| `GET` | `/source/{chunk_id}/context` | Source context for a citation chunk | No |
| `GET` | `/books/{book_name}` | Full raw text of a classical book | No |
| `GET` | `/graph/subgraph` | Knowledge graph subgraph for an entity | No |
| `GET` | `/graph/search` | Search KG entities by name | No |
| `POST` | `/arena/query` | Blind A/B arena query with dual SSE streams | **Yes** |
| `POST` | `/arena/vote` | Submit an arena vote | No |
| `GET` | `/arena/models` | List available arena model presets | No |
| `GET` | `/arena/stats` | Arena evaluation statistics with T-Test | No |

---

## GET /health

Health check for deployment monitoring.

**Response `200 OK`:**
```json
{
  "status": "ok",
  "timestamp": "2026-04-11T10:00:00.000000"
}
```

**Example:**
```bash
curl http://127.0.0.1:8000/health
```

---

## GET /config

Returns the current resolved pipeline configuration including provider, model, temperature, and retrieval settings.

**Response `200 OK`:**
```json
{
  "provider": "alibaba",
  "model": "qwen-plus",
  "informational_temperature": 0.1,
  "prescriptive_temperature": 0.0,
  "classifier_follow_main": false,
  "classifier_provider": "alibaba",
  "classifier_model": "qwen3-0.6b",
  "verifier_follow_main": true,
  "verifier_provider": "alibaba",
  "verifier_model": "qwen-plus",
  "retrieval_k": 5,
  "hybrid_enabled": true,
  "hybrid_available": true,
  "graph_depth": 1,
  "graph_max_results": 20
}
```

**Error `500`:** Config failed to load — check `.env` setup.

**Example:**
```bash
curl http://127.0.0.1:8000/config
```

---

## POST /query

Execute a query against the RAG pipeline. Returns a **Server-Sent Events (SSE)** stream.

**Request body:**
```json
{
  "question": "What are the symptoms of Wind-Cold invasion?",
  "chat_history": [
    { "role": "user", "content": "Tell me about Qi" },
    { "role": "assistant", "content": "Qi is the vital energy..." }
  ],
  "settings": {
    "provider": "alibaba",
    "model": "qwen-plus",
    "informational_temperature": 0.1,
    "prescriptive_temperature": 0.0,
    "classifier_follow_main": true,
    "classifier_provider": null,
    "classifier_model": null,
    "verifier_follow_main": true,
    "verifier_provider": null,
    "verifier_model": null,
    "retrieval_k": 5,
    "hybrid_retrieval_enabled": true,
    "graph_depth": 1,
    "graph_max_results": 20
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `question` | `string` | **Yes** | The query text. Must not be empty. |
| `chat_history` | `array` | No | Previous turns. Each item: `{ "role": "user"|"assistant", "content": "..." }`. Last 6 turns are used. |
| `settings` | `object` | No | Per-request overrides. All fields optional — omit to use server defaults. |

**Response headers:**
```
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive
X-Accel-Buffering: no
```

### SSE Event Types

#### Text chunks (unnamed `data:` events)

Each LLM output chunk arrives as a plain `data:` line. Newlines within the chunk are escaped as `\n`:

```
data: Wind-Cold invasion\n\n
data:  manifests as...\n\n
```

Concatenate all text chunks to reconstruct the full answer.

#### `metadata` event (final event)

Emitted once after the full answer is generated. Contains citations, verification result, and pipeline diagnostics:

```
event: metadata
data: {"type":"metadata","question":"...","answer":"...","severity":"informational","temperature":0.1,"timestamp":"2026-04-11T10:00:00","provider":"alibaba","model":"qwen-plus","retrieval_k":5,"verification":{"status":"SUPPORTED","explanation":"The answer appears supported by the retrieved citations."},"verification_result":"SUPPORTED","citation_bounds":{"is_valid":true,"out_of_range":[],"found_citations":[1,2]},"citations":[...],"debug_context":"..."}
```

**`metadata` payload fields:**

| Field | Type | Description |
|-------|------|-------------|
| `type` | `"metadata"` | Always `"metadata"` |
| `question` | `string` | Echo of the query |
| `answer` | `string` | Full concatenated answer |
| `severity` | `"informational"` \| `"prescriptive"` | Query classification result |
| `temperature` | `number` | Temperature used for generation |
| `timestamp` | `string` | ISO 8601 UTC timestamp |
| `provider` | `string` | LLM provider used |
| `model` | `string` \| `null` | Model name used |
| `retrieval_k` | `number` | Number of chunks retrieved |
| `verification` | `object` | `{ "status": "SUPPORTED"|"UNSUPPORTED"|..., "explanation": "..." }` |
| `verification_result` | `string` | Raw verifier output |
| `citation_bounds` | `object` | Inline citation range check with `is_valid`, `out_of_range`, and `found_citations` |
| `citations` | `array` | List of `TextCitation` and `GraphCitation` objects |
| `debug_context` | `string` | Full formatted context sent to the LLM (for debugging) |

**Citation object shapes:**

`TextCitation` (type `"text"`):
```json
{
  "number": 1,
  "type": "text",
  "source": "伤寒论 · 辨太阳病脉证并治法上",
  "content": "太阳病，发热，汗出，恶风，脉缓者，名为中风。",
  "chunk_id": "shanghanlun_001_0003",
  "score": 0.312,
  "relevance_percent": 91.5
}
```

`GraphCitation` (type `"graph"`):
```json
{
  "number": 2,
  "type": "graph",
  "fact": "桂枝 --treats--> 风寒感冒 (Disease)",
  "depth": 1,
  "source_ref": null
}
```

#### `error` event

Emitted if the pipeline throws an exception:

```
event: error
data: {"type":"error","message":"LLM API call failed: rate limit exceeded"}
```

**Error `400`:** Question is empty.

**Example:**
```bash
curl -N -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What does 桂枝汤 treat?"}'
```

---

## GET /source/{chunk_id}/context

Retrieve the deduplicated source paragraph context for a specific citation chunk. Used by the frontend citation panel and `source/[chunkId]` drill-down page.

**Path parameter:** `chunk_id` — URL-encoded chunk ID from a `TextCitation` object (e.g., `shanghanlun_001_0003`).

**Response `200 OK`:**
```json
{
  "chunk_id": "shanghanlun_001_0003",
  "book": "伤寒论",
  "chapter": "辨太阳病脉证并治法上",
  "chapter_display": "辨太阳病脉证并治法上",
  "chunk_index": 3,
  "full_chapter_text": "太阳之为病，脉浮，头项强痛而恶寒。...",
  "highlight_start": 42,
  "highlight_end": 65,
  "paragraph_text": "太阳病，发热，汗出，恶风，脉缓者，名为中风。",
  "paragraph_highlight_start": 0,
  "paragraph_highlight_end": 23,
  "total_chunks_in_chapter": 18
}
```

| Field | Description |
|-------|-------------|
| `full_chapter_text` | Full reconstructed chapter with overlap-deduplication applied |
| `highlight_start` / `highlight_end` | Character offsets of the chunk within `full_chapter_text` |
| `paragraph_text` | The paragraph block containing the highlighted chunk |
| `paragraph_highlight_start` / `paragraph_highlight_end` | Character offsets within `paragraph_text` |

**Errors:**
- `404`: Chunk ID not found in vectorstore or chunks data.
- `500`: Incomplete metadata or file read failure.

**Example:**
```bash
curl "http://127.0.0.1:8000/source/shanghanlun_001_0003/context"
```

---

## GET /books/{book_name}

Retrieve the full raw text of a classical TCM book from `data/source/`. Supports fuzzy matching on the book stem name. Tries encodings: UTF-8 → UTF-8-SIG → GB18030 → GBK → BIG5.

**Path parameter:** `book_name` — book filename stem or full name (e.g., `伤寒论` or `01-伤寒论`).

**Response `200 OK`:**
```json
{
  "content": "辨太阳病脉证并治法上\n\n太阳之为病，脉浮..."
}
```

**Error `404`:** Book not found. Response includes `requested_stem`, `decoded_name`, and `sample_stems` for debugging.

**Example:**
```bash
curl "http://127.0.0.1:8000/books/%E4%BC%A4%E5%AF%92%E8%AE%BA"
```

---

## GET /graph/subgraph

Retrieve a knowledge graph subgraph centered on a named entity. Used by the `/kg/[entityId]` explorer page. Returns empty arrays if the knowledge graph is unavailable.

**Query parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `entity` | `string` | **Required** | Entity name to search (e.g., `桂枝`) |
| `hops` | `integer` | `2` | Graph traversal depth (capped at 3) |

**Response `200 OK`:**
```json
{
  "nodes": [
    { "id": "herb_guizhi_001", "label": "桂枝", "type": "Herb" },
    { "id": "disease_fenghan_001", "label": "风寒感冒", "type": "Disease" }
  ],
  "edges": [
    { "source": "herb_guizhi_001", "target": "disease_fenghan_001", "label": "treats" }
  ],
  "cited_ids": ["herb_guizhi_001"]
}
```

**Example:**
```bash
curl "http://127.0.0.1:8000/graph/subgraph?entity=%E6%A1%82%E6%9E%9D&hops=2"
```

---

## GET /graph/search

Search knowledge graph entities by name. Supports partial matching for autocomplete and the KG explorer search bar.

**Query parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `q` | `string` | **Required** | Search term |
| `limit` | `integer` | `20` | Maximum number of results |

**Response `200 OK`:**
```json
{
  "results": [
    { "id": "herb_guizhi_001", "label": "桂枝", "type": "Herb" },
    { "id": "herb_guizhi_002", "label": "桂枝茯苓丸", "type": "Formula" }
  ]
}
```

Returns `{ "results": [] }` if the knowledge graph is unavailable.

**Example:**
```bash
curl "http://127.0.0.1:8000/graph/search?q=%E6%A1%82%E6%9E%9D&limit=10"
```

---

## POST /arena/query

Execute a blind A/B arena comparison. One side receives the full RAG pipeline response; the other receives a plain LLM response augmented with web search results. Which side is RAG and which is plain is randomly assigned per request. Both streams are multiplexed over a single **SSE** connection.

**Request body:**
```json
{
  "question": "What are the indications for 桂枝汤?",
  "chat_history_a": [],
  "chat_history_b": [],
  "model_name": "qwen-flash",
  "session_id": "arena-session-abc123",
  "round_number": 1
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `question` | `string` | — | **Required.** Must not be empty. |
| `chat_history_a` | `array` | `[]` | Chat history for panel A |
| `chat_history_b` | `array` | `[]` | Chat history for panel B |
| `model_name` | `string` | `"qwen-flash"` | LLM model name for both sides |
| `session_id` | `string` | `""` | Client-assigned session identifier |
| `round_number` | `integer` | `1` | Round counter within a session |

**Timeout:** Controlled by `ARENA_STREAM_TIMEOUT_SECONDS` env variable (default: `60` seconds per panel).

### Arena SSE Event Types

All events are multiplexed on the same stream. Panels are labeled `_a` and `_b`.

| Event name | Description |
|------------|-------------|
| `text_a` | Text chunk for panel A |
| `text_b` | Text chunk for panel B |
| `metadata_a` | Final metadata for panel A (citations, verification) |
| `metadata_b` | Final metadata for panel B |
| `arena_config` | **Final event.** Reveals `position_mapping`, `session_id`, `round_number` |
| `error` | Error for a specific panel. Payload: `{ "panel": "a"|"b", "message": "..." }` |

**`arena_config` payload (always the last event):**
```
event: arena_config
data: {"position_mapping": {"a": "rag", "b": "plain"}, "session_id": "arena-session-abc123", "round_number": 1}
```

`position_mapping` maps panel labels (`"a"`, `"b"`) to roles (`"rag"` or `"plain"`). This is revealed only at the end to preserve blind evaluation integrity.

**Example:**
```bash
curl -N -X POST http://127.0.0.1:8000/arena/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Explain Yin-Yang theory",
    "model_name": "qwen-flash",
    "session_id": "my-session-001",
    "round_number": 1
  }'
```

---

## POST /arena/vote

Submit a user vote for a completed arena round.

**Request body:**
```json
{
  "session_id": "arena-session-abc123",
  "round_number": 1,
  "query": "What are the indications for 桂枝汤?",
  "response_a": "Panel A full response text...",
  "response_b": "Panel B full response text...",
  "model_name": "qwen-flash",
  "position_mapping": { "a": "rag", "b": "plain" },
  "vote": "a",
  "comment": "Panel A gave more specific classical citations.",
  "timestamp": "2026-04-11T10:00:00.000000"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `session_id` | `string` | **Yes** | Matches the arena session |
| `round_number` | `integer` | **Yes** | Round within the session |
| `query` | `string` | **Yes** | The original question text |
| `response_a` | `string` | **Yes** | Full text of panel A response |
| `response_b` | `string` | **Yes** | Full text of panel B response |
| `model_name` | `string` | **Yes** | Model used for the round |
| `position_mapping` | `object` | **Yes** | From `arena_config` event: `{"a": "rag"|"plain", "b": ...}` |
| `vote` | `"a"` \| `"b"` \| `"tie"` | **Yes** | User's choice |
| `comment` | `string` \| `null` | No | Optional free-text comment |
| `timestamp` | `string` | No | ISO 8601 timestamp. Defaults to server UTC now if empty. |

Votes are appended to `data/feedback/arena_votes.jsonl`.

**Response `200 OK`:**
```json
{ "status": "ok" }
```

**Example:**
```bash
curl -X POST http://127.0.0.1:8000/arena/vote \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "my-session-001",
    "round_number": 1,
    "query": "Explain Yin-Yang",
    "response_a": "...",
    "response_b": "...",
    "model_name": "qwen-flash",
    "position_mapping": {"a": "rag", "b": "plain"},
    "vote": "a"
  }'
```

---

## GET /arena/models

Returns the available model presets for the arena.

**Response `200 OK`:**
```json
{
  "flash": "qwen-flash",
  "plus": "qwen-plus",
  "max": "qwen-max"
}
```

Default presets can be extended by setting `ARENA_MODELS` as a JSON string in `.env`:
```bash
ARENA_MODELS='{"flash":"qwen-flash","plus":"qwen-plus","max":"qwen-max"}'
```

**Example:**
```bash
curl http://127.0.0.1:8000/arena/models
```

---

## GET /arena/stats

Compute evaluation statistics from all stored arena votes. Includes win rates, a paired T-Test, Cohen's d effect size, and per-query breakdown.

**Response `200 OK` (with votes):**
```json
{
  "total_votes": 12,
  "rag_wins": 8,
  "plain_wins": 3,
  "ties": 1,
  "rag_win_rate": 66.7,
  "plain_win_rate": 25.0,
  "tie_rate": 8.3,
  "t_test": {
    "t_statistic": 2.4495,
    "p_value": 0.031500,
    "cohens_d": 0.7071,
    "mean_rag_score": 0.7083,
    "sample_size": 12,
    "significant": true,
    "interpretation": "Statistically significant preference for RAG"
  },
  "query_results": [
    {
      "query": "What does 桂枝汤 treat?",
      "winner": "rag",
      "model": "qwen-plus",
      "timestamp": "2026-04-11T10:00:00",
      "session_id": "arena-session-abc123"
    }
  ]
}
```

**Response `200 OK` (no votes yet):**
```json
{ "total_votes": 0, "votes": [], "statistics": null }
```

T-Test requires at least 3 votes; `t_test` is `null` below that threshold.

**Example:**
```bash
curl http://127.0.0.1:8000/arena/stats
```

---

## Error Codes

| HTTP Status | Meaning | Common Causes |
|-------------|---------|---------------|
| `400` | Bad Request | Empty `question` field in `/query` or `/arena/query` |
| `404` | Not Found | Chunk ID not in vectorstore; book name not matched in `data/source/` |
| `500` | Internal Server Error | Config load failure; file read error; incomplete chunk metadata |

**Standard error response shape:**
```json
{
  "detail": "Human-readable error message or structured object"
}
```

For `/books/{book_name}` 404 errors, `detail` is a structured object with debugging fields:
```json
{
  "detail": {
    "message": "Book '伤寒论' not found in source repository",
    "requested_stem": "伤寒论",
    "decoded_name": "伤寒论",
    "normalized_requested": "伤寒论",
    "sample_stems": ["01-黄帝内经", "02-伤寒论", "03-金匮要略方论"]
  }
}
```

---

## Rate Limits

No rate limiting is configured in the application. <!-- VERIFY: confirm no rate limiting is applied at the reverse-proxy or hosting layer -->

---

## Next.js Frontend Proxy

All frontend API calls are routed through the Next.js catch-all proxy at `web/app/api/backend/[...path]/route.ts`. This proxy:

- Forwards any HTTP method (`GET`, `POST`) to the FastAPI backend
- Backend URL is resolved from `BACKEND_URL` → `NEXT_PUBLIC_BACKEND_URL` → `http://127.0.0.1:8000`
- Strips `host`, `connection`, `content-length`, `accept-encoding` request headers
- Strips `content-encoding`, `content-length` response headers (required for SSE passthrough)

**Frontend call pattern:**
```
GET  /api/backend/health         → GET  http://127.0.0.1:8000/health
POST /api/backend/query          → POST http://127.0.0.1:8000/query
POST /api/backend/arena/query    → POST http://127.0.0.1:8000/arena/query
```

To point the frontend at a remote backend, set `BACKEND_URL` in `web/.env.local`:
```bash
BACKEND_URL=https://your-backend-host.example.com
```
