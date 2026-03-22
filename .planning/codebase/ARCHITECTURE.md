# Architecture

**Analysis Date:** 2026-03-23

## Pattern Overview

**Overall:** Retrieval-Augmented Generation (RAG) with Hybrid Retrieval (Vector + Knowledge Graph).

**Key Characteristics:**
- **Hybrid Retrieval**: Combines semantic vector search (`ChromaDB`) with structured knowledge graph traversal (`NetworkX`).
- **3-Pass KG Extraction**: Employs a multi-pass LLM extraction process (Entity -> Relation -> Critique) to build a high-fidelity knowledge graph.
- **Evidence-Centric Pipeline**: Prioritizes provenance and citations, ensuring all LLM claims are backed by source text or graph facts.

## Layers

**API Layer:**
- Purpose: Exposes the RAG pipeline as a REST API with SSE support for real-time streaming.
- Location: `src/api.py`
- Contains: FastAPI application, Pydantic models for requests/responses, and streaming generators.
- Depends on: `src/ui_backend.py`
- Used by: `web/lib/api.ts` (Frontend)

**Orchestration Layer:**
- Purpose: Manages the RAG pipeline lifecycle, from query classification to answer verification.
- Location: `src/ui_backend.py`, `src/main.py`
- Contains: LangChain runnables, prompt templates, model configurations, and streaming logic.
- Depends on: `src/retriever.py`, `src/graph_builder.py`
- Used by: `src/api.py`

**Retrieval Layer:**
- Purpose: Retrieves relevant context from multiple data sources.
- Location: `src/retriever.py`, `src/graph_builder.py`
- Contains: `HybridRetriever` (Vector search via ChromaDB, Graph search via NetworkX).
- Depends on: `vectorstore/chroma/`, `data/graph/`
- Used by: `src/ui_backend.py`

**Data Processing Layer:**
- Purpose: Prepares raw TCM texts for retrieval (ingestion and KG extraction).
- Location: `src/ingest.py`, `src/kg_extractor.py`
- Contains: `SentenceAwareChineseTextSplitter`, LLM-based entity/relation extractors.
- Depends on: `data/source/`
- Used by: Offline processing scripts, `scripts/`

## Data Flow

**Standard RAG Query:**

1. **User Query**: Received via `/query` endpoint in `src/api.py`.
2. **Classification**: Query is classified as "informational" or "prescriptive" in `src/ui_backend.py` to set LLM temperature.
3. **Hybrid Retrieval**: `HybridRetriever` fetches semantic chunks from `ChromaDB` and related facts from the Knowledge Graph.
4. **Context Formatting**: Retrieved docs are formatted into a context string with numeric citations in `src/main.py`.
5. **Generation**: LLM (Qwen/OpenAI) generates an answer using the context and chat history, streamed via SSE.
6. **Verification**: Post-hoc verification check (`src/verifier.py`) ensures the answer is supported by the context.
7. **Metadata Injection**: Citations and verification results are appended to the stream as a final JSON metadata event.

**State Management:**
- **Backend**: Stateless API; chat history is passed in each request from the frontend.
- **Frontend**: Managed in React components and hooks (`web/hooks/useChat.ts`, `web/hooks/useHistory.ts`).

## Key Abstractions

**HybridRetriever:**
- Purpose: Orchestrates multi-source retrieval (semantic vector search and graph traversal).
- Examples: `src/retriever.py`
- Pattern: Strategy/Ensemble pattern for combining search results.

**TCMKnowledgeGraph:**
- Purpose: Represents TCM entities and their relationships as a directed graph.
- Examples: `src/graph_builder.py`
- Pattern: Graph data structure built on `networkx`.

**SentenceAwareChineseTextSplitter:**
- Purpose: Splits Chinese text at sentence boundaries (。；！？) to maintain semantic coherence.
- Examples: `src/ingest.py`
- Pattern: Custom text splitting strategy for LangChain.

## Entry Points

**FastAPI Server:**
- Location: `src/api.py`
- Triggers: HTTP requests from frontend or external clients.
- Responsibilities: Routing, SSE streaming, error handling, CORS configuration.

**Main CLI App:**
- Location: `src/main.py`
- Triggers: Direct execution for testing or batch processing.
- Responsibilities: Core pipeline initialization, LLM provider selection, command-line interface.

**Next.js Frontend:**
- Location: `web/app/page.tsx`
- Triggers: User interaction in the browser.
- Responsibilities: Chat UI, KG visualization, settings management, streaming consumption.

## Error Handling

**Strategy:** Fail-fast on configuration/initialization errors; soft-fail with informative messages during streaming queries.

**Patterns:**
- **SSE Error Events**: Backend yields `event: error` with JSON detail if pipeline fails mid-stream.
- **HTTP Exceptions**: FastAPI `HTTPException` for standard validation or retrieval failures (e.g., `404 Not Found` for source context).
- **Graceful Fallbacks**: Hybrid retrieval falls back to vector-only if graph data is missing.

## Cross-Cutting Concerns

**Logging:** Standard Python console logging and print statements for debugging.
**Validation:** Pydantic models for API request/response schemas in `src/api.py`.
**Authentication:** CORS middleware in `src/api.py` for frontend-backend communication. No user-level auth implemented yet.
**Configuration:** Environment-variable based configuration via `python-dotenv` and `src/config.py`.

---

*Architecture analysis: 2026-03-23*
