"""
FastAPI backend for TCM-Sage.

This module exposes the RAG pipeline as a REST API
with Server-Sent Events (SSE) streaming support.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

load_dotenv()

from arena import ARENA_MODELS, ArenaVoteRecord, store_vote
from arena_stats import compute_arena_stats
from arena_stream import generate_arena_sse_stream
from source_context import get_book_text as resolve_book_text
from source_context import get_chunk_context as resolve_chunk_context
from ui_backend import PipelineConfig, get_runtime_config, run_query_stream

app = FastAPI(
    title="TCM-Sage API",
    description="Evidence-synthesis API for Traditional Chinese Medicine",
    version="1.0.0",
)

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RuntimeSettingsRequest(BaseModel):
    """Per-request runtime overrides for generation and retrieval."""

    provider: str | None = None
    model: str | None = None
    informational_temperature: float | None = None
    prescriptive_temperature: float | None = None
    classifier_follow_main: bool | None = None
    classifier_provider: str | None = None
    classifier_model: str | None = None
    verifier_follow_main: bool | None = None
    verifier_provider: str | None = None
    verifier_model: str | None = None
    retrieval_k: int | None = None
    hybrid_retrieval_enabled: bool | None = None
    graph_depth: int | None = None
    graph_max_results: int | None = None


class QueryRequest(BaseModel):
    """Request body for the /query endpoint."""

    question: str
    chat_history: list[dict] = Field(default_factory=list)
    settings: RuntimeSettingsRequest | None = None


class ConfigResponse(BaseModel):
    """Response body for the /config endpoint."""

    provider: str
    model: str | None
    informational_temperature: float
    prescriptive_temperature: float
    classifier_follow_main: bool
    classifier_provider: str
    classifier_model: str | None
    verifier_follow_main: bool
    verifier_provider: str
    verifier_model: str | None
    retrieval_k: int
    hybrid_enabled: bool
    hybrid_available: bool
    graph_depth: int
    graph_max_results: int


class ArenaQueryRequest(BaseModel):
    question: str
    chat_history_a: list[dict] = Field(default_factory=list)
    chat_history_b: list[dict] = Field(default_factory=list)
    model_name: str = "qwen-flash"
    session_id: str = ""
    round_number: int = 1


class ArenaVoteRequest(BaseModel):
    session_id: str
    round_number: int
    query: str
    response_a: str
    response_b: str
    model_name: str
    position_mapping: dict
    vote: Literal["a", "b", "tie"]
    comment: str | None = None
    timestamp: str = ""


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint for deployment monitoring."""

    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.get("/config", response_model=ConfigResponse)
async def get_config() -> ConfigResponse:
    """Get the current pipeline configuration."""

    try:
        config: PipelineConfig = get_runtime_config()
        return ConfigResponse(
            provider=config.provider,
            model=config.model,
            informational_temperature=config.informational_temperature,
            prescriptive_temperature=config.prescriptive_temperature,
            classifier_follow_main=config.classifier_follow_main,
            classifier_provider=config.classifier_provider,
            classifier_model=config.classifier_model,
            verifier_follow_main=config.verifier_follow_main,
            verifier_provider=config.verifier_provider,
            verifier_model=config.verifier_model,
            retrieval_k=config.retrieval_k,
            hybrid_enabled=config.hybrid_enabled,
            hybrid_available=config.hybrid_available,
            graph_depth=config.graph_depth,
            graph_max_results=config.graph_max_results,
        )
    except Exception as exc:  # pragma: no cover - passthrough for runtime failures
        raise HTTPException(status_code=500, detail=f"Failed to load config: {exc}") from exc


def generate_sse_stream(
    question: str,
    chat_history: list[dict] | None = None,
    settings: dict[str, Any] | None = None,
) -> Generator[str, None, None]:
    """Generate Server-Sent Events from the RAG pipeline stream."""

    if chat_history is None:
        chat_history = []

    try:
        for item in run_query_stream(question, chat_history, settings):
            if isinstance(item, dict) and item.get("type") == "metadata":
                import json

                yield f"event: metadata\ndata: {json.dumps(item)}\n\n"
            else:
                chunk = str(item).replace("\n", "\\n")
                yield f"data: {chunk}\n\n"
    except Exception as exc:  # pragma: no cover - passthrough for runtime failures
        import json

        error_data = {"type": "error", "message": str(exc)}
        yield f"event: error\ndata: {json.dumps(error_data)}\n\n"


@app.post("/query")
async def query(request: QueryRequest) -> StreamingResponse:
    """Execute a query against the RAG pipeline with a streaming response."""

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    return StreamingResponse(
        generate_sse_stream(
            request.question,
            request.chat_history,
            request.settings.model_dump(exclude_none=True) if request.settings else None,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/source/{chunk_id}/context")
async def get_chunk_context(chunk_id: str) -> Dict[str, Any]:
    """Get deduplicated source context for a specific chunk."""
    return await resolve_chunk_context(chunk_id)


@app.get("/books/{book_name}")
async def get_book_text(book_name: str) -> Dict[str, str]:
    """Retrieve the full raw text of a book from the source directory."""
    return await resolve_book_text(book_name)


@app.get("/graph/subgraph")
async def get_graph_subgraph(entity: str, hops: int = 2) -> Dict[str, Any]:
    from ui_backend import _get_knowledge_graph

    try:
        config = get_runtime_config()
        if not config.hybrid_available:
            return {"nodes": [], "edges": [], "cited_ids": []}
        kg = _get_knowledge_graph(config.graph_data_path)
    except Exception:
        return {"nodes": [], "edges": [], "cited_ids": []}

    entity_ids = kg.search_by_name(entity)
    if not entity_ids:
        return {"nodes": [], "edges": [], "cited_ids": []}

    seed_id = entity_ids[0]
    related = kg.get_related_entities(seed_id, max_depth=min(hops, 3), max_results=100)

    nodes = []
    seed_attrs = kg.graph.nodes.get(seed_id, {})
    nodes.append({
        "id": seed_id,
        "label": seed_attrs.get("name", seed_id),
        "type": seed_attrs.get("type", "Unknown"),
    })

    for related_item in related:
        entity_data = related_item["entity"]
        nodes.append({
            "id": entity_data["id"],
            "label": entity_data.get("name", entity_data["id"]),
            "type": entity_data.get("type", "Unknown"),
        })

    edges = []
    node_ids = {node["id"] for node in nodes}
    for related_item in related:
        relationship = related_item["relationship"]
        if relationship["source"] in node_ids and relationship["target"] in node_ids:
            edges.append({
                "source": relationship["source"],
                "target": relationship["target"],
                "label": relationship.get("type", ""),
            })

    return {
        "nodes": nodes,
        "edges": edges,
        "cited_ids": [seed_id],
    }

@app.get("/graph/search")
async def search_graph_entities(q: str, limit: int = 20) -> Dict[str, Any]:
    """Search KG entities by name for autocomplete / explorer search bar."""
    from ui_backend import _get_knowledge_graph

    try:
        config = get_runtime_config()
        if not config.hybrid_available:
            return {"results": []}
        kg = _get_knowledge_graph(config.graph_data_path)
    except Exception:
        return {"results": []}

    entity_ids = kg.search_by_name(q)
    results = []
    for eid in entity_ids[:limit]:
        attrs = kg.graph.nodes.get(eid, {})
        results.append({
            "id": eid,
            "label": attrs.get("name", eid),
            "type": attrs.get("type", "Unknown"),
        })

    return {"results": results}


@app.post("/arena/query")
async def arena_query(request: ArenaQueryRequest) -> StreamingResponse:
    """Execute a blind A/B arena query with dual multiplexed SSE streaming."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    return StreamingResponse(
        generate_arena_sse_stream(
            question=request.question,
            chat_history_a=request.chat_history_a,
            chat_history_b=request.chat_history_b,
            model_name=request.model_name,
            session_id=request.session_id,
            round_number=request.round_number,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/arena/vote")
async def arena_vote(request: ArenaVoteRequest) -> dict:
    """Store an arena vote."""
    record: ArenaVoteRecord = {
        "session_id": request.session_id,
        "round_number": request.round_number,
        "query": request.query,
        "response_a": request.response_a,
        "response_b": request.response_b,
        "model_name": request.model_name,
        "position_mapping": request.position_mapping,
        "vote": request.vote,
        "comment": request.comment,
        "timestamp": request.timestamp or datetime.utcnow().isoformat(),
    }
    store_vote(record)
    return {"status": "ok"}


@app.get("/arena/models")
async def arena_models() -> dict:
    """Return available arena model presets."""
    return ARENA_MODELS


@app.get("/arena/stats")
async def get_arena_stats() -> dict:
    """Compute arena evaluation statistics with T-Test."""
    return compute_arena_stats()

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        reload=False,
    )
