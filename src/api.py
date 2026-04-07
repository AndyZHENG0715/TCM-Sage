"""
FastAPI backend for TCM-Sage.

This module exposes the RAG pipeline as a REST API
with Server-Sent Events (SSE) streaming support.
"""

from __future__ import annotations

import asyncio
import os
import random
import re
import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, AsyncGenerator, AsyncIterator, Dict, Generator, Literal, cast
from urllib.parse import unquote

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

load_dotenv()

from arena import ARENA_MODELS, ArenaVoteRecord, generate_raw_llm_response, generate_rag_response, store_vote
from ui_backend import PipelineConfig, get_runtime_config, get_shared_vectorstore, run_query_stream

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
    model_name: str = "qwen-plus"
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


DEFAULT_ARENA_STREAM_TIMEOUT_SECONDS = 60.0


def resolve_arena_stream_timeout_seconds(timeout_override: float | None = None) -> float:
    if timeout_override is not None and timeout_override > 0:
        return float(timeout_override)

    raw_timeout = os.getenv("ARENA_STREAM_TIMEOUT_SECONDS", str(DEFAULT_ARENA_STREAM_TIMEOUT_SECONDS))
    try:
        timeout_value = float(raw_timeout)
    except ValueError:
        return DEFAULT_ARENA_STREAM_TIMEOUT_SECONDS

    return timeout_value if timeout_value > 0 else DEFAULT_ARENA_STREAM_TIMEOUT_SECONDS


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


_PARAGRAPH_BREAK_RE = re.compile(r"\n\s*\n")
_LOW_QUALITY_SOURCE_RE = re.compile(
    r"^卷[一二三四五六七八九十百千万0-9]+(?:第[一二三四五六七八九十百千万0-9]+)?(?:上编|中编|下编)?$"
)


def clean_source_label(source: str | None) -> str:
    """Clean a raw source label for UI display."""

    if not source:
        return ""

    stripped = re.sub(r"<[^>]+>", "", source)
    return re.sub(r"[。．、:：;；)\]）】」』]+$", "", stripped).strip()


def is_low_quality_source_label(source: str | None) -> bool:
    """Detect labels that are only volume/index detritus."""

    cleaned = clean_source_label(source).replace(" ", "")
    return bool(cleaned) and bool(_LOW_QUALITY_SOURCE_RE.fullmatch(cleaned))


def find_overlap_length(existing_text: str, next_chunk: str) -> int:
    """Return the largest suffix/prefix overlap between adjacent chunks."""

    max_overlap = min(len(existing_text), len(next_chunk))
    for overlap in range(max_overlap, 0, -1):
        if existing_text.endswith(next_chunk[:overlap]):
            return overlap
    return 0


def build_full_source_text(chapter_chunks: list[dict]) -> tuple[str, dict[str, tuple[int, int]]]:
    """Reconstruct source text while removing chunk-overlap duplication."""

    full_text = ""
    chunk_ranges: dict[str, tuple[int, int]] = {}

    for chunk in chapter_chunks:
        chunk_id = chunk.get("id")
        chunk_content = chunk.get("content", "")
        if not chunk_id:
            continue

        if not full_text:
            chunk_start = 0
            full_text = chunk_content
        else:
            overlap = find_overlap_length(full_text, chunk_content)
            chunk_start = len(full_text) - overlap
            full_text += chunk_content[overlap:]

        chunk_ranges[chunk_id] = (chunk_start, chunk_start + len(chunk_content))

    return full_text, chunk_ranges


def extract_paragraph_context(
    full_text: str,
    highlight_start: int,
    highlight_end: int,
) -> tuple[str, int, int]:
    """Extract the paragraph block containing the highlighted span."""

    paragraph_start = 0
    paragraph_end = len(full_text)

    for match in _PARAGRAPH_BREAK_RE.finditer(full_text):
        if match.end() <= highlight_start:
            paragraph_start = match.end()
        elif match.start() >= highlight_end:
            paragraph_end = match.start()
            break

    raw_paragraph = full_text[paragraph_start:paragraph_end]
    trimmed_paragraph = raw_paragraph.strip()
    leading_trim = len(raw_paragraph) - len(raw_paragraph.lstrip())
    local_start = max(0, highlight_start - paragraph_start - leading_trim)
    local_end = max(local_start, highlight_end - paragraph_start - leading_trim)

    return trimmed_paragraph, local_start, local_end



@app.get("/source/{chunk_id}/context")
async def get_chunk_context(chunk_id: str) -> Dict[str, Any]:
    """Get deduplicated source context for a specific chunk."""

    try:
        normalized_chunk_id = unquote(chunk_id)
        vectorstore = get_shared_vectorstore()
        result = vectorstore._collection.get(ids=[normalized_chunk_id], include=["metadatas"])

        if not result or not result["ids"]:
            raise HTTPException(
                status_code=404,
                detail=f"Chunk {normalized_chunk_id} not found in VectorStore",
            )

        metadatas = result.get("metadatas")
        if not metadatas or metadatas[0] is None:
            raise HTTPException(
                status_code=500,
                detail=f"Incomplete metadata for chunk {normalized_chunk_id}",
            )

        metadata = metadatas[0]
        book_raw = metadata.get("book")
        chapter_raw = metadata.get("source")
        book = book_raw if isinstance(book_raw, str) else ""
        chapter = chapter_raw if isinstance(chapter_raw, str) else ""
        if not book or not chapter:
            raise HTTPException(
                status_code=500,
                detail=f"Incomplete metadata for chunk {normalized_chunk_id}",
            )

        chapter_chunks = [
            chunk
            for chunk in load_chunks_data()
            if chunk.get("metadata", {}).get("book") == book
            and chunk.get("metadata", {}).get("source") == chapter
        ]
        if not chapter_chunks:
            raise HTTPException(status_code=404, detail=f"Chapter chunks not found in data store for {book} - {chapter}")

        chapter_chunks.sort(
            key=lambda chunk: (
                chunk.get("metadata", {}).get("char_start", 0),
                chunk.get("metadata", {}).get("chunk_index", 0),
            )
        )

        full_text, chunk_ranges = build_full_source_text(chapter_chunks)
        if normalized_chunk_id not in chunk_ranges:
            raise HTTPException(
                status_code=404,
                detail=f"Chunk {normalized_chunk_id} not found in reconstructed source context",
            )

        highlight_start, highlight_end = chunk_ranges[normalized_chunk_id]
        paragraph_text, paragraph_highlight_start, paragraph_highlight_end = extract_paragraph_context(
            full_text,
            highlight_start,
            highlight_end,
        )

        chapter_display = clean_source_label(chapter)
        if is_low_quality_source_label(chapter_display):
            chapter_display = ""

        return {
            "chunk_id": normalized_chunk_id,
            "book": book,
            "chapter": chapter,
            "chapter_display": chapter_display,
            "chunk_index": metadata.get("chunk_index"),
            "full_chapter_text": full_text,
            "highlight_start": highlight_start,
            "highlight_end": highlight_end,
            "paragraph_text": paragraph_text,
            "paragraph_highlight_start": paragraph_highlight_start,
            "paragraph_highlight_end": paragraph_highlight_end,
            "total_chunks_in_chapter": len(chapter_chunks),
        }
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - passthrough for runtime failures
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@lru_cache(maxsize=1)
def load_chunks_data() -> list[dict]:
    """Load and cache the chunks.json file."""

    chunks_path = Path(__file__).parent.parent / "data" / "processed" / "chunks.json"
    if not chunks_path.exists():
        raise RuntimeError(f"Chunks data not found at {chunks_path}")

    import json

    with open(chunks_path, "r", encoding="utf-8") as file:
        return json.load(file)


@app.get("/books/{book_name}")
async def get_book_text(book_name: str) -> Dict[str, str]:
    """Retrieve the full raw text of a book from the source directory."""

    source_dir = Path(__file__).parent.parent / "data" / "source"
    resolved_book_name = unquote(book_name).strip()
    requested_stem = Path(resolved_book_name).stem
    book_path = source_dir / f"{requested_stem}.txt"
    decoded_name = resolved_book_name

    if not book_path.exists():
        # Try to find a match if the extension was already included or casing differs
        matches = list(source_dir.glob("*.txt"))
        normalized_requested = re.sub(r"^\d+[-_]", "", requested_stem).strip().lower()
        found_path = None
        for p in matches:
            normalized_stem = re.sub(r"^\d+[-_]", "", p.stem).strip().lower()
            if (
                p.stem.lower() == requested_stem.lower()
                or p.name.lower() == resolved_book_name.lower()
                or normalized_stem == normalized_requested
                or p.stem.lower().endswith(requested_stem.lower())
            ):
                found_path = p
                break

        if not found_path:
            raise HTTPException(
                status_code=404,
                detail={
                    "message": f"Book '{book_name}' not found in source repository",
                    "requested_stem": requested_stem,
                    "decoded_name": decoded_name,
                    "normalized_requested": normalized_requested,
                    "sample_stems": [p.stem for p in matches[:5]],
                    "sample_normalized_stems": [
                        re.sub(r"^\\d+[-_]", "", p.stem).strip().lower() for p in matches[:5]
                    ],
                },
            )
        book_path = found_path

    try:
        raw_bytes = book_path.read_bytes()
        content = None
        selected_encoding = None
        for encoding in ("utf-8", "utf-8-sig", "gb18030", "gbk", "big5"):
            try:
                content = raw_bytes.decode(encoding)
                selected_encoding = encoding
                break
            except UnicodeDecodeError:
                continue

        if content is None:
            raise UnicodeDecodeError(
                "unknown",
                raw_bytes,
                0,
                min(len(raw_bytes), 1),
                "Unable to decode source file with supported encodings",
            )
        return {"content": content}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read book: {exc}")


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


async def generate_arena_sse_stream(
    question: str,
    chat_history_a: list[dict],
    chat_history_b: list[dict],
    model_name: str,
    session_id: str,
    round_number: int,
    stream_timeout_seconds: float | None = None,
) -> AsyncGenerator[str, None]:
    import json as _json

    timeout_seconds = resolve_arena_stream_timeout_seconds(stream_timeout_seconds)
    poll_timeout_seconds = min(1.0, timeout_seconds)
    loop = asyncio.get_running_loop()

    assignment = random.choice(["rag_a_plain_b", "rag_b_plain_a"])
    if assignment == "rag_a_plain_b":
        position_mapping = {"a": "rag", "b": "plain"}
        gen_a = generate_rag_response(question, chat_history_a, model_name)
        gen_b = generate_raw_llm_response(question, chat_history_b, model_name)
    else:
        position_mapping = {"a": "plain", "b": "rag"}
        gen_a = generate_raw_llm_response(question, chat_history_a, model_name)
        gen_b = generate_rag_response(question, chat_history_b, model_name)

    queue: asyncio.Queue[tuple[str, object | None]] = asyncio.Queue()
    open_panels = {"a", "b"}
    panel_last_activity = {
        "a": loop.time(),
        "b": loop.time(),
    }
    errored_panels: set[str] = set()

    async def close_async_generator(generator: AsyncIterator[Any]) -> None:
        aclose = getattr(generator, "aclose", None)
        if aclose is None:
            return

        try:
            await aclose()
        except Exception:
            return

    async def drain_panel(panel: Literal["a", "b"], generator: AsyncIterator[Any]) -> None:
        try:
            async for item in generator:
                await queue.put((panel, item))
        except asyncio.CancelledError:
            await close_async_generator(generator)
            raise
        except Exception as exc:
            await queue.put((f"error_{panel}", str(exc)))
        finally:
            await queue.put((f"done_{panel}", None))

    producer_tasks: dict[str, asyncio.Task[None]] = {
        "a": asyncio.create_task(drain_panel("a", gen_a)),
        "b": asyncio.create_task(drain_panel("b", gen_b)),
    }

    try:
        while open_panels:
            now = loop.time()
            timed_out_panels = [
                panel for panel in tuple(open_panels) if now - panel_last_activity[panel] >= timeout_seconds
            ]
            for panel in timed_out_panels:
                errored_panels.add(panel)
                open_panels.discard(panel)
                producer_task = producer_tasks[panel]
                if not producer_task.done():
                    producer_task.cancel()
                yield (
                    "event: error\n"
                    f"data: {_json.dumps({'panel': panel, 'message': f'Stream timed out after {int(timeout_seconds)} seconds'})}\n\n"
                )

            if not open_panels:
                break

            try:
                slot, item = await asyncio.wait_for(queue.get(), timeout=poll_timeout_seconds)
            except asyncio.TimeoutError:
                continue

            if slot == "done_a":
                open_panels.discard("a")
                continue

            if slot == "done_b":
                open_panels.discard("b")
                continue

            if slot == "error_a":
                errored_panels.add("a")
                open_panels.discard("a")
                panel_last_activity["a"] = loop.time()
                yield f"event: error\ndata: {_json.dumps({'panel': 'a', 'message': str(item)})}\n\n"
                continue

            if slot == "error_b":
                errored_panels.add("b")
                open_panels.discard("b")
                panel_last_activity["b"] = loop.time()
                yield f"event: error\ndata: {_json.dumps({'panel': 'b', 'message': str(item)})}\n\n"
                continue

            if slot == "a":
                if "a" not in open_panels:
                    continue
                panel_last_activity["a"] = loop.time()
                if isinstance(item, dict):
                    if item.get("type") == "metadata":
                        yield f"event: metadata_a\ndata: {_json.dumps(item)}\n\n"
                    else:
                        chunk = str(item.get("content", "")).replace("\n", "\\n")
                        yield f"event: text_a\ndata: {chunk}\n\n"
                else:
                    chunk = str(item).replace("\n", "\\n")
                    yield f"event: text_a\ndata: {chunk}\n\n"
            elif slot == "b":
                if "b" not in open_panels:
                    continue
                panel_last_activity["b"] = loop.time()
                if isinstance(item, dict):
                    if item.get("type") == "metadata":
                        yield f"event: metadata_b\ndata: {_json.dumps(item)}\n\n"
                    else:
                        chunk = str(item.get("content", "")).replace("\n", "\\n")
                        yield f"event: text_b\ndata: {chunk}\n\n"
                else:
                    chunk = str(item).replace("\n", "\\n")
                    yield f"event: text_b\ndata: {chunk}\n\n"
    finally:
        for producer_task in producer_tasks.values():
            if not producer_task.done():
                producer_task.cancel()

        try:
            producer_results = await asyncio.wait_for(
                asyncio.gather(*producer_tasks.values(), return_exceptions=True),
                timeout=timeout_seconds,
            )
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            for panel, producer_task in producer_tasks.items():
                if not producer_task.done() and panel not in errored_panels:
                    errored_panels.add(panel)
                    yield (
                        "event: error\n"
                        f"data: {_json.dumps({'panel': panel, 'message': f'Stream cancellation timed out after {int(timeout_seconds)} seconds'})}\n\n"
                    )

            try:
                producer_results = await asyncio.wait_for(
                    asyncio.gather(*producer_tasks.values(), return_exceptions=True),
                    timeout=1.0,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                producer_results = [None, None]
        except Exception as exc:
            producer_results = [exc, exc]

        for panel, result in zip(("a", "b"), producer_results):
            if panel in errored_panels:
                continue
            if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
                yield f"event: error\ndata: {_json.dumps({'panel': panel, 'message': str(result)})}\n\n"

    yield (
        "event: arena_config\n"
        f"data: {_json.dumps({'position_mapping': position_mapping, 'session_id': session_id, 'round_number': round_number})}\n\n"
    )


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
async def get_arena_stats():
    """Compute arena evaluation statistics with T-Test."""
    import json
    from scipy import stats as scipy_stats
    
    votes_path = Path(__file__).parent.parent / "data" / "feedback" / "arena_votes.jsonl"
    if not votes_path.exists():
        return {"total_votes": 0, "votes": [], "statistics": None}
    
    votes = []
    with open(votes_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                votes.append(json.loads(line))
    
    if not votes:
        return {"total_votes": 0, "votes": [], "statistics": None}
    
    # Compute win counts
    rag_wins = 0
    plain_wins = 0
    ties = 0
    rag_scores = []  # 1 if user preferred RAG, 0 if preferred plain, 0.5 if tie
    
    for v in votes:
        mapping = v.get("position_mapping", {})
        vote = v.get("vote", "")
        
        # Determine which side was RAG
        rag_side = None
        for panel, role in mapping.items():
            if role == "rag":
                rag_side = panel
                break
        
        if not rag_side:
            continue
        
        if vote == "tie":
            ties += 1
            rag_scores.append(0.5)
        elif vote == rag_side:
            rag_wins += 1
            rag_scores.append(1.0)
        else:
            plain_wins += 1
            rag_scores.append(0.0)
    
    total = rag_wins + plain_wins + ties
    
    # Paired t-test: compare RAG scores vs Plain scores for same queries
    t_test = None
    if len(rag_scores) >= 3:
        plain_scores = [1.0 - s for s in rag_scores]  # mirror: RAG win=1→Plain=0, tie=0.5→0.5
        t_stat, p_value = scipy_stats.ttest_rel(rag_scores, plain_scores)
        # Cohen's d effect size
        import numpy as np
        diffs = np.array(rag_scores) - np.array(plain_scores)
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs, ddof=1)
        cohens_d = mean_diff / std_diff if std_diff > 0 else 0
        t_stat_value = t_stat[0] if isinstance(t_stat, tuple) else t_stat
        p_value_value = p_value[0] if isinstance(p_value, tuple) else p_value
        cohens_d_value = cohens_d[0] if isinstance(cohens_d, tuple) else cohens_d
        mean_score = float(np.mean(rag_scores))
        mean_score_value = mean_score
        t_stat_float = float(t_stat_value)
        p_value_float = float(p_value_value)
        cohens_d_float = float(cohens_d_value)
        mean_score_float = float(mean_score_value)
        
        t_test = {
            "t_statistic": round(t_stat_float, 4),
            "p_value": round(p_value_float, 6),
            "cohens_d": round(cohens_d_float, 4),
            "mean_rag_score": round(mean_score_float, 4),
            "sample_size": len(rag_scores),
            "significant": bool(p_value_float < 0.05),
            "interpretation": (
                "Statistically significant preference for RAG" if p_value_float < 0.05 and mean_score_float > 0.5
                else "Statistically significant preference for Plain LLM" if p_value_float < 0.05 and mean_score_float < 0.5
                else "No statistically significant difference detected"
            ),
        }
    
    # Per-query breakdown
    query_results = []
    for v in votes:
        mapping = v.get("position_mapping", {})
        vote = v.get("vote", "")
        rag_side = None
        for panel, role in mapping.items():
            if role == "rag":
                rag_side = panel
                break
        
        winner = "tie" if vote == "tie" else ("rag" if vote == rag_side else "plain")
        query_results.append({
            "query": v.get("query", ""),
            "winner": winner,
            "model": v.get("model_name", ""),
            "timestamp": v.get("timestamp", ""),
            "session_id": v.get("session_id", ""),
        })
    
    return {
        "total_votes": total,
        "rag_wins": rag_wins,
        "plain_wins": plain_wins,
        "ties": ties,
        "rag_win_rate": round(rag_wins / total * 100, 1) if total > 0 else 0,
        "plain_win_rate": round(plain_wins / total * 100, 1) if total > 0 else 0,
        "tie_rate": round(ties / total * 100, 1) if total > 0 else 0,
        "t_test": t_test,
        "query_results": query_results,
    }

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        reload=False,
    )
