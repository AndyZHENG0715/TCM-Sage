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
from typing import Any, AsyncGenerator, Dict, Generator
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
    vote: str  # "a" | "b" | "tie"
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


@app.get("/config")
async def get_config():
    """Retrieve the current pipeline configuration for the UI."""
    from ui_backend import get_runtime_config
    try:
        config = get_runtime_config()
        return {
            "provider": config.provider,
            "model": config.model,
            "informational_temperature": config.informational_temperature,
            "prescriptive_temperature": config.prescriptive_temperature,
            "retrieval_k": config.retrieval_k,
            "hybrid_enabled": config.hybrid_enabled,
            "graph_depth": config.graph_depth,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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

        metadata = result["metadatas"][0]
        book = metadata.get("book")
        chapter = metadata.get("source")
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


async def generate_arena_sse_stream(
    question: str,
    chat_history_a: list[dict],
    chat_history_b: list[dict],
    model_name: str,
    session_id: str,
    round_number: int,
) -> AsyncGenerator[str, None]:
    import json as _json

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

    async def drain_a() -> None:
        try:
            async for item in gen_a:
                await queue.put(("a", item))
        except Exception as exc:
            await queue.put(("error_a", str(exc)))
        finally:
            await queue.put(("done_a", None))

    async def drain_b() -> None:
        try:
            async for item in gen_b:
                await queue.put(("b", item))
        except Exception as exc:
            await queue.put(("error_b", str(exc)))
        finally:
            await queue.put(("done_b", None))

    producers = asyncio.gather(drain_a(), drain_b(), return_exceptions=True)
    done_count = 0

    try:
        while done_count < 2:
            slot, item = await queue.get()

            if slot == "done_a" or slot == "done_b":
                done_count += 1
                continue

            if slot == "error_a":
                yield f"event: error\ndata: {_json.dumps({'panel': 'a', 'message': str(item)})}\n\n"
                continue

            if slot == "error_b":
                yield f"event: error\ndata: {_json.dumps({'panel': 'b', 'message': str(item)})}\n\n"
                continue

            if slot == "a":
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
        producers.cancel()
        try:
            await producers
        except Exception:
            pass

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


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        reload=False,
    )
