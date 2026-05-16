"""Source text lookup and reconstruction helpers for API routes."""

from __future__ import annotations

import json
import re
import traceback
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict
from urllib.parse import unquote

from fastapi import HTTPException

from ui_backend import get_shared_vectorstore

PROJECT_ROOT = Path(__file__).parent.parent
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.json"
SOURCE_DIR = PROJECT_ROOT / "data" / "source"

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


@lru_cache(maxsize=1)
def load_chunks_data() -> list[dict]:
    """Load and cache the chunks.json file."""

    if not CHUNKS_PATH.exists():
        raise RuntimeError(f"Chunks data not found at {CHUNKS_PATH}")

    with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


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
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc


async def get_book_text(book_name: str) -> Dict[str, str]:
    """Retrieve the full raw text of a book from the source directory."""

    resolved_book_name = unquote(book_name).strip()
    requested_stem = Path(resolved_book_name).stem
    book_path = SOURCE_DIR / f"{requested_stem}.txt"
    decoded_name = resolved_book_name

    if not book_path.exists():
        matches = list(SOURCE_DIR.glob("*.txt"))
        normalized_requested = re.sub(r"^\d+[-_]", "", requested_stem).strip().lower()
        found_path = None
        for path in matches:
            normalized_stem = re.sub(r"^\d+[-_]", "", path.stem).strip().lower()
            if (
                path.stem.lower() == requested_stem.lower()
                or path.name.lower() == resolved_book_name.lower()
                or normalized_stem == normalized_requested
                or path.stem.lower().endswith(requested_stem.lower())
            ):
                found_path = path
                break

        if not found_path:
            raise HTTPException(
                status_code=404,
                detail={
                    "message": f"Book '{book_name}' not found in source repository",
                    "requested_stem": requested_stem,
                    "decoded_name": decoded_name,
                    "normalized_requested": normalized_requested,
                    "sample_stems": [path.stem for path in matches[:5]],
                    "sample_normalized_stems": [
                        re.sub(r"^\\d+[-_]", "", path.stem).strip().lower() for path in matches[:5]
                    ],
                },
            )
        book_path = found_path

    try:
        raw_bytes = book_path.read_bytes()
        content = None
        for encoding in ("utf-8", "utf-8-sig", "gb18030", "gbk", "big5"):
            try:
                content = raw_bytes.decode(encoding)
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
        raise HTTPException(status_code=500, detail=f"Failed to read book: {exc}") from exc
