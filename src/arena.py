"""
TCM-Sage Arena Backend

Core functions for the Arena mode: blind A/B comparison of RAG vs raw LLM
responses. Provides type definitions, response generators, and vote storage.

This module does NOT define API endpoints — those live in api.py.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Literal, TypedDict, Union

from dotenv import load_dotenv

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from config import PROJECT_ROOT
from ui_backend import run_query_stream

load_dotenv()

# ---------------------------------------------------------------------------
# Arena model tiers — override with ARENA_MODELS env var (JSON string)
# ---------------------------------------------------------------------------
_DEFAULT_ARENA_MODELS: Dict[str, str] = {
    "flash": "qwen-flash",
    "plus": "qwen-plus",
    "max": "qwen-max",
}

_env_models_raw = os.getenv("ARENA_MODELS", "{}")
try:
    _env_models = json.loads(_env_models_raw)
except (json.JSONDecodeError, TypeError):
    _env_models = {}

ARENA_MODELS: Dict[str, str] = {**_DEFAULT_ARENA_MODELS, **_env_models}

# ---------------------------------------------------------------------------
# Type definitions
# ---------------------------------------------------------------------------


class ArenaVoteRecord(TypedDict):
    """Single arena vote persisted as one JSONL line."""

    session_id: str
    round_number: int
    query: str
    response_a: str
    response_b: str
    model_name: str
    position_mapping: dict  # e.g. {"a": "rag", "b": "plain"}
    vote: Literal["a", "b", "tie"]
    comment: str | None
    timestamp: str


class ArenaSessionConfig(TypedDict):
    """Configuration snapshot for an arena session."""

    session_id: str
    model_tier: str
    model_name: str
    provider: str


# ---------------------------------------------------------------------------
# Raw LLM response generator (NO retrieval)
# ---------------------------------------------------------------------------



async def generate_raw_llm_response(
    question: str,
    chat_history: list[dict],
    model_name: str,
) -> AsyncIterator[str]:
    """Async generator that streams plain LLM output *without* RAG context.

    Yields plain ``str`` chunks suitable for direct concatenation.
    """
    provider = os.getenv("LLM_PROVIDER", "alibaba").lower()

    from langchain_openai import ChatOpenAI
    api_key = os.getenv('DASHSCOPE_API_KEY', '')
    llm_base = ChatOpenAI(
        model=model_name,
        streaming=True,
        api_key=api_key,
        base_url='https://dashscope-intl.aliyuncs.com/compatible-mode/v1',
    )
    llm = llm_base
    # Build a simple history string (last 6 turns, same as ui_backend)
    history_lines: list[str] = []
    for msg in (chat_history or [])[-6:]:
        role = msg.get("role", "user").upper()
        content = msg.get("content", "")
        history_lines.append(f"{role}: {content}")
    history_text = "\n".join(history_lines)

    # Web search for grounding (mirrors what users get from ChatGPT/Gemini/Kimi)
    search_context = ""
    try:
        from ddgs import DDGS
        results = list(DDGS().text(question, max_results=5))
        if results:
            snippets = []
            for i, r in enumerate(results, 1):
                snippets.append(f"[{i}] {r.get('title', '')}: {r.get('body', '')}")
            search_context = "Web search results:\n" + "\n".join(snippets) + "\n\n"
            print(f'[Debug] Arena plain LLM: {len(results)} web search results for "{question[:30]}"')
    except Exception as e:
        print(f'[Debug] Arena web search failed (non-fatal): {e}')

    # Plain LLM uses default prompt + web search (no TCM-specific instructions)
    # — the system prompt and curated retrieval are part of TCM-Sage's engineering
    prompt_text = "You are a helpful assistant.\n\n"
    if search_context:
        prompt_text += search_context
    if history_text:
        prompt_text += f"Chat History:\n{history_text}\n\n"
    prompt_text += f"Question:\n{question}\n\n"
    async for chunk in llm.astream(prompt_text):
        # LangChain ChatModel chunks have a .content attribute
        text = chunk.content if hasattr(chunk, "content") else str(chunk)
        yield text


# ---------------------------------------------------------------------------
# RAG response generator (wraps run_query_stream)
# ---------------------------------------------------------------------------


async def generate_rag_response(
    question: str,
    chat_history: list[dict],
    model_name: str,
) -> AsyncIterator[Dict[str, Any]]:
    """Async generator wrapping the sync RAG pipeline with progressive streaming.

    Uses a thread + queue so chunks are yielded as they arrive,
    rather than waiting for the entire generation to complete.
    """
    import queue
    import threading

    provider = os.getenv("LLM_PROVIDER", "alibaba").lower()

    runtime_settings: Dict[str, Any] = {
        "provider": provider,
        "model": model_name,
    }

    q: queue.Queue[Dict[str, Any] | None] = queue.Queue()

    def _produce() -> None:
        try:
            for item in run_query_stream(
                user_query=question,
                chat_history=chat_history,
                runtime_settings=runtime_settings,
            ):
                if isinstance(item, dict):
                    q.put({
                        "type": "metadata",
                        "citations": item.get("citations", []),
                        "verification": item.get("verification"),
                    })
                else:
                    q.put({"type": "text", "content": item})
        except Exception as exc:
            q.put({"type": "error", "message": str(exc)})
        finally:
            q.put(None)  # sentinel

    thread = threading.Thread(target=_produce, daemon=True)
    thread.start()

    while True:
        while q.empty():
            await asyncio.sleep(0.02)
        item = q.get()
        if item is None:
            break
        yield item


# ---------------------------------------------------------------------------
# Vote storage (append-only JSONL)
# ---------------------------------------------------------------------------

_VOTES_PATH = PROJECT_ROOT / "data" / "feedback" / "arena_votes.jsonl"


def store_vote(record: ArenaVoteRecord) -> None:
    """Persist a single arena vote as a JSONL line.

    Creates the parent directory if it does not exist.
    """
    _VOTES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_VOTES_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
