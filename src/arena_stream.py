"""Dual-panel SSE streaming for Arena blind evaluation."""

from __future__ import annotations

import asyncio
import json
import os
import random
from typing import Any, AsyncGenerator, AsyncIterator, Literal

from arena import generate_raw_llm_response, generate_rag_response

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


async def _close_async_generator(generator: AsyncIterator[Any]) -> None:
    aclose = getattr(generator, "aclose", None)
    if aclose is None:
        return

    try:
        await aclose()
    except Exception:
        return


async def generate_arena_sse_stream(
    question: str,
    chat_history_a: list[dict],
    chat_history_b: list[dict],
    model_name: str,
    session_id: str,
    round_number: int,
    stream_timeout_seconds: float | None = None,
) -> AsyncGenerator[str, None]:
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

    async def drain_panel(panel: Literal["a", "b"], generator: AsyncIterator[Any]) -> None:
        try:
            async for item in generator:
                await queue.put((panel, item))
        except asyncio.CancelledError:
            await _close_async_generator(generator)
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
                    f"data: {json.dumps({'panel': panel, 'message': f'Stream timed out after {int(timeout_seconds)} seconds'})}\n\n"
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
                yield f"event: error\ndata: {json.dumps({'panel': 'a', 'message': str(item)})}\n\n"
                continue

            if slot == "error_b":
                errored_panels.add("b")
                open_panels.discard("b")
                panel_last_activity["b"] = loop.time()
                yield f"event: error\ndata: {json.dumps({'panel': 'b', 'message': str(item)})}\n\n"
                continue

            if slot == "a":
                if "a" not in open_panels:
                    continue
                panel_last_activity["a"] = loop.time()
                if isinstance(item, dict):
                    if item.get("type") == "metadata":
                        yield f"event: metadata_a\ndata: {json.dumps(item)}\n\n"
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
                        yield f"event: metadata_b\ndata: {json.dumps(item)}\n\n"
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
                        f"data: {json.dumps({'panel': panel, 'message': f'Stream cancellation timed out after {int(timeout_seconds)} seconds'})}\n\n"
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
                yield f"event: error\ndata: {json.dumps({'panel': panel, 'message': str(result)})}\n\n"

    yield (
        "event: arena_config\n"
        f"data: {json.dumps({'position_mapping': position_mapping, 'session_id': session_id, 'round_number': round_number})}\n\n"
    )
