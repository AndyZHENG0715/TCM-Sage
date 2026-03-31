"""Minimal spike test for concurrent DashScope streaming calls."""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

load_dotenv(dotenv_path=ROOT_DIR / ".env")

from config import DEFAULT_LLM_PROVIDER  # noqa: E402
from main import create_llm  # noqa: E402


PROMPT = "What is qi in TCM?"


@dataclass
class StreamResult:
    name: str
    content: str
    elapsed: float


def resolve_provider_model() -> tuple[str, str | None]:
    provider = os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER).strip().lower()
    raw_model = os.getenv("LLM_MODEL", "").strip()
    return provider, (raw_model or None)


def is_rate_limited(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in ("rate limit", "rate-limit", "too many requests", "429", "quota"))


async def consume_stream(llm, name: str) -> StreamResult:
    started_at = perf_counter()
    pieces: list[str] = []
    messages = [HumanMessage(content=PROMPT)]

    if hasattr(llm, "astream"):
        async for chunk in llm.astream(messages):
            text = getattr(chunk, "content", "")
            if text:
                pieces.append(text if isinstance(text, str) else str(text))
    elif hasattr(llm, "stream"):
        def _collect_sync() -> list[str]:
            sync_pieces: list[str] = []
            for chunk in llm.stream(messages):
                text = getattr(chunk, "content", "")
                if text:
                    sync_pieces.append(text if isinstance(text, str) else str(text))
            return sync_pieces

        pieces = await asyncio.to_thread(_collect_sync)
    else:
        raise RuntimeError("LLM does not expose astream() or stream().")

    return StreamResult(name=name, content="".join(pieces).strip(), elapsed=perf_counter() - started_at)


async def run() -> int:
    provider, model = resolve_provider_model()
    print(f"Provider: {provider} | Model: {model or '(default)'}")
    print(f"Start: {datetime.now().isoformat(sep=' ', timespec='seconds')}")

    try:
        llm = create_llm(provider=provider, model=model, streaming=True)
    except Exception as exc:  # pragma: no cover - validation spike
        print(f"FAIL: {type(exc).__name__}: {exc}")
        return 1

    overall_started = perf_counter()
    results = await asyncio.gather(
        consume_stream(llm, "Stream 1"),
        consume_stream(llm, "Stream 2"),
        return_exceptions=True,
    )
    overall_elapsed = perf_counter() - overall_started

    exceptions = [result for result in results if isinstance(result, Exception)]
    if exceptions:
        if any(is_rate_limited(exc) for exc in exceptions):
            print("FAIL (rate limited — sequential fallback needed)")
            print(f"Detail: {type(exceptions[0]).__name__}: {exceptions[0]}")
        else:
            exc = exceptions[0]
            print(f"FAIL: {type(exc).__name__}: {exc}")
        return 1

    typed_results = [result for result in results if isinstance(result, StreamResult)]
    if len(typed_results) != 2 or any(not result.content for result in typed_results):
        print("FAIL: one or more streams completed without content")
        for result in typed_results:
            print(f"{result.name} completed at +{result.elapsed:.2f}s ({len(result.content)} chars)")
        print(f"Sequential estimate: {sum(result.elapsed for result in typed_results):.2f}s")
        print(f"Concurrent actual: {overall_elapsed:.2f}s")
        return 1

    for result in typed_results:
        print(f"{result.name} completed at +{result.elapsed:.2f}s ({len(result.content)} chars)")

    sequential_estimate = sum(result.elapsed for result in typed_results)
    print(f"Sequential estimate: {sequential_estimate:.2f}s")
    print(f"Concurrent actual: {overall_elapsed:.2f}s")
    print("Total: {:.2f}s".format(overall_elapsed))
    print("PASS")
    return 0


def main() -> int:
    return asyncio.run(run())


if __name__ == "__main__":
    raise SystemExit(main())
