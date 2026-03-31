"""Standalone tests for src/arena.py — no pytest required."""

from __future__ import annotations

import json
from collections.abc import Callable
import sys
import tempfile
import traceback
from pathlib import Path
from types import FunctionType
from typing import cast

# sys.path bootstrap — src/ is not a package
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import arena as _arena_mod  # noqa: E402

ARENA_MODELS = _arena_mod.ARENA_MODELS
ArenaVoteRecord = _arena_mod.ArenaVoteRecord
store_vote = _arena_mod.store_vote


def test_store_vote() -> None:
    """store_vote() appends a valid JSONL line."""
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
        tmp_path = Path(f.name)

    orig_path = cast(Path, getattr(_arena_mod, "_VOTES_PATH"))
    setattr(_arena_mod, "_VOTES_PATH", tmp_path)

    try:
        record: ArenaVoteRecord = {
            "session_id": "test-session-001",
            "round_number": 1,
            "query": "What is qi?",
            "response_a": "Qi is vital energy.",
            "response_b": "Qi is life force.",
            "model_name": "qwen-plus",
            "position_mapping": {"a": "rag", "b": "plain"},
            "vote": "a",
            "comment": None,
            "timestamp": "2026-03-31T00:00:00",
        }
        store_vote(record)

        lines = tmp_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1, f"Expected 1 line, got {len(lines)}"
        d = cast(dict[str, object], json.loads(lines[0]))
        assert d["session_id"] == "test-session-001"
        assert d["vote"] == "a"
        assert d["position_mapping"] == {"a": "rag", "b": "plain"}
    finally:
        setattr(_arena_mod, "_VOTES_PATH", orig_path)
        tmp_path.unlink(missing_ok=True)


def test_arena_models() -> None:
    """ARENA_MODELS has at least 3 entries with non-empty string values."""
    assert isinstance(ARENA_MODELS, dict), "ARENA_MODELS must be a dict"
    assert len(ARENA_MODELS) >= 3, f"Expected >= 3 model tiers, got {len(ARENA_MODELS)}"
    for key, val in ARENA_MODELS.items():
        assert isinstance(key, str) and key, f"Key must be non-empty str: {key!r}"
        assert isinstance(val, str) and val, f"Value must be non-empty str: {val!r}"


def test_raw_llm_prompt_has_no_context() -> None:
    """generate_raw_llm_response prompt template must NOT contain {context}."""
    import inspect

    source = inspect.getsource(cast(FunctionType, getattr(_arena_mod, "generate_raw_llm_response")))
    assert "{context}" not in source, (
        "generate_raw_llm_response must not inject {context} into the prompt"
    )


def test_vote_record_fields() -> None:
    """ArenaVoteRecord TypedDict has all required fields."""
    record: ArenaVoteRecord = {
        "session_id": "s1",
        "round_number": 1,
        "query": "q",
        "response_a": "a",
        "response_b": "b",
        "model_name": "qwen-plus",
        "position_mapping": {"a": "rag", "b": "plain"},
        "vote": "tie",
        "comment": "nice",
        "timestamp": "2026-01-01T00:00:00",
    }
    required_keys = {
        "session_id", "round_number", "query", "response_a", "response_b",
        "model_name", "position_mapping", "vote", "comment", "timestamp",
    }
    missing = required_keys - record.keys()
    assert not missing, f"Missing fields: {missing}"
    assert record["vote"] in ("a", "b", "tie"), f"Invalid vote: {record['vote']!r}"


def run_test(name: str, fn: Callable[[], None]) -> bool:
    try:
        fn()
        print(f"  ✅ {name}")
        return True
    except Exception:
        print(f"  ❌ {name}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    tests: list[tuple[str, Callable[[], None]]] = [
        ("test_store_vote", test_store_vote),
        ("test_arena_models", test_arena_models),
        ("test_raw_llm_prompt_has_no_context", test_raw_llm_prompt_has_no_context),
        ("test_vote_record_fields", test_vote_record_fields),
    ]

    print("Arena backend tests")
    print("=" * 40)
    results = [run_test(name, fn) for name, fn in tests]
    passed = sum(results)
    total = len(results)
    print(f"\n{passed}/{total} passed")
    sys.exit(0 if all(results) else 1)
