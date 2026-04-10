<!-- generated-by: gsd-doc-writer -->
# TESTING.md — TCM-Sage

TCM-Sage uses **standalone Python scripts** for backend testing — no pytest, no test runner framework. Each script runs directly with the project venv and exits with a summary. Frontend validation is handled by ESLint only (no JS/TS test suite). Arena blind evaluation provides a live A/B quality signal at `/arena`.

---

## Test Framework and Setup

### Backend

There is **no pytest**. All backend tests are standalone Python scripts with `if __name__ == "__main__":` blocks that execute test functions sequentially and print pass/fail to stdout.

**Prerequisites before running any backend test:**

1. Project venv activated (or invoke via full path — see below).
2. Backend dependencies installed:
   ```bash
   # Windows
   venv\Scripts\python.exe -m pip install -r requirements.txt

   # macOS / Linux
   venv/bin/python -m pip install -r requirements.txt
   ```
3. `.env` file present with at least `DASHSCOPE_API_KEY` set (required by integration tests that call the embedding model or LLM).

**Import pattern:** All test scripts use a `sys.path` bootstrap to add `src/` to the module search path because `src/` is not a Python package:

```python
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
```

Scripts in `scripts/` that import from `src/` use `ROOT / "src"` as the insertion target instead.

### Frontend

ESLint (v9 with `eslint-config-next`) is the only frontend quality check. There is no Jest, Vitest, Playwright, or any other frontend test suite.

```bash
cd web && npm run lint
```

---

## Running Tests

### Core Backend Tests (`src/`)

Run from the **project root**:

```bash
# Citation formatting and reconstruction logic
venv\Scripts\python.exe src/test_citations.py

# Knowledge graph unit tests (entity CRUD, traversal, serialization)
venv\Scripts\python.exe src/test_graph.py

# Hybrid retriever: vector search + graph ensemble
venv\Scripts\python.exe src/test_hybrid_retriever.py

# Arena backend: vote storage, model config, prompt isolation
venv\Scripts\python.exe src/test_arena.py
```

> **macOS / Linux:** replace `venv\Scripts\python.exe` with `venv/bin/python` in all commands.

### Verification Scripts (`scripts/`)

Sanity-check scripts that confirm live data and running services behave correctly:

```bash
# SymMap KG graph traversal and ui_backend integration
venv\Scripts\python.exe scripts/verify_symmap_retrieval.py

# /source/<id>/context API endpoint shape (requires backend running on :8000)
venv\Scripts\python.exe scripts/verify_context_endpoint.py
```

### RAG Quality Evaluation Scripts (`scripts/`)

These scripts require a running ChromaDB vectorstore (`vectorstore/chroma/`) and valid API credentials. They produce comparative output — not pass/fail — to assess retrieval and generation quality:

```bash
# Side-by-side RAG vs plain LLM answers across 10 TCM questions
venv\Scripts\python.exe scripts/test_llm_vs_rag.py

# Answer length and formatting comparison (Pattern Priming validation)
venv\Scripts\python.exe scripts/test_answer_length.py

# Context format impact: plain text vs markdown-structured context
venv\Scripts\python.exe scripts/test_context_format.py
```

### Utility Test Scripts (`scripts/`)

```bash
# DashScope API connectivity and embedding round-trip
venv\Scripts\python.exe scripts/test_tongyi.py

# Arena vote write concurrency stress test (requires valid LLM credentials)
venv\Scripts\python.exe scripts/test_arena_concurrency.py

# ChromaDB document ID and metadata key inspection
venv\Scripts\python.exe scripts/test_chroma_ids.py

# Single-chunk retrieval diagnostic
venv\Scripts\python.exe scripts/test_single_chunk.py
```

### Older Vector Store Smoke Test (`src/`)

```bash
# ChromaDB load + similarity search using legacy HuggingFace embeddings
# Note: uses nomic-embed model, not production DashScope embedding
venv\Scripts\python.exe src/test_retriever.py
```

### Frontend Lint

```bash
cd web
npm run lint
```

---

## Writing New Tests

### Naming Convention

| Location | Pattern | Example |
|----------|---------|---------|
| `src/` | `test_<feature>.py` | `test_citations.py` |
| `scripts/` | `test_<feature>.py` or `verify_<feature>.py` | `verify_symmap_retrieval.py` |

Use `test_` prefix for unit/integration tests with assertions. Use `verify_` prefix for sanity checks that print diagnostic output and `assert` on critical invariants.

### Script Structure

Every new test script must follow this pattern:

```python
"""
Module docstring describing what is tested.
"""

import sys
from pathlib import Path

# sys.path bootstrap — src/ is not a package
SRC_DIR = Path(__file__).resolve().parent   # adjust for scripts/ if needed
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# --- imports from src/ ---
from my_module import my_function


def test_something():
    result = my_function("input")
    assert result == "expected", f"Got: {result}"
    print("✅ test_something passed")


if __name__ == "__main__":
    test_something()
    print("All tests passed.")
```

For scripts that import from `src/` but live in `scripts/`:

```python
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
```

### Test Design Guidelines

- **Unit tests** (no external dependencies): place in `src/`, use mock data or `tempfile`.
- **Integration tests** (require vectorstore or live API): guard with `if not path.exists(): print("⚠️ skipped"); return` to make them skippable in CI or clean environments.
- **Graph tests**: build minimal in-memory `TCMKnowledgeGraph` instances rather than loading production data where possible.
- **Arena tests**: use `tempfile` for vote storage path to avoid polluting `data/feedback/arena_votes.jsonl`.
- **No pytest fixtures or decorators**: all setup happens inside the function body.

---

## Coverage Requirements

No coverage threshold is configured. TCM-Sage does not use coverage measurement tooling (no `.nycrc`, no `pytest-cov`, no `c8`). Test completeness is validated by running all scripts and verifying zero assertion failures.

---

## Arena Blind Evaluation

The Arena is a live quality-evaluation system, not a unit test. It provides a human-in-the-loop signal to validate that RAG-augmented responses outperform plain LLM responses.

**Access:** Start the backend (`src/api.py`) and frontend (`web/npm run dev`), then navigate to:

| URL | Purpose |
|-----|---------|
| `http://localhost:3000/arena` | Blind A/B voting interface |
| `http://localhost:3000/arena/stats` | Statistical analysis of accumulated votes |

**Statistics page** (`/arena/stats`) computes:
- Win rates (RAG vs Plain LLM vs Tie)
- Paired t-test: t-statistic, p-value
- Cohen's d effect size
- Significance interpretation
- Downloadable bar chart and pie chart (PNG)

Vote records are appended to `data/feedback/arena_votes.jsonl` in real time.

**Arena backend test:** `src/test_arena.py` verifies the vote storage mechanism, `ARENA_MODELS` dictionary, `ArenaVoteRecord` TypedDict field completeness, and the invariant that `generate_raw_llm_response` does not inject RAG context into the prompt (which would compromise arena blinding).

---

## CI Integration

There is no CI/CD pipeline configured. No `.github/workflows/` files exist in the repository. All tests are run manually in the local development environment.

---

## Quick Reference

| Script | Type | Requires vectorstore | Requires API key |
|--------|------|----------------------|-----------------|
| `src/test_citations.py` | Unit | No | No |
| `src/test_graph.py` | Unit | No | No |
| `src/test_arena.py` | Unit | No | No |
| `src/test_hybrid_retriever.py` | Integration | Yes (skips if absent) | No |
| `scripts/verify_symmap_retrieval.py` | Verification | No | No |
| `scripts/verify_context_endpoint.py` | Integration | No | No (requires backend on :8000) |
| `scripts/test_llm_vs_rag.py` | Quality | Yes | Yes |
| `scripts/test_answer_length.py` | Quality | Yes | Yes |
| `scripts/test_context_format.py` | Quality | No | Yes |
| `scripts/test_arena_concurrency.py` | Integration | No | Yes |
| `src/test_retriever.py` | Legacy smoke test | Yes | No |
