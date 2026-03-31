# Testing Patterns

**Analysis Date:** 2026-04-01

## Test Framework

**Runner:**
- Python validation is script-driven, not pytest-driven.
- `requirements.txt` does not include `pytest`.
- Tests in `src/test_*.py` are plain Python functions with `assert`, usually executed via `if __name__ == "__main__":`.

**Assertion Library:**
- Built-in `assert` only (examples in `src/test_citations.py`, `src/test_graph.py`, `src/test_hybrid_retriever.py`).

**Run Commands (current project reality):**
```bash
venv\Scripts\python.exe src\test_citations.py
venv\Scripts\python.exe src\test_graph.py
venv\Scripts\python.exe src\test_hybrid_retriever.py
venv\Scripts\python.exe src\test_retriever.py
venv\Scripts\python.exe src\test_kg_extraction.py
venv\Scripts\python.exe scripts\verify_symmap_retrieval.py
cd web && npm run lint
```

## Test File Organization

**Location pattern:**
- Backend tests are co-located with source code under `src/`, not in a dedicated `tests/` tree.
- Additional verification scripts live under `scripts/`.

**Naming pattern:**
- `src/test_<feature>.py` for module-focused checks.
- `scripts/verify_*.py`, `scripts/check_*.py`, `scripts/e2e_test.py` for operator-driven validation and diagnostics.

**Current layout:**
```
src/
├── test_citations.py
├── test_graph.py
├── test_hybrid_retriever.py
├── test_retriever.py
└── test_kg_extraction.py

scripts/
├── verify_symmap_retrieval.py
├── check_health.py
└── e2e_test.py
```

## Test Structure

**Suite organization pattern:**
```python
def test_example_behavior():
    # arrange
    # act
    # assert
    assert condition

if __name__ == "__main__":
    test_example_behavior()
    print("All tests passed.")
```

**Observed patterns:**
- Unit-like function tests: `src/test_citations.py`.
- Graph domain tests with temporary files and in-memory graph setup: `src/test_graph.py`.
- Integration tests with data-presence guards and conditional skips: `src/test_hybrid_retriever.py`.
- Print-heavy smoke scripts (manual interpretation required): `src/test_retriever.py`, `src/test_kg_extraction.py`, `scripts/e2e_test.py`.

## Mocking, Fixtures, and Data Setup

**Mocking:**
- No dedicated mocking framework pattern is currently used.
- Tests prefer real objects (`langchain_core.documents.Document`, `TCMKnowledgeGraph`) and real on-disk resources when available.

**Fixture strategy:**
- Inline fixtures inside each file.
- No shared fixture registry (`conftest.py`) or factory module.

**Environment/data dependency pattern:**
```python
if not vectorstore_path.exists():
    print("⚠️ ... skipped ...")
    return
```
- This appears in `src/test_hybrid_retriever.py` and similar scripts; behavior is practical but can hide regressions when assets are missing.

## Frontend Verification Strategy

**Automated checks currently present:**
- `web/package.json` includes `lint` only (`eslint`).
- ESLint is configured by `web/eslint.config.mjs`.

**Automated checks currently missing:**
- No `jest`/`vitest` component tests in `web/`.
- No browser E2E test framework wired in `web/package.json`.

## CI and Enforcement

**Repository CI state:**
- No active `.github/workflows/*.yml` or `.github/workflows/*.yaml` detected.
- Verification runs are currently manual/developer-triggered.

**Coverage enforcement:**
- No coverage threshold configuration detected.
- No centralized test report/coverage publishing pipeline detected.

## Gaps and Risks in Current Verification

**High-impact gaps:**
- Frontend behavior can regress without automated component/E2E tests (`web/components/*`, `web/hooks/*`).
- Integration checks may silently skip when `vectorstore/chroma` or graph files are absent (`src/test_hybrid_retriever.py`).
- Script-style tests (`src/test_retriever.py`, `src/test_kg_extraction.py`, `scripts/e2e_test.py`) rely on manual output inspection instead of strict assertions.
- No CI gate means merges can occur without lint or backend test execution.

**Operational risks:**
- Contract drift risk between `src/citation_types.py` and `web/lib/types.ts` if tests are not run together.
- Env-dependent behavior in `src/api.py` and `src/main.py` can change runtime output with minimal automated guardrails.

## Recommended Practical Checks Before Merge

Run from repo root using project venv:

```bash
venv\Scripts\python.exe src\test_citations.py
venv\Scripts\python.exe src\test_graph.py
venv\Scripts\python.exe src\test_hybrid_retriever.py
venv\Scripts\python.exe scripts\verify_symmap_retrieval.py
cd web && npm run lint
```

If the change touches retrieval, graph, or ingestion code, also run:

```bash
venv\Scripts\python.exe src\test_retriever.py
venv\Scripts\python.exe src\test_kg_extraction.py
venv\Scripts\python.exe scripts\e2e_test.py
```

Manual UX sanity checks after backend/frontend start:
- Start backend: `venv\Scripts\python.exe src\api.py`
- Start frontend: `cd web && npm run dev`
- Validate end-to-end flows on `web/app/page.tsx`, `web/components/CitationPanel.tsx`, and `web/app/source/[chunkId]/page.tsx`.

## Near-Term Hardening Priorities

- Add a consistent Python test runner (`pytest`) while keeping existing assertions in `src/test_*.py`.
- Add at least one CI workflow for backend checks + frontend lint.
- Add API/contract tests around citation payload shape bridging `src/api.py` and `web/lib/api.ts`.
- Add one browser-level smoke test for chat stream + citation panel rendering.

---

*Testing analysis: 2026-04-01*
