# Codebase Concerns

**Analysis Date:** 2026-03-23

## Tech Debt

- **Lack of Unified Test Runner:** Reliance on ad-hoc scripts (`scripts/`, `src/`) makes automated CI/CD and regression testing difficult.
- **Extraction Logic Duplication:** Logic for graph search and fact formatting may still be duplicated between `src/retriever.py` and `src/ui_backend.py`.
- **Manual Path Manipulation:** Frequent use of `sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))` in scripts is brittle.

## Operational Concerns

- **Extraction Quality:** A high number of chunks fail to produce entities during KG extraction (referenced in `scripts/quality_check.py`).
- **Encoding Issues:** Evidence of character encoding issues (mojibake) when handling Chinese text in some scripts (e.g., `scripts/e2e_test.py`).
- **Data Fragmentation:** Partial JSON files (e.g., `data/graph/entities_partial.json`) suggest an incomplete or fragmented knowledge graph state.

## Feature Gaps

- **Multi-Provider Streaming:** `src/main.py` contains a TODO regarding full support for multi-provider streaming.
- **KG Visualization Performance:** Potential performance issues with large-scale graph visualization in the UI.

## Reliability Risks

- **LLM Rate Limiting:** High dependency on external APIs (DashScope) for extraction and generation without robust retry/fallback mechanisms.
- **Consistency:** Risk of inconsistent retrieval results between the direct API (`src/api.py`) and the UI backend (`src/ui_backend.py`).
