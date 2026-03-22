# Testing Patterns

**Analysis Date:** 2025-05-14

## Test Framework

**Runner:**
- **Python:** Standalone Python scripts with `assert` statements. No dedicated runner (like `pytest`) found in `requirements.txt`, but tests are structured to be compatible with `pytest`.
- **TypeScript:** Not detected (no `vitest`, `jest`, or `cypress` in `package.json`).

**Assertion Library:**
- **Python:** Built-in `assert` statements.
- **TypeScript:** Not detected.

**Run Commands:**
```bash
python src/test_citations.py         # Test citation system
python src/test_graph.py             # Test graph builder
python src/test_hybrid_retriever.py  # Test hybrid retriever
python scripts/e2e_test.py           # End-to-end query tests
```

## Test File Organization

**Location:**
- **Python:** Co-located in `src/` for unit/integration tests and in `scripts/` for E2E/utility tests.
- **TypeScript:** Not detected.

**Naming:**
- **Python:** `test_*.py` (e.g., `src/test_retriever.py`) or `*_test.py` (e.g., `scripts/e2e_test.py`).

**Structure:**
```
src/
├── test_citations.py
├── test_graph.py
├── test_hybrid_retriever.py
├── test_kg_extraction.py
└── test_retriever.py

scripts/
├── e2e_test.py
├── quality_check.py
├── test_single_chunk.py
└── verify_context_endpoint.py
```

## Test Structure

**Suite Organization:**
```python
def test_function_name():
    # Setup
    # Execution
    # Assertion
    assert condition

if __name__ == "__main__":
    test_function_name()
    print("Test passed")
```

**Patterns:**
- **Setup:** Manual instantiation of classes (e.g., `kg = TCMKnowledgeGraph()`).
- **Teardown:** Occasional use of `tempfile` and `Path.unlink()` (e.g., in `src/test_graph.py`).
- **Assertion:** Direct use of `assert` to verify counts, types, and content.

## Mocking

**Framework:** Not explicitly used (no `unittest.mock` or `pytest-mock` seen in test files).

**Patterns:**
- **Manual Mocks:** Use of small, hardcoded data structures to represent entities or documents for isolated testing (e.g., `docs = [Document(...)]` in `src/test_citations.py`).

**What to Mock:**
- API responses (not explicitly mocked in found tests; some tests perform real searches).
- File system (using `tempfile`).

**What NOT to Mock:**
- Core logic classes like `TCMKnowledgeGraph` or `HybridRetriever` (tested as integrated units).

## Fixtures and Factories

**Test Data:**
```python
data = {
    "entities": [
        {"id": "s1", "type": "Symptom", "name": "頭痛", "name_en": "Headache"},
        {"id": "h1", "type": "Herb", "name": "川芎", "name_en": "Chuanxiong"},
    ],
    "relationships": [
        {"source": "h1", "target": "s1", "type": "TREATS", "description": "Test"}
    ],
}
```

**Location:**
- Hardcoded within test functions in `src/` test files.
- Actual project data located in `data/graph/` and `vectorstore/` is used for integration tests.

## Coverage

**Requirements:** None enforced.

**View Coverage:**
No coverage tools (like `coverage.py`) detected in `requirements.txt`.

## Test Types

**Unit Tests:**
- Focused on individual components like `TCMKnowledgeGraph` (in `src/test_graph.py`) and citation formatting (in `src/test_citations.py`).

**Integration Tests:**
- Verifying the interaction between the vector store and the Knowledge Graph (in `src/test_hybrid_retriever.py`).

**E2E Tests:**
- Query-based tests in `scripts/e2e_test.py` that exercise the full retrieval pipeline against real data.

## Common Patterns

**Async Testing:**
- Not observed in the found test files, although the core API (`src/api.py`) and UI backend use async patterns.

**Error Testing:**
- Use of `try...except` in test scripts to catch and report errors during manual execution.
- Verification of citation bounds and valid/invalid scenarios in `src/test_citations.py`.

---

*Testing analysis: 2025-05-14*
