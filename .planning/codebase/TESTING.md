# Testing Patterns

**Analysis Date:** 2026-03-23

## Test Framework

- **Python:** No formal test runner (e.g., `pytest`, `unittest`) is configured. Testing is performed via standalone Python scripts with `assert` statements.
- **TypeScript:** No formal test runner (e.g., `jest`, `vitest`) detected in `web/package.json`.

## Test Suites

### Integration Tests
- **RAG Pipeline:** `scripts/e2e_test.py` validates the full retrieval and KG extraction flow.
- **Retriever:** `src/test_retriever.py` directly tests ChromaDB similarity search.
- **Hybrid Retriever:** `src/test_hybrid_retriever.py` tests the combination of vector and KG retrieval.

### Quality & Audit Tests
- **KG Extraction Quality:** `scripts/quality_check.py` analyzes the extraction process, specifically tracking chunks that fail to produce entities.
- **Audit:** `scripts/comprehensive_audit.py` provides a high-level overview of the pipeline's status.

### Specialized Tests
- **Citations:** `src/test_citations.py` verifies the accuracy and formatting of generated citations.
- **Graph Construction:** `src/test_graph.py` validates the structure and connectivity of the knowledge graph.

## Testing Strategy

- **Manual Execution:** Tests are currently executed manually via the command line (e.g., `python scripts/e2e_test.py`).
- **Data-Centric:** Focuses on data integrity, retrieval accuracy, and the quality of the generated knowledge graph rather than unit testing individual logic branches.
