# Codebase Concerns

**Analysis Date:** 2025-02-18

## Tech Debt

**Knowledge Graph Extraction Logic Duplication:**
- Issue: Logic for graph search and fact formatting is duplicated between `src/retriever.py` and `src/ui_backend.py`.
- Files: `src/retriever.py`, `src/ui_backend.py`
- Impact: Inconsistent behavior and higher maintenance effort. Changes to graph search must be applied in multiple places.
- Fix approach: Refactor `ui_backend.py` to use the `HybridRetriever` class from `src/retriever.py`.

**Brittle JSON Extraction:**
- Issue: LLM outputs are parsed for JSON using regex patterns, which is fragile and fails on malformed or complex outputs.
- Files: `src/kg_extractor.py`
- Impact: Extraction failures and data loss when LLM output doesn't perfectly match expected regex patterns.
- Fix approach: Use structured output libraries (like Pydantic with LangChain) or more robust JSON repair libraries.

**Pickle for Graph Persistence:**
- Issue: The knowledge graph is persisted using Python's `pickle` module.
- Files: `src/graph_builder.py`
- Impact: Fragile across Python versions and potentially insecure if loading untrusted data.
- Fix approach: Standardize on JSON/JSONL for small-to-medium graphs or transition to a proper graph database (e.g., Neo4j, Kùzu) for scaling.

**Hardcoded TCM Variant Mappings:**
- Issue: Simplified to Traditional Chinese mapping is handled by a small, hardcoded dictionary.
- Files: `src/graph_builder.py`
- Impact: Search misses many valid entities due to incomplete character variant coverage.
- Fix approach: Use a dedicated library like `OpenCC` for robust Chinese variant conversion.

## Performance Bottlenecks

**Sequential KG Extraction (O(3N) Complexity):**
- Issue: KG extraction uses a 3-pass LLM approach (Entity -> Relation -> Critique) processed sequentially per chunk.
- Files: `src/kg_extractor.py`, `src/extract_kg_durable.py`
- Cause: Lack of concurrency/batching for LLM requests. Processing the full ~4.1MB text (approx. 4,000-8,000 chunks) could take 30+ hours.
- Improvement path: Implement async/concurrent processing for chunks and use batch API features if available from the provider.

**In-Memory Graph Traversal:**
- Issue: Large graphs using `NetworkX` are stored entirely in RAM.
- Files: `src/graph_builder.py`
- Cause: Architectural choice for prototype simplicity.
- Improvement path: Transition to a disk-backed graph database for larger datasets.

**Durable Save Write Overhead:**
- Issue: `extract_kg_durable.py` saves the entire `entities_partial.json` file every 5 chunks.
- Files: `src/extract_kg_durable.py`
- Cause: Inefficient persistence strategy for incremental updates.
- Improvement path: Use JSONL (JSON Lines) to append new extractions without re-writing the entire dataset.

## Fragile Areas

**Entity Resolution:**
- Files: `src/kg_extractor.py`, `src/graph_builder.py`
- Why fragile: Current deduplication is based on exact ID matches. Slight variations in LLM-extracted names (e.g., "手太阴" vs "手太阴肺经") result in duplicate or fragmented nodes.
- Safe modification: Implement fuzzy matching or LLM-based entity resolution during the merging phase.
- Test coverage: Minimal; relies on manual inspection of `entities.json`.

**3-Pass Prompt Dependency:**
- Files: `src/kg_extractor.py`
- Why fragile: The quality of the KG depends heavily on the model's ability to follow complex Chinese prompts and maintain strict JSON structure across three separate passes.
- Safe modification: Use few-shot examples in prompts or fine-tuned models for extraction tasks.

## Scaling Limits

**Local LLM Throughput:**
- Current capacity: Sequential processing at ~5-15s per chunk.
- Limit: Becomes impractical for datasets larger than a single book (e.g., the full "Huangdi Neijing" collection).
- Scaling path: Cloud-based LLM providers or multi-GPU local setups for parallel extraction.

## Test Coverage Gaps

**Integration & E2E Testing:**
- What's not tested: Full pipeline from ingest to retrieval is only verified via manual scripts (`scripts/e2e_test.py`).
- Files: `src/main.py`, `src/retriever.py`, `src/api.py`
- Risk: Changes to the retrieval logic or prompt templates may introduce regressions in answer quality or citation accuracy without immediate detection.
- Priority: High

**KG Extraction Quality:**
- What's not tested: No automated benchmark for KG extraction accuracy (Recall/Precision).
- Files: `src/kg_extractor.py`
- Risk: Prompt adjustments might unknowingly degrade the quality of extracted TCM relationships.
- Priority: Medium

---

*Concerns audit: 2025-02-18*
