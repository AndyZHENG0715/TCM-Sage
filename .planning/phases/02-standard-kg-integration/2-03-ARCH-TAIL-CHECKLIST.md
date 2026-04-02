# 2-03 Architecture Tail Checklist (SymMap-only + Bridge)

Date: 2026-04-02
Status: In progress

## A) Completed

- [x] Runtime KG source of truth is SymMap (`GRAPH_DATA_PATH` -> `data/graph/symmap/symmap_entities.json`).
- [x] Removed API/UI runtime fallback to legacy `entities.json` / `entities_partial.json`.
- [x] Added query-time bridge module: `src/crosswalk_bridge.py`.
- [x] Wired bridge into both retrieval paths:
  - `src/ui_backend.py` (`_search_graph_documents`)
  - `src/retriever.py` (`graph_search`)

## B) Bridge matching implementation (current)

`src/crosswalk_bridge.py` currently uses **rule-based exact/substring matching**, not tokenization or NLP:

1. Load `data/graph/crosswalk/seed_crosswalk_approved.csv` (path override via `CROSSWALK_APPROVED_PATH`).
2. For each row, keep:
   - `canonical_symmap_id`
   - `neijing_name`
   - `normalized_name`
3. Normalize query with NFKC + lowercase, keep alnum + CJK.
4. Match rules:
   - raw substring: `neijing_name in query`
   - normalized substring: `normalized_name in normalized_query`
5. Return a set of candidate SymMap IDs; retrieval unions these IDs with normal graph name search.

## C) What it is NOT doing (important)

- No tokenization / segmentation (no jieba path in bridge matching).
- No synonym expansion/alias dictionary.
- No fuzzy matching / edit distance.
- No embedding semantic mapping.
- No context-aware disambiguation.

## D) Known limitations (for defense Q&A)

- Misses paraphrases where the canonical term is not a contiguous substring.
  - Example: query `我头很痛` may not match approved term `头痛`.
- Sensitive to lexical surface form; robust for exact term mentions, weaker for colloquial variants.
- Potential false positives for very short terms due to substring strategy.
- Crosswalk quality depends on approved CSV coverage.

## E) Recommended next hardening (not in current scope)

- Add synonym lexicon for high-frequency symptom variants (e.g., colloquial -> canonical).
- Add lightweight tokenization/phrase rewrite before bridge lookup.
- Add bounded fuzzy fallback only for high-confidence candidates.
- Add bridge hit-rate telemetry (query -> matched IDs count) for evaluation.

## F) Legacy scripts still tied to `entities_partial.json` (candidate archive/deprecate list)

- `scripts/check_lingshu.py`
- `scripts/spot_check.py`
- `scripts/quality_check.py`
- `scripts/e2e_test.py`
- `scripts/diagnose_graph.py`
- `scripts/deep_investigation.py`
- `scripts/comprehensive_audit.py`
- `scripts/check_status.py`
- `scripts/check_results.py`
- `src/extract_kg_durable.py`
- `src/extract_kg_subset.py`
- `src/kg_extractor.py`

Decision rule: keep as archival/debug tools only, not runtime dependencies for the SymMap-only architecture.
