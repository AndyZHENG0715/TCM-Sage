# 2-03 Real SymMap Validation Report

Date: 2026-03-29
Scope: Real SymMap v2.0 ingestion, KG trigger validation, checkpoint readiness.

## 1) Verdict

- The core migration path to real SymMap data is working.
- Your latest test ("头痛" only) returning many KG facts is considered a pass for KG trigger behavior.
- "Long prompt did not trigger KG" is most likely a query-entity matching issue, not a graph import failure.

## 2) What was completed

- Downloaded official SymMap v2.0 component files into `data/graph/symmap/raw`.
- Fetched herb->TCM symptom relationships and generated:
  - `data/graph/symmap/raw/rel_smhb_smts.tsv`
- Rebuilt graph JSON:
  - `data/graph/symmap/symmap_entities.json`
- Fixed ID normalization in importer so relationship endpoints match entity IDs:
  - `scripts/import_symmap_kg.py`
- Updated retrieval verification script for real SymMap IDs:
  - `scripts/verify_symmap_retrieval.py`

## 3) Evidence snapshot

- Import output: `18450 entities`, `21476 relationships`.
- Edge integrity check: `21476 / 21476` valid endpoints (no broken edges).
- Retrieval verification:
  - `_search_graph_documents("頭痛", ...)` returns graph documents.
  - `_search_graph_documents("头痛", ...)` returns graph documents.

## 4) Why long prompts may not trigger KG

Current KG retrieval in `src/graph_builder.py` + `src/ui_backend.py` is lexical/entity-name based:

- It works best when the query contains a graph entity term directly (for example `头痛` or `頭痛`).
- If a long prompt uses paraphrase/synonym not present in graph labels, matching can fail.
- This is expected with the current matcher design and does not indicate ingestion failure.

## 5) Test recommendation (human verify)

Use these in UI with Hybrid Retrieval ON:

1. `头痛` (or `頭痛`) -> should produce graph citations.
2. Long prompt that still includes the literal token `头痛` -> should also produce graph citations.
3. A synonym-only prompt (for example not containing `头痛`) -> may fail today; this is a known matcher limitation.

If 1 and 2 pass, KG trigger checkpoint for 2-03 can be considered functionally passed.

## 6) Human review outcome (L0 seed decision)

- Reviewed candidates: 37
- Approved: 33 (written to `data/graph/crosswalk/seed_crosswalk_approved.csv`)
- Rejected: 4 (kept in `seed_crosswalk_pending.csv` with rationale)
  - `symptom::心痛` (context-sensitive term; `心中痛` vs `心下痛`)
  - `symptom::寒热` (pattern/framework concept, not stable standalone symptom)
  - `symptom::积聚` (disease-name axis mismatch; should route to Disease/Pattern typing)
  - `symptom::癫狂` (clinical usage separates 癫 and 狂)

Decision basis: domain expert feedback (human-in-the-loop checkpoint evidence).

## 7) GSD workflow note

For GSD compliance:

- Keep this report plus crosswalk CSV artifacts as execution evidence for 2-03.
- Mark the human checkpoint as passed once reviewer sign-off is recorded.
- Proceed to next phase only with the approved seed table as L0 input.
