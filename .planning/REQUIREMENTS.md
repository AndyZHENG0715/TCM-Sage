# Requirements: FYP Stabilization & KG Pivot

**User Story:** As an FYP student, I need a stable, academically defensible RAG application with an interactive UI so that I can successfully present the project on April 13th without critical bugs or data validity questions.

## Functional Requirements
- [ ] **REQ-1 (KG Swap):** Replace `entities_partial.json` with a recognized TCM dataset (e.g., TCMID or SymMap).
- [ ] **REQ-2 (Graph Adapter):** Update `src/graph_builder.py` to ingest the new standard dataset format.
- [ ] **REQ-3 (Context UI):** Frontend must allow toggling between the 100-character snippet and the full paragraph for retrieved text citations.
- [ ] **REQ-4 (KG Viz):** Frontend must render graph citations visually using `@xyflow/react`.

## Non-Functional Requirements
- [ ] **NFR-1 (Markdown Fix):** Fix Markdown parsing bugs regarding Chinese quotes (prevent `**"text"**` rendering issues).
- [ ] **NFR-2 (Source Stripping):** Strip redundant LLM-generated "Sources:" lists from the bottom of chat bubbles.
- [ ] **NFR-3 (Encoding):** Ensure robust handling of Chinese character encoding across the pipeline.
