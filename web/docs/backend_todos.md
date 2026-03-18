# Backend TODOs for Frontend Features

These items require backend changes before the frontend can fully support them.
Drafted during frontend bug-fix round on 2026-02-17.

---

## 1. Propagate Retrieval Relevance Scores

**Status**: ✅ **DONE** (Verified 2026-02-17)

**Details**: Backend already sends `score` (L2 distance) in citation metadata.
Frontend now converts this to a similarity percentage for display.

---

## 2. Full-Text Context Endpoint

**Current state**: The citation content is truncated to 100 characters by
`SNIPPET_LENGTH` in `format_docs_with_citations()`.

**Frontend impact**: The Citation Panel shows only a short snippet. The
"View Full Context" button is disabled (Coming Soon).

**Suggested fix**:
- Create a `GET /source/{chunk_id}/context` endpoint that, given a `chunk_id`,
  returns the full chapter or surrounding chunks from the vector store.
- The response should include:
  - `full_text`: the complete chapter/section text
  - `highlighted_range`: `{ start, end }` character offsets of the cited chunk
    within the full text
  - `chapter_title`: cleaned chapter name
- Optionally increase `SNIPPET_LENGTH` in `format_docs_with_citations()` to
  send more content in the streaming metadata.

---

## 3. Knowledge Graph Visualization Endpoint

**Current state**: Graph citations show structured metadata (fact, depth,
provenance) but no visual graph. The "View Graph" button is disabled.

**Frontend impact**: Users cannot explore the knowledge graph interactively.

**Suggested fix**:
- Create a `GET /graph/neighbors/{entity}` endpoint that returns:
  - `nodes`: list of `{ id, label, type }` (e.g., herbs, symptoms, meridians)
  - `edges`: list of `{ source, target, relationship, source_ref }`
  - `center`: the queried entity
- The frontend can then render this with a graph visualization library
  (e.g., `react-force-graph`, `d3-force`, or `cytoscape.js`).

---

## Priority

| # | Item                      | Complexity | Frontend Blocked? |
|---|---------------------------|------------|-------------------|
| 1 | Relevance Scores          | Low        | Cosmetic only     |
| 2 | Full-Text Context         | Medium     | Yes (button)      |
| 3 | Graph Visualization       | High       | Yes (button)      |
