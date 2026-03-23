# Change: Integrate Knowledge Graph for Hybrid Retrieval

## Why

The current RAG system relies solely on vector-based semantic search, which struggles with resolving the ambiguity of classical TCM terminology. A hybrid retriever combining vector search with a knowledge graph will enable structured entity traversal (e.g., "Headache" → `TREATS` → "Herbs"), improving result precision and explainability for TCM practitioners.

## What Changes

- **Add** `networkx` dependency for in-memory graph storage
- **Add** `src/graph_builder.py` to extract TCM entities (Symptom, Herb, Formula) and build relationships
- **Add** `src/retriever.py` to encapsulate hybrid retrieval logic (vector + graph)
- **Modify** `src/main.py` and `src/ui_backend.py` to use the new hybrid retriever
- **Add** `data/graph/` directory for serialized graph data

## Impact

- **Affected specs**: New capability `retrieval/graph`
- **Affected code**:
  - `src/graph_builder.py` (new)
  - `src/retriever.py` (new)
  - `src/main.py` (integration)
  - `src/ui_backend.py` (integration)
  - `requirements.txt` (new dependency)
- **Phase alignment**: Implements Phase 3 "Hybrid Retriever" from project roadmap
