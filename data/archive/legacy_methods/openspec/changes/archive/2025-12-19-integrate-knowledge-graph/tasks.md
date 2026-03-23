## 1. Setup & Dependencies

- [x] 1.1 Add `networkx` to `requirements.txt` *(already present)*
- [x] 1.2 Create `data/graph/` directory structure
- [x] 1.3 Create seed data file `data/graph/entities.json` with sample TCM entities

## 2. Graph Builder Module

- [x] 2.1 Create `src/graph_builder.py` with `TCMKnowledgeGraph` class
- [x] 2.2 Implement `load_from_json()` to parse entity data
- [x] 2.3 Implement `add_entity()` for nodes (Symptom, Herb, Formula)
- [x] 2.4 Implement `add_relationship()` for edges (TREATS, CONTAINS, ASSOCIATED_WITH)
- [x] 2.5 Implement `get_related_entities()` for traversal (configurable hop depth)
- [x] 2.6 Implement `save_graph()` / `load_graph()` for persistence

## 3. Hybrid Retriever Module

- [x] 3.1 Create `src/retriever.py` with `HybridRetriever` class
- [x] 3.2 Implement `vector_search()` wrapping existing ChromaDB logic
- [x] 3.3 Implement `graph_search()` for entity-based traversal
- [x] 3.4 Implement `hybrid_search()` with ensemble context aggregation

## 4. Integration

- [x] 4.1 Add environment variables to `.env.example`
- [x] 4.2 Update `src/main.py` to use `HybridRetriever` when enabled
- [x] 4.3 *(Skipped)* `ui_backend.py` inherits updated `format_docs` from main.py
- [x] 4.4 Ensure backward compatibility (pure vector when flag disabled)

## 5. Testing & Validation

- [x] 5.1 Create `src/test_graph.py` - 8/8 tests passed
- [x] 5.2 Create `src/test_hybrid_retriever.py` - 5/5 tests passed
- [x] 5.3 *(Deferred)* Update existing test_retriever.py
- [x] 5.4 Manual verification ready (set HYBRID_RETRIEVAL_ENABLED=true)

## 6. Documentation

- [x] 6.1 Update `docs/CONFIG.md` with hybrid retrieval configuration
- [x] 6.2 Update `README.md` with knowledge graph capabilities
