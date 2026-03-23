## Context

TCM-Sage currently uses a pure vector retrieval approach (ChromaDB + Sentence Transformers). Phase 3 of the project roadmap specifies implementing a "Hybrid Retriever" that combines vector search with a Knowledge Graph to resolve classical terminology ambiguity.

**Current architecture:**

- `src/main.py`: Inline `vectorstore.as_retriever()` calls
- `src/ui_backend.py`: Same pattern for UI pipeline
- No abstraction layer for retrieval

**Stakeholders:** TCM practitioners (end users), project maintainers

## Goals / Non-Goals

**Goals:**

- Implement lightweight in-memory knowledge graph using NetworkX
- Define schema: `Symptom`, `Herb`, `Formula` nodes with `TREATS`, `CONTAINS`, `ASSOCIATED_WITH` edges
- Create hybrid retriever that merges vector and graph results
- Maintain backward compatibility with existing CLI/UI

**Non-Goals:**

- Neo4j or external graph database (deferred to future phase)
- Automated entity extraction from raw text (manual/curated data for now)
- Graph visualization UI

## Decisions

### Decision 1: NetworkX for Graph Storage

**Choice:** Use `networkx` Python library for in-memory graph

**Rationale:**

- Lightweight, no external service dependencies
- Sufficient for prototype phase with manual entity data
- Easy migration path to Neo4j later (graph patterns remain similar)

**Alternatives considered:**

- Neo4j: Too heavy for current phase, requires infrastructure
- SQLite with adjacency tables: Less intuitive for graph traversal
- Custom dict-based graph: Reinventing the wheel

### Decision 2: Ensemble Context Strategy

**Choice:** Context Aggregation (Ensemble Retrieval)

**Rationale:**

- Vector search returns **Chunks** (text passages)
- Graph search returns **Entities** (facts/triplets)
- These are different data types that cannot be mathematically merged without entity linking
- Instead, retrieve both independently and append graph facts as a distinct section in LLM context

**Benefit:** Preserves the specific "reasoning path" found by the graph without diluting it with vector scores

**Alternatives considered:**

- Score fusion (RRF, weighted linear): Requires entity linking to map entities → chunks
- Re-ranking with LLM: Additional latency and cost

### Decision 3: Graph Data Source

**Choice:** Manual JSON file with curated TCM entities

**Rationale:**

- Ensures data quality for prototype
- Allows validation by domain experts
- Decoupled from ingestion pipeline (can be enhanced later)

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| Graph data maintenance burden | Start with small curated set (<100 entities); document format clearly |
| Performance impact of graph traversal | Limit to 2-hop traversal; benchmark before/after |
| Merge weight tuning | Make `α` configurable via `.env`; default to 0.7 (favor vector) |

## Migration Plan

1. Add new files without modifying existing retrieval
2. Add feature flag `HYBRID_RETRIEVAL_ENABLED` (default: `false`)
3. Integrate into main pipeline behind flag
4. Enable by default after validation

## Open Questions

1. **Entity data source**: Where should the initial graph data come from?
   - Option A: Manual JSON file in `data/graph/entities.json`
   - Option B: Extract from existing chunks using LLM
   - **Recommended**: Option A for controlled quality
