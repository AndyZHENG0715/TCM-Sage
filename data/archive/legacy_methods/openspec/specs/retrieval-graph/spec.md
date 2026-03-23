# retrieval-graph Specification

## Purpose
TBD - created by archiving change integrate-knowledge-graph. Update Purpose after archive.
## Requirements
### Requirement: Knowledge Graph Entity Model

The system SHALL maintain an in-memory knowledge graph with the following entity types:

- **Symptom**: Clinical symptoms and conditions (e.g., "頭痛", "失眠")
- **Herb**: Medicinal herbs (e.g., "川芎", "白芷")
- **Formula**: Classical prescriptions (e.g., "川芎茶調散")

The following relationship types SHALL be supported:

- **TREATS**: Links Herb/Formula to Symptom
- **CONTAINS**: Links Formula to Herb
- **ASSOCIATED_WITH**: Links Symptom to Symptom

#### Scenario: Load graph from JSON

- **WHEN** the system initializes with `HYBRID_RETRIEVAL_ENABLED=true`
- **THEN** the knowledge graph is loaded from `GRAPH_DATA_PATH`
- **AND** all entities and relationships are available for traversal

---

### Requirement: Graph Traversal

The system SHALL support entity traversal with configurable hop depth.

Given a query term, the system SHALL:

1. Identify matching entity nodes (exact or fuzzy match)
2. Traverse connected edges up to the specified depth
3. Return related entities with their relationship types

#### Scenario: Single-hop traversal for symptom query

- **GIVEN** a graph with Symptom "頭痛" connected via TREATS to Herb "川芎"
- **WHEN** user queries "頭痛"
- **THEN** the graph search returns Herb "川芎" with relationship TREATS

#### Scenario: Two-hop traversal

- **GIVEN** Symptom "頭痛" → TREATS → Herb "川芎" → CONTAINS → Formula "川芎茶調散"
- **WHEN** user queries "頭痛" with hop_depth=2
- **THEN** both "川芎" and "川芎茶調散" are returned

---

### Requirement: Ensemble Context Aggregation

The system SHALL retrieve results from both sources and combine them for the Generator:

- **Vector Retrieval**: Fetch top K semantic text chunks
- **Graph Retrieval**: Fetch top M related entities/facts based on traversal
- **Aggregation**: Return a result set that distinguishes between `source: vector` and `source: graph`

The downstream Prompt Template SHALL format these as distinct sections.

#### Scenario: Ensemble retrieval with source metadata

- **GIVEN** a query "頭痛"
- **WHEN** hybrid search is performed
- **THEN** vector_docs contain text chunks with `metadata.source = "vector"`
- **AND** graph_docs contain fact strings with `metadata.source = "graph"`
- **AND** combined result includes both sets

#### Scenario: Graph facts formatted as text

- **GIVEN** graph traversal finds entity "川芎" with relationship "TREATS" to "頭痛"
- **WHEN** graph results are formatted
- **THEN** output includes: "KG Fact: 川芎 TREATS 頭痛"

---

### Requirement: Feature Flag Control

The hybrid retrieval capability SHALL be controlled by a feature flag.

- When `HYBRID_RETRIEVAL_ENABLED=false` (default), use pure vector retrieval
- When `HYBRID_RETRIEVAL_ENABLED=true`, use hybrid retrieval

#### Scenario: Disabled hybrid retrieval

- **GIVEN** HYBRID_RETRIEVAL_ENABLED=false
- **WHEN** a query is submitted
- **THEN** only vector search is performed
- **AND** graph traversal is skipped

#### Scenario: Enabled hybrid retrieval

- **GIVEN** HYBRID_RETRIEVAL_ENABLED=true
- **WHEN** a query is submitted
- **THEN** both vector and graph searches are performed
- **AND** results are merged using the configured alpha weight

