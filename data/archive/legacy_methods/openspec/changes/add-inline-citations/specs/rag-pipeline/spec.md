## ADDED Requirements

### Requirement: Inline Citation Generation
The system SHALL generate inline citations in `[n]` format within LLM responses, where `n` corresponds to a numbered source from the retrieved context.

#### Scenario: Basic inline citation
- **WHEN** the LLM generates an answer using retrieved context
- **THEN** claims derived from specific sources MUST include inline citations like `[1]`, `[2]`
- **AND** each citation number MUST map to a valid source in the citation map

#### Scenario: Multiple citations in one sentence
- **WHEN** a claim is supported by multiple sources
- **THEN** the LLM MAY include multiple citations like `[1][2]` or `[1, 2]`

#### Scenario: No relevant sources
- **WHEN** no sources directly support a claim
- **THEN** the system MUST NOT invent citation numbers
- **AND** the claim SHOULD be marked as unsupported or omitted

---

### Requirement: Citation Map Response Structure
The system SHALL return a structured `citations` array in the API response, alongside the `answer`.

#### Scenario: Standard response with citations
- **WHEN** a query is processed successfully
- **THEN** the response MUST include:
  - `answer`: The generated text with inline `[n]` citations
  - `citations`: An array of citation objects
- **AND** each citation object MUST contain:
  - `number`: The citation index (1-based)
  - `source`: The chapter/source title
  - `content`: The full text of the cited chunk
  - `chunk_id`: The unique identifier of the chunk
  - `score`: The retrieval relevance score (if available)

#### Scenario: Knowledge graph citation
- **WHEN** a citation references a knowledge graph fact
- **THEN** the citation MUST include:
  - `type`: "graph" (to distinguish from vector citations)
  - `fact`: The formatted KG relationship string
  - `depth`: The graph traversal depth

---

### Requirement: Citation-Aware Context Formatting
The system SHALL format retrieved documents with numbered citations for LLM consumption.

#### Scenario: Vector document formatting
- **WHEN** retrieved documents are formatted for the LLM prompt
- **THEN** each document MUST be prefixed with a citation number
- **AND** the format MUST be: `[n] Source: {chapter}\n{content}`

#### Scenario: Mixed vector and graph context
- **WHEN** both vector and graph results are included
- **THEN** vector citations MUST be numbered first
- **AND** graph citations MUST follow sequentially as `[KG-n]` or continue numbering

---

### Requirement: Citation Validation in Self-Critique
The self-critique verification step SHALL validate that citation numbers reference valid sources.

#### Scenario: Valid citations
- **WHEN** all `[n]` citations in the answer map to provided sources
- **THEN** the verification result MUST be SUPPORTED (assuming content is faithful)

#### Scenario: Invalid citation number
- **WHEN** the answer contains a citation `[n]` where `n` exceeds the number of provided sources
- **THEN** the system MUST flag the response as UNSUPPORTED
- **AND** the verification result SHOULD note "Invalid citation: [n]"

---

### Requirement: Enhanced Chunk Metadata
The ingestion pipeline SHALL store enhanced metadata for each chunk to support citation rendering.

#### Scenario: Chunk ingestion
- **WHEN** a new chunk is created during ingestion
- **THEN** the chunk metadata MUST include:
  - `source`: Chapter title (existing)
  - `chunk_id`: Unique identifier (existing as `id`)
- **AND** the chunk metadata MAY include:
  - `chunk_index`: Sequential index within the corpus (optional)
  - `line_start`: Starting line number in original text (optional)
  - `line_end`: Ending line number in original text (optional)
