# Phase 2: Data Pipeline Overhaul - Implementation Plan

Replace incomplete source data and manual KG with complete TCM sources and LLM-extracted knowledge graph.

## User Review Required

> [!IMPORTANT]
> **Breaking change**: Vector store has been rebuilt. Existing queries may return different results.

> [!CAUTION]
> **Schema change**: Entity types expanded from 4 to 7 (WHO-aligned). KG extraction uses 3-pass approach with self-critique.

---

## Proposed Changes

### Data Migration ✅ COMPLETE

- Archived old files to `data/archive/`
- Copied 3 source texts (~1.7MB) to `data/source/`
- Rebuilt vector store: **2,400 chunks** generated
- Added `temp_tcm_books/` to `.gitignore`

---

### Multi-Source Ingestion ✅ COMPLETE

#### [MODIFY] [ingest.py](file:///d:/Dev/TCM-Sage/src/ingest.py)

Rewritten with:
- `extract_book_name()` - Parse book names from filenames
- `ingest_all_sources()` - Process all `.txt` files
- `process_single_source()` - Character offset tracking
- Metadata: `{book, source, char_start, char_end}`

---

### LLM-Based KG Extraction (PENDING)

#### [MODIFY] [kg_extractor.py](file:///d:/Dev/TCM-Sage/src/kg_extractor.py)

Rewrite with 3-pass architecture:

```
Pass 1: Entity Extraction
├── Input: text chunk + metadata
├── Output: entities[] with evidence spans
└── Model: qwen3:8b (4b fallback)

Pass 2: Relation Extraction
├── Input: text chunk + entities[]
├── Output: relations[] (head/tail must be from entities)
└── Constraint: Only use entity IDs from Pass 1

Pass 3: Self-Critique
├── Input: text chunk + entities[] + relations[]
├── Output: verified KG with confidence scores
└── Task: Mark unsupported facts, add confidence
```

**Entity Types (7, WHO-aligned):**
- `Symptom`, `Pattern`, `Herb`, `Formula`, `TreatmentMethod`, `Meridian`, `Acupoint`

**Relationship Types (6):**
- `TREATS`, `CONTAINS`, `INDICATES`, `APPLIES_TO`, `LOCATED_ON`, `ASSOCIATED_WITH`

**Key Implementation Details:**
- Use `/no_think` suffix for stable structured output
- Temperature: 0.0-0.3 for extraction
- Require `evidence` field with exact text spans
- JSON Schema validation for output

---

### Documentation Updates ✅ COMPLETE

#### [MODIFY] [README.md](file:///d:/Dev/TCM-Sage/README.md)

Added data source credits to xiaopangxia/TCM-Ancient-Books.

---

## Verification Plan

### Prerequisites

```bash
# Ensure Ollama is running with qwen3
ollama pull qwen3:8b  # or qwen3:4b for lower VRAM
```

### Automated Tests

1. **Ingestion Test** ✅ PASSED
   ```bash
   python src/ingest.py
   # Expected: 2,400 chunks generated
   ```

2. **KG Extraction Sample Test** (PENDING)
   ```bash
   # Test extraction on 10 sample chunks
   python -c "
   from src.kg_extractor import extract_kg_batch
   import json
   chunks = json.load(open('data/processed/chunks.json'))[:10]
   result = extract_kg_batch(chunks, model='qwen3:8b', limit=10)
   print(f'Entities: {len(result[\"entities\"])}, Relations: {len(result[\"relationships\"])}')
   "
   ```

3. **Self-Critique Validation** (PENDING)
   - After extraction, verify `confidence` scores are populated
   - Check that low-confidence facts have `unsupported=True`

### Manual Verification

1. **Spot-check extracted KG**: Review 10-20 entities for accuracy
2. **E2E Query Test**: Ask "川芎治什么?" and verify citations link to source
3. **针灸 Query Test**: Ask about meridians/acupoints to verify coverage
