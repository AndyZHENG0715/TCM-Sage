# TCM-Sage Enhancement Walkthrough

**Session**: 2026-01-14 to 2026-01-18
**Focus**: Inline Citations & KG Pipeline Overhaul

---

## Original Goal

Make TCM-Sage a credible RAG system where:
1. Answers include **inline citations** `[1]`, `[2]` linking to source passages
2. KG facts display **correct relationship direction** (川芎 → 頭痛)
3. Every fact is **traceable** to exact character range in 黄帝内经

---

## Phase 1: Inline Citation System ✅ COMPLETE

### What Was Done
- Created `src/citation_types.py` with TypedDict schemas
- Added `format_docs_with_citations()` in `main.py`
- Added `CITATION_INSTRUCTION` for LLM prompts
- Updated `ui_backend.py` to return `citations` array
- Fixed KG direction in `retriever.py` (`_format_graph_fact`)
- Added `src/test_citations.py` (6/6 tests pass)

### Key Files Modified
| File | Changes |
|------|---------|
| `src/main.py` | Added `format_docs_with_citations()`, `CITATION_INSTRUCTION` |
| `src/ui_backend.py` | Returns `citations` in both `run_query()` and `run_query_stream()` |
| `src/retriever.py` | Fixed `_format_graph_fact` direction |
| `src/citation_types.py` | New: TypedDict schemas |
| `src/test_citations.py` | New: Unit tests |

### Deferred
- §1 Metadata enhancement (needs re-ingestion)
- §3.3-3.4 Citation validation
- §5 Frontend clickable citations
- §6.2-6.4 Integration tests

---

## Phase 2: Data Pipeline Overhaul ✅ CORE COMPLETE

### What Was Done

**Data Migration**:
- Archived old `huangdi_neijing.txt` and `entities.json` to `data/archive/`
- Added 3 complete 黄帝内经 volumes from TCM-Ancient-Books:
  - `437-黄帝内经素问.txt` (237KB)
  - `431-黄帝内经灵枢集注.txt` (601KB)
  - `439-黄帝内经太素.txt` (870KB)

**Ingestion**:
- Updated `src/ingest.py` for multi-source with `book`, `chapter`, `char_start`, `char_end`
- Generated **2,027 chunks** in `data/processed/chunks.json`

**KG Extractor**:
- Created `src/kg_extractor.py` with 3-pass architecture:
  - Pass 1: Entity extraction with `/nothink`
  - Pass 2: Relation extraction
  - Pass 3: Self-critique with confidence scores
- 9 entity types: Symptom, Pattern, Herb, Formula, TreatmentMethod, Meridian, Acupoint, BodyPart, Substance
- 10 relationship types including TREATS, LOCATED_ON, FLOWS_THROUGH, etc.

**Bug Fixes**:
- Fixed `graph_builder.py` `load_from_json` to use 'mention' as name
- Fixed `retriever.py` `source_type` metadata
- Created `src/extract_kg_durable.py` for graceful interrupt handling

### Key Files
| File | Purpose |
|------|---------|
| `src/ingest.py` | Multi-source ingestion with provenance |
| `src/kg_extractor.py` | LLM-based KG extraction (3-pass) |
| `src/extract_kg_durable.py` | Durable extraction with incremental saving |
| `src/graph_builder.py` | Fixed entity name loading |

---

## KG Extraction Status ⏸️ 22% COMPLETE

| Book | Chunks | Entities | Status |
|------|--------|----------|--------|
| 黄帝内经灵枢集注 | 454 | 1,626 | ✅ Complete |
| 黄帝内经素问 | 298 | 0 | ⏸️ Pending |
| 黄帝内经太素 | 1,275 | 0 | ⏸️ Pending |
| **Total** | **2,027** | **1,626** | **22%** |

**Resume**: `python src/extract_kg_durable.py`

---

## Decisions Made

1. **Two-phase approach** - Phase 1 for citations, Phase 2 for data
2. **Local Ollama** with `qwen3:8b` for KG extraction (not API)
3. **WHO-aligned schema** - 9 entity types, 10 relationship types
4. **Durable extraction** - saves every 5 chunks, graceful Ctrl+C

---

## Next Steps

1. Resume KG extraction for 素问 and 太素
2. E2E test: Query "川芎治什么?" with citations
3. Optional: Frontend clickable citations
