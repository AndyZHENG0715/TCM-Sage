# Data Pipeline Overhaul - Tasks

## Data Migration ✅ COMPLETE

- [x] Create `data/archive/` directory
- [x] Move `data/source/huangdi_neijing.txt` to `data/archive/`
- [x] Move `data/graph/entities.json` to `data/archive/`
- [x] Copy source files from `temp_tcm_books/` to `data/source/`:
  - [x] `437-黄帝内经素问.txt`
  - [x] `431-黄帝内经灵枢集注.txt`
  - [x] `439-黄帝内经太素.txt`
- [x] Add `temp_tcm_books/` to `.gitignore`
- [x] Clear `vectorstore/chroma/` directory

---

## Multi-Source Ingestion ✅ COMPLETE

- [x] Modify `src/ingest.py`:
  - [x] Add `extract_book_name()` function
  - [x] Add `ingest_all_sources()` function
  - [x] Add `process_single_source()` with char offset tracking
  - [x] Include `book` field in metadata
- [x] Run ingestion: **2,027 chunks generated**

---

## LLM-Based KG Extraction ✅ IMPLEMENTED

### Schema Update ✅
- [x] 9 entity types: `Symptom`, `Pattern`, `Herb`, `Formula`, `TreatmentMethod`, `Meridian`, `Acupoint`, `BodyPart`, `Substance`
- [x] 10 relationship types including new 经络循行 types
- [x] Anti-hallucination rules

### 3-Pass Architecture ✅
- [x] Pass 1: Entity Extraction with `/nothink`
- [x] Pass 2: Relation Extraction
- [x] Pass 3: Self-Critique with confidence scores

### Bug Fix ✅
- [x] Fixed `extract_kg_durable.py` to only mark chunks as "processed" if entities are extracted

---

## KG Extraction Progress ⏸️ PARTIAL

| Book | Chunks | Entities | Status |
|------|--------|----------|--------|
| 黄帝内经灵枢集注 | 454 | 1,626 | ✅ Complete |
| 黄帝内经素问 | 298 | 0 | ⏸️ Pending |
| 黄帝内经太素 | 1,275 | 0 | ⏸️ Pending |
| **Total** | **2,027** | **1,626** | **22% complete** |

**Current Data**: 1,626 entities, 820 relationships (Lingshu only)
**Resume Command**: `python src/extract_kg_durable.py`
**Data Location**: `data/graph/entities_partial.json`

---

## Graph Search Fix ✅ COMPLETE

- [x] Fixed `load_from_json` to extract 'mention' as entity name
- [x] Fixed `search_by_name` to skip empty names
- [x] Fixed retriever `source_type` metadata
- [x] E2E test passes for available entities

---

## Documentation ✅ COMPLETE

- [x] Updated `README.md` with data source credits
- [x] Updated `proposal.md` with research findings
- [x] Updated this `tasks.md` with accurate status
