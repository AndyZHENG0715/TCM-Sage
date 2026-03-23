# Tasks: Add Inline Citation System

> **Status (2026-01-18)**: Core implementation complete. KG source_ref propagation added. Section 1 (chunk_index metadata) complete. Section 5 is future phase.

## 1. Metadata Enhancement (Ingestion) ✅

- [x] 1.1 Update `ingest.py` to include `chunk_index` (1-based) in chunk metadata
- [ ] 1.2 Add optional `line_start` and `line_end` fields if extractable from source
- [x] 1.3 Re-ingest corpus to populate new metadata fields (2027 chunks)
- [x] 1.4 Verify ChromaDB stores and returns new metadata

## 2. Citation-Aware Context Formatting ✅

- [x] 2.1 Create `format_docs_with_citations()` function in `main.py`
- [x] 2.2 Generate numbered citation references `[1]`, `[2]`, etc., in context
- [x] 2.3 Build `citation_map` dictionary mapping numbers to full source metadata
- [x] 2.4 ~~Update existing `format_docs()`~~ Created new function, kept original for backward compat
- [x] 2.5 Pass KG `source_ref` provenance from graph_builder → retriever → main *(added 2026-01-18)*

## 3. Prompt Engineering for Citation Generation

- [x] 3.1 Update system prompt to instruct LLM to use `[n]` inline citations
- [x] 3.2 Include citation format guidelines in prompt (cite only provided sources)
- [x] 3.3 Add verification check for out-of-range citation numbers → `verify_citation_bounds()`
- [ ] 3.4 Test with various query types to ensure consistent citation format

## 4. API Response Enhancement ✅

- [x] 4.1 Update `run_query()` return dict to include `citations` array
- [x] 4.2 Update `run_query_stream()` metadata payload with citations
- [x] 4.3 Document citation response structure in docstrings

## 5. Frontend Integration (Future Phase)

> ⏸️ **DEFERRED**: Explicitly marked as future phase in design doc.

- [ ] 5.1 Parse inline citations from answer text using regex
- [ ] 5.2 Render citations as clickable elements
- [ ] 5.3 Implement citation panel/modal showing full source text
- [ ] 5.4 Add visual highlighting for cited passages

## 6. Testing & Validation

- [x] 6.1 Unit tests for `format_docs_with_citations()` → `src/test_citations.py` (9 tests)
- [x] 6.2 Unit tests for `verify_citation_bounds()` *(added 2026-01-18)*
- [x] 6.3 Unit test for graph citation `source_ref` extraction *(added 2026-01-18)*
- [ ] 6.4 Integration tests verifying citation map accuracy
- [ ] 6.5 Manual testing with sample queries
- [ ] 6.6 Verify self-critique handles citation out-of-bounds

---

## Implementation Notes (Session 2026-01-14)

### Files Created
- `src/citation_types.py` – TypedDict schemas (`TextCitation`, `GraphCitation`, `Citation`)
- `src/test_citations.py` – 6 unit tests, all passing

### Files Modified
- `src/main.py` – Added `CITATION_INSTRUCTION`, `format_docs_with_citations()`
- `src/ui_backend.py` – Updated `run_query()`, `run_query_stream()` to return `citations`
- `src/retriever.py` – **Fixed KG direction bug** (now shows `川芎 --TREATS--> 頭痛`)

---

## Implementation Notes (Session 2026-01-18)

### KG Source Provenance Pipeline
Fixed full `source_ref` propagation chain:

1. **`graph_builder.py`** – `load_from_json()` now stores `source_ref` on edges
2. **`graph_builder.py`** – `get_related_entities()` includes `source_ref` in relationship dict
3. **`retriever.py`** – Graph Documents include `source_ref` in metadata
4. **`main.py`** – `format_docs_with_citations()` extracts `source_ref` for graph citations

### Citation Bounds Verification
Added `verify_citation_bounds(answer, max_citation)` function that:
- Scans answer for `[n]` citation markers
- Returns `is_valid`, `out_of_range`, `found_citations`
- Helps detect hallucinated citations

### Test Results
```
✅ test_citations.py         - 9/9 passed
✅ test_graph.py             - 8/8 passed (1 skipped - no data)
```

### API Response Structure (Updated)
```python
{
    "answer": "根據內經 [1]...",
    "citations": [
        {"number": 1, "type": "text", "source": "素問·陰陽應象大論", ...},
        {"number": 2, "type": "graph", "fact": "營氣 --FLOWS_THROUGH--> 脈", 
         "source_ref": {"book": "黄帝内经灵枢集注", "chapter": "营卫生会篇第十八", ...}}
    ]
}
```

