# Inline Citation System Implementation

**Session Date:** 2026-01-14  
**OpenSpec Change:** `add-inline-citations`

---

## Summary

Implemented inline citation system enabling LLM responses to reference sources using `[n]` notation, with structured citation metadata in API responses.

## Files Created

| File | Purpose |
|------|---------|
| `src/citation_types.py` | TypedDict schemas: `TextCitation`, `GraphCitation`, `Citation` |
| `src/test_citations.py` | 6 unit tests for citation formatting |

## Files Modified

### `src/main.py`
- Added `CITATION_INSTRUCTION` constant for LLM prompt
- Added `format_docs_with_citations(docs)` → returns `Tuple[str, List[dict]]`
- Original `format_docs()` preserved for backward compatibility

### `src/ui_backend.py`
- `run_query()` now returns `citations` array in response
- `run_query_stream()` final metadata includes `citations`
- Imports updated for new function

### `src/retriever.py`
- **Bug fix:** KG relationship direction corrected
- Before: `頭痛 --TREATS--> 川芎` (symptom treats herb ❌)
- After: `川芎 --TREATS--> 頭痛` (herb treats symptom ✅)

## API Response Structure

```python
{
    "answer": "根據內經 [1]，陰陽是天地之道...",
    "citations": [
        {
            "number": 1,
            "type": "text",
            "source": "素問·陰陽應象大論",
            "content": "陰陽者，天地之道也...",
            "chunk_id": "chunk_42",
            "score": 0.451
        },
        {
            "number": 2,
            "type": "graph",
            "fact": "川芎 --TREATS--> 頭痛",
            "depth": 1,
            "source_ref": null
        }
    ]
}
```

## Test Results

```
✅ test_citations.py         - 6/6 passed
✅ test_graph.py             - 8/8 passed
✅ test_hybrid_retriever.py  - 5/5 passed
```

Run tests with:
```powershell
& "D:\Dev\TCM-Sage\venv\Scripts\python.exe" src/test_citations.py
```

## OpenSpec Tasks Completed

- [x] 2.1-2.4 Citation-aware context formatting
- [x] 3.1-3.2 Prompt engineering for citations
- [x] 4.1-4.3 API response enhancement
- [x] 6.1 Unit tests

## Remaining Work

| Task | Status |
|------|--------|
| 1.1-1.4 Metadata enhancement | Deferred to `data-pipeline-overhaul` |
| 3.3 Out-of-range citation verification | Not started |
| 3.4 Query type testing | Not started |
| 5.1-5.4 Frontend integration | Future phase |
| 6.2-6.4 Integration/manual tests | Not started |

## Key Design Decisions

1. **New function vs modifying existing:** Created separate `format_docs_with_citations()` to maintain backward compatibility with CLI
2. **Module naming:** Originally named `types.py`, renamed to `citation_types.py` to avoid Python stdlib conflict
3. **Citation ordering:** Vector (text) citations numbered first, then graph citations
