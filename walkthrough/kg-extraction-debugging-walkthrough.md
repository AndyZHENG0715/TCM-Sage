# KG Extraction Debugging & Pipeline Fixes
**Session Date**: 2026-01-18

## Summary
This session focused on debugging KG extraction issues, verifying extraction status with multiple independent agents, and fixing the extraction pipeline.

---

## Key Findings

### KG Extraction Status (Verified by 4 independent agents)
| Book | Chunks | Entities | Status |
|------|--------|----------|--------|
| 黄帝内经灵枢集注 (Lingshu) | 454 | 1,626 | ✅ Complete |
| 黄帝内经素问 (Suwen) | 298 | 0 | ⏸️ Pending |
| 黄帝内经太素 (Taisu) | 1,275 | 0 | ⏸️ Pending |
| **Total** | **2,027** | **1,626** | **22%** |

> **Root cause discovered**: The extraction script was marking chunks as "processed" even when Ollama failed to extract entities. This caused false 100% completion reports.

---

## Changes Made

### 1. Fixed `src/extract_kg_durable.py`
**Problem**: Chunks marked as "processed" regardless of extraction success.

**Fix** (lines 124-131):
```python
# Only mark as processed if extraction produced entities
if new_entities or new_relations:
    processed_ids.add(chunk.get('id'))
```

### 2. Fixed `src/graph_builder.py`
**Problems**:
- `load_from_json()` failed to extract entity names from 'mention' field
- `search_by_name()` matched all entities when name was empty string

**Fixes**:
- Get `name = entity.get("mention")` before consuming the field
- Skip entities with empty names in search loop

### 3. Fixed `src/retriever.py`
**Problem**: `source_type` metadata was 'graph' but E2E test filtered for 'knowledge_graph'

**Fix**: Changed `source_type` to 'knowledge_graph' (line 170)

### 4. Reset Tracking File
Ran `scripts/reset_tracking.py` to keep only 454 Lingshu chunk IDs in `data/graph/processed_chunk_ids.json`. Suwen and Taisu are now ready for re-extraction.

---

## Scripts Created

| Script | Purpose |
|--------|---------|
| `scripts/check_status.py` | Quick check of extraction counts |
| `scripts/check_lingshu.py` | Verify Lingshu extraction completion |
| `scripts/comprehensive_audit.py` | Full audit of all extraction data |
| `scripts/deep_investigation.py` | Compare processed IDs vs chunks |
| `scripts/quality_check.py` | Check entity source distribution |
| `scripts/reset_tracking.py` | Reset processed IDs to Lingshu only |
| `scripts/diagnose_graph.py` | Debug graph search issues |
| `scripts/e2e_test.py` | E2E query test for RAG system |

---

## Current Data State

| File | Contents |
|------|----------|
| `data/graph/entities_partial.json` | 1,626 entities, 820 relationships |
| `data/graph/processed_chunk_ids.json` | 454 Lingshu IDs only |
| `data/processed/chunks.json` | 2,027 total chunks |

---

## To Resume Extraction
```bash
cd d:\Dev\TCM-Sage
python src/extract_kg_durable.py
```
This will extract Suwen (298 chunks) and Taisu (1,275 chunks).

---

## Next Steps
1. Resume extraction when Ollama performance allows
2. Consider using a smaller/faster model (qwen3:4b)
3. Monitor for Ollama timeouts during extraction
