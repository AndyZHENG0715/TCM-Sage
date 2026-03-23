# Design: Inline Citation System

## Context

TCM-Sage is a RAG-based evidence-synthesis tool for Traditional Chinese Medicine. Currently, citations appear only at the end of responses in a "Sources:" section. This limits traceability—users cannot easily verify which claims come from which source passages.

**Stakeholders**: TCM practitioners, researchers, system maintainers
**Timeline**: Phase 3-5 development (Jan-Apr 2026)

## Goals / Non-Goals

### Goals
- Enable inline citations like `[1]`, `[2]` in LLM responses
- Map each citation number to specific source metadata (chapter, content snippet)
- Support clickable citations in frontend (future)
- Maintain existing response quality and latency

### Non-Goals
- Full-text search within citations (out of scope)
- Citation editing or curation by users
- Multi-document synthesis across different classical texts (future)

## Current State Analysis

### Existing Chunk Metadata
```json
{
  "id": "chunk_42",
  "content": "帝曰：诸痈肿筋挛骨痛...",
  "metadata": {
    "source": "脉要精微论篇第十七"
  }
}
```

### Current `format_docs()` Output
```
=== Text Passages ===
--- Source: 脉要精微论篇第十七 ---
帝曰：诸痈肿筋挛骨痛...

=== References (Debug) ===
1. [Vec: 0.451] 脉要精微论篇第十七: "帝曰：诸痈肿筋挛骨痛..."
```

The debug references already use numbered format—we can leverage this for citation mapping.

## Decisions

### Decision 1: Dynamic Citation Numbering
**What**: Citation numbers are generated per-response based on retrieval order (1-indexed)
**Why**: Avoids reliance on global chunk IDs; simpler for LLM to follow; supports deduplication
**Alternatives**: 
- Use chunk_id directly (rejected: "chunk_42" is not user-friendly)
- Use source chapter as citation (rejected: multiple chunks from same chapter)

### Decision 2: Prompt-Based Citation Instruction
**What**: System prompt explicitly instructs LLM to use `[n]` format
**Example prompt addition**:
```
When citing information, use inline citations in the format [1], [2], etc.
Each number corresponds to the numbered sources provided in the context.
Only cite sources that are actually provided—do not invent citation numbers.
```
**Why**: LLMs follow explicit formatting instructions well; no model fine-tuning needed
**Alternatives**:
- Post-process answer to inject citations (rejected: loses semantic connection)
- Fine-tune model for citation (rejected: expensive, scope creep)

### Decision 3: Citation Map Structure
**What**: Return a `citations` array alongside the `answer`
```python
{
  "answer": "According to the Neijing [1], yin and yang are the fundamental...",
  "citations": [
    {
      "number": 1,
      "source": "阴阳应象大论篇第五",
      "content": "阴阳者，天地之道也，万物之纲纪...",
      "chunk_id": "chunk_12",
      "score": 0.892
    }
  ]
}
```
**Why**: Enables frontend to render clickable citations; decoupled from answer parsing

### Decision 4: Verification for Invalid Citations
**What**: Self-critique step checks if cited numbers exist in citation map
**How**: After answer generation, scan for `[n]` patterns and validate against map size
**Why**: Prevents hallucinated citations (e.g., `[7]` when only 5 sources provided)

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| LLM ignores citation instruction | Iterative prompt tuning; temperature routing |
| Citations clutter answer readability | Future: toggle citation visibility in UI |
| Performance overhead from citation parsing | Minimal regex scan; not a bottleneck |
| Re-ingestion required for metadata | One-time cost; script already exists |

## Migration Plan

1. **Phase A (Backend)**: Implement citation-aware `format_docs_with_citations()`, update prompts
2. **Phase B (API)**: Add citations to response payload
3. **Phase C (Frontend)**: Render clickable citations in Streamlit UI
4. **Rollback**: Feature flag `ENABLE_INLINE_CITATIONS` in `.env` (default: false initially)

## Open Questions

1. **Line ranges**: Should we track exact line numbers within original source text?
   - *Tentative*: Not for MVP; chapter-level granularity sufficient initially
   
2. **KG citations**: How should knowledge graph facts be cited?
   - *Tentative*: Use `[KG-n]` format to distinguish from text citations

3. **Citation style preference**: Academic `[1]` vs. superscript `¹` vs. (Author, Year)?
   - *Tentative*: Bracket format `[1]` for clarity and accessibility
