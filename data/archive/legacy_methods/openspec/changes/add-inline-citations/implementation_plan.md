# Phase 1: Inline Citation System

Backend implementation for inline citations `[1]`, `[2]` in LLM responses, with structured citation mapping in API responses.

## User Review Required

> [!IMPORTANT]
> **Relationship direction fix**: Current KG display shows `頭痛 --TREATS--> 川芎` (semantically backwards). Will fix to `川芎 --TREATS--> 頭痛`.

---

## Proposed Changes

### Context Formatting & Citation Map

#### [MODIFY] [main.py](file:///d:/Dev/TCM-Sage/src/main.py)

Replace `format_docs()` with `format_docs_with_citations()`:

```python
def format_docs_with_citations(docs: List[Document]) -> Tuple[str, List[Dict]]:
    """
    Format documents for LLM with numbered citations.
    
    Returns:
        Tuple of (formatted_context, citations_list)
    """
    citations = []
    context_parts = []
    
    for i, doc in enumerate(docs, start=1):
        source_type = doc.metadata.get('source_type', 'vector')
        
        if source_type == 'graph':
            # KG fact: source → relationship → target
            fact = _format_graph_fact_corrected(doc)  # Fixed direction
            context_parts.append(f"[{i}] Knowledge Graph: {fact}")
            citations.append({
                "number": i,
                "type": "graph",
                "fact": fact,
                "depth": doc.metadata.get('depth', 1),
                "source_ref": doc.metadata.get('source_ref')  # Provenance if available
            })
        else:
            source = doc.metadata.get('source', 'Unknown')
            context_parts.append(f"[{i}] Source: {source}\n{doc.page_content}")
            citations.append({
                "number": i,
                "type": "text",
                "source": source,
                "content": doc.page_content,
                "chunk_id": doc.metadata.get('id'),
                "score": doc.metadata.get('score', 0.0)
            })
    
    return "\n\n".join(context_parts), citations
```

Update system prompt to instruct inline citation:

```python
CITATION_INSTRUCTION = """
When answering, cite sources using [n] notation where n matches the source number provided.
Only cite sources that directly support your statements. Do not cite sources you were not given.
Example: "川芎 can treat headaches [1] and is often combined with 白芷 [2]."
"""
```

---

### API Response Enhancement

#### [MODIFY] [ui_backend.py](file:///d:/Dev/TCM-Sage/src/ui_backend.py)

Update `run_query()` and `run_query_stream()` return to include citations:

```python
# In run_query()
formatted_context, citations = format_docs_with_citations(retrieved_docs)

return {
    "question": user_query,
    "answer": answer,
    "citations": citations,  # NEW: structured citation data
    "severity": severity,
    # ... rest unchanged
}

# In run_query_stream() final metadata yield
yield {
    "type": "metadata",
    "citations": citations,  # NEW
    # ... rest unchanged
}
```

**TypedDict for type safety** (in new `src/types.py`):

```python
from typing import TypedDict, Optional, List, Literal

class TextCitation(TypedDict):
    number: int
    type: Literal["text"]
    source: str
    content: str
    chunk_id: str
    score: float

class GraphCitation(TypedDict):
    number: int
    type: Literal["graph"]
    fact: str
    depth: int
    source_ref: Optional[dict]  # {book, chapter, char_start, char_end}

Citation = TextCitation | GraphCitation
```

---

### KG Direction Fix

#### [MODIFY] [retriever.py](file:///d:/Dev/TCM-Sage/src/retriever.py)

Fix `_format_graph_fact()` to show correct semantic direction:

```diff
- # Current (wrong): symptom → TREATS → herb
+ # Fixed: herb → TREATS → symptom (medicine treats symptom)

def _format_graph_fact_corrected(doc: Document) -> str:
    """Format KG relationship with correct semantic direction."""
    rel_type = doc.metadata.get('relationship_type', '')
    source_name = doc.metadata.get('source_name', '')
    target_name = doc.metadata.get('target_name', '')
    
    # Relationship direction depends on type
    if rel_type == 'TREATS':
        # Medicine treats symptom (source is herb, target is symptom)
        return f"{source_name} --{rel_type}--> {target_name}"
    # ... handle other relationship types
```

---

## Verification Plan

### Existing Tests

Found existing test files:
- `src/test_hybrid_retriever.py` - Tests hybrid retriever
- `src/test_graph.py` - Tests graph operations  
- `src/test_retriever.py` - Tests retriever

### Automated Tests

```bash
# Run existing tests to ensure no regression
cd d:\Dev\TCM-Sage
python -m pytest src/test_*.py -v

# New test: test_citations.py
python -m pytest src/test_citations.py -v
```

New test file `src/test_citations.py` will cover:
1. `format_docs_with_citations()` returns correct structure
2. Citation numbers are sequential 1-indexed
3. Graph facts show correct relationship direction
4. API response includes `citations` array

### Manual Verification

1. Run Streamlit app: `streamlit run src/ui_app.py`
2. Ask: "什么中药可以治疗头痛?"
3. Verify response contains `[1]`, `[2]` inline citations
4. Check browser DevTools Network tab for API response containing `citations` array
5. Verify KG facts display as `川芎 --TREATS--> 頭痛` (not reversed)
