# Change: Add Inline Citation System

## Why

TCM-Sage currently provides source citations only as a "Sources:" section at the end of LLM responses. Users cannot easily trace which claims come from which sources. Modern AI assistants (Perplexity, academic papers) use inline citations like [1], [2] that map to specific sources, improving trust and verifiability.

## What Changes

### Backend (RAG Pipeline)
- **Enhance chunk metadata** during ingestion to include richer citation data (chunk_id, chapter title, line ranges)
- **Modify LLM prompt** to instruct citation generation with `[n]` format referencing numbered sources
- **Return structured citation mapping** in API response alongside the answer, enabling frontend rendering

### Frontend (Future Phase)
- Render inline citations as clickable elements
- Display citation panel or modal when clicked
- Show full source context with highlighting

### Key Design Decisions
1. **Citation numbering**: Dynamic per-response (1-indexed based on retrieved context order)
2. **Metadata enhancement**: Minimal overhead—retain existing chunk structure, add optional fields
3. **LLM prompt update**: Explicit instruction to cite using `[n]` format, only citing passages provided
4. **Hallucination guard**: If LLM cites a number outside retrieved context, verification step flags it

## Impact

- **Affected specs**: `rag-pipeline` (new capability)
- **Affected code**:
  - `src/ingest.py` – Add `chunk_index` and optional line range metadata
  - `src/main.py` – Update `format_docs()` to produce numbered, referenceable context; update prompts
  - `src/ui_backend.py` – Return `citations` array in response alongside `answer`
  - `src/ui_app.py` – (Future) Render clickable citations in Streamlit UI
