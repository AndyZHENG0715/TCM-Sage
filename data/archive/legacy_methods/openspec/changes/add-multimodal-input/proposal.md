# Change: Add Multimodal Input Support

## Why

TCM diagnosis traditionally relies on 四诊 (Four Examinations): 望 (inspection), 闻 (listening/smelling), 问 (inquiry), and 切 (palpation). Currently, TCM-Sage only supports text-based inquiry (问). Adding image upload enables **visual inspection (望诊)**, particularly:

- **Tongue diagnosis (舌诊)**: Analyzing tongue color, coating, shape, and moisture
- **Face diagnosis (面诊)**: Observing facial complexion and features
- **Skin/lesion photos**: Visual symptoms the patient wants to describe

This makes TCM-Sage more useful for practitioners who want AI assistance with visual diagnostic clues.

## What Changes

### Frontend
- Add file upload button (📎) in input area
- Support drag-and-drop for images
- Show thumbnail preview before sending
- Display uploaded images in chat history

### API
- Modify `/query` endpoint to accept `multipart/form-data`
- Support image attachments alongside text query
- Return image analysis in response when applicable

### Backend
- Integrate vision-capable LLM (GPT-4o, Gemini 2.0 Flash, or Qwen-VL)
- Add tongue/face analysis prompt templates
- Combine visual analysis with RAG text retrieval

### Key Design Decisions

1. **LLM Provider**: Use Gemini 2.0 Flash (free tier) or GPT-4o-mini for vision
2. **File types**: Initially support JPEG, PNG, WebP (common photo formats)
3. **Size limits**: Max 10MB per image, max 3 images per query
4. **Privacy**: Images are not stored permanently—processed and discarded
5. **Fallback**: If vision LLM fails, return text-only response with warning

## Impact

### Affected Files
- `src/api.py` – Add multipart handling to `/query`
- `src/main.py` – Add vision LLM integration
- `web/components/ChatInput.tsx` – File upload UI
- `web/components/MessageBubble.tsx` – Display images in messages
- `web/lib/api.ts` – FormData submission

### New Dependencies
- `python-multipart` – FastAPI file upload handling
- Vision-capable LLM SDK (already have langchain integrations)

## Open Questions

1. **Storage**: Should we store uploaded images for debugging/improvement?
2. **Consent**: Do we need explicit user consent for image analysis?
3. **Disclaimer**: Should we add medical disclaimer specifically for visual diagnosis?
