# Multimodal Input - Design

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Frontend (Next.js)                     │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  ChatInput                                          │    │
│  │  - Text input                                       │    │
│  │  - File upload (📎 button + drag-drop)              │    │
│  │  - Image preview thumbnails                         │    │
│  └─────────────────────────────────────────────────────┘    │
│                            │                                │
│                            ▼                                │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  API Client (FormData)                              │    │
│  │  POST /query { question: str, images: File[] }      │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    Backend (FastAPI)                        │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  /query endpoint                                    │    │
│  │  - Parse multipart/form-data                        │    │
│  │  - Validate images (type, size)                     │    │
│  │  - Base64 encode for LLM                            │    │
│  └─────────────────────────────────────────────────────┘    │
│                            │                                │
│              ┌─────────────┴─────────────┐                  │
│              ▼                           ▼                  │
│  ┌───────────────────┐       ┌───────────────────┐          │
│  │  Text RAG         │       │  Vision LLM       │          │
│  │  (existing)       │       │  (new)            │          │
│  │  - Retrieval      │       │  - Image analysis │          │
│  │  - Text LLM       │       │  - Tongue/face    │          │
│  └───────────────────┘       └───────────────────┘          │
│              │                           │                  │
│              └─────────────┬─────────────┘                  │
│                            ▼                                │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Response Combiner                                  │    │
│  │  - Merge visual observations with RAG citations     │    │
│  │  - Add disclaimer for visual diagnosis              │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## Vision LLM Selection

| Provider | Model | Cost | Pros | Cons |
|----------|-------|------|------|------|
| Google | Gemini 2.0 Flash | Free tier | Fast, good vision | Rate limits |
| OpenAI | GPT-4o-mini | $0.15/1M tokens | High quality | Paid |
| Alibaba | Qwen-VL | Free tier | Chinese optimized | Less common |

**Recommendation**: Start with Gemini 2.0 Flash (free), fallback to GPT-4o-mini.

## Prompt Template for Tongue Diagnosis

```
You are a Traditional Chinese Medicine diagnostic assistant analyzing a tongue image.

Observe and describe:
1. 舌质 (Tongue body): Color (淡白/淡红/红/绛红/紫), shape, moisture
2. 舌苔 (Tongue coating): Color (白/黄/灰黑), thickness, distribution
3. 舌形 (Tongue shape): Size, cracks, teeth marks, trembling

Based on TCM theory, suggest possible patterns (证型) this tongue may indicate.

IMPORTANT: This is for educational reference only. Always verify with a qualified practitioner.

Image: [attached]
User question: {question}
```

## Security Considerations

1. **No permanent storage**: Images processed in memory, not saved to disk
2. **Size limits**: 10MB max per image, 3 images max per request
3. **Type validation**: Only accept image/jpeg, image/png, image/webp
4. **Rate limiting**: Apply stricter limits for multimodal queries (more expensive)

## Migration Path

1. **Phase 1**: UI only (button grayed out with "Coming soon")
2. **Phase 2**: Enable for text+image, vision analysis only
3. **Phase 3**: Full integration with RAG context
