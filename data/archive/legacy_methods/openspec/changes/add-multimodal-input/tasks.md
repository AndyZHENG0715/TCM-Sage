# Multimodal Input Support - Tasks

## Phase 1: Frontend UI (Can start before backend)

- [ ] 1.1 Add file upload button to ChatInput component
- [ ] 1.2 Implement drag-and-drop zone for images
- [ ] 1.3 Show thumbnail preview with remove button
- [ ] 1.4 Display uploaded images in MessageBubble
- [ ] 1.5 Add "Coming soon" tooltip if backend not ready

## Phase 2: API Layer

- [ ] 2.1 Install `python-multipart` dependency
- [ ] 2.2 Modify `/query` to accept `multipart/form-data`
- [ ] 2.3 Parse and validate image attachments (type, size)
- [ ] 2.4 Pass images to backend processing

## Phase 3: Vision LLM Integration

- [ ] 3.1 Add vision LLM provider config (VISION_LLM_PROVIDER)
- [ ] 3.2 Create `create_vision_llm()` function in main.py
- [ ] 3.3 Create tongue/face analysis prompt templates
- [ ] 3.4 Combine vision analysis with RAG context
- [ ] 3.5 Return combined response with visual observations

## Phase 4: Testing & Validation

- [ ] 4.1 Unit tests for file upload parsing
- [ ] 4.2 Integration test with sample tongue image
- [ ] 4.3 E2E test: upload image + text query
- [ ] 4.4 Error handling: invalid file type, too large, etc.

## Dependencies

- Phase 1 can run in parallel with Phase 2-3
- Phase 3 depends on Phase 2 completion
- Phase 4 depends on all previous phases
