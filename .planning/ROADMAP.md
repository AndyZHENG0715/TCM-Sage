# Project Roadmap

## Phase 1: Stabilization & Bug Fixes
*Goal: Resolve frontend tech debt and UI inconsistencies.*
- [x] Task 1.1: Fix Markdown bolding and Chinese quote rendering in `MessageBubble.tsx`.
- [x] Task 1.2: Implement trailing "Sources:" regex stripper in `MessageBubble.tsx`.
- [x] Task 1.3: Wire up Full-Text Context toggle in `CitationPanel.tsx`.
- [x] Task 1.4: Verify `KGViewer.tsx` basic rendering with existing data.

## Phase 2: Standard KG Integration
*Goal: Pivot to an academically recognized Knowledge Graph source (SymMap 2.0).*
- **Plans:** 3 plans
- [x] 2-01-PLAN.md — SymMap 2.0 Research & Mapping
- [x] 2-02-PLAN.md — Data Adapter Implementation
- [x] 2-03-PLAN.md — Pipeline Integration & Verification

## Phase 2.5: Arena Blind Evaluation
*Goal: Add blind model-comparison workflow for reliable answer quality checks.*
- [x] Backend arena endpoints and orchestration (`arena.py`)
- [x] Frontend arena page and model-side-by-side blind review flow
- [x] KG retrieval `max_results` setting support for arena runs
- [x] Arena model configuration wiring
- [x] Clickable citations and verification badge integration in arena results
- [x] Architecture docs synchronized via map-codebase

## Phase 2.7: Corpus Expansion & Embedding Upgrade
*Goal: Expand knowledge base and improve retrieval quality.*
- [x] Expand corpus from 1 text to 17 classical TCM texts (3.72M characters)
- [x] Upgrade embedding from nomic-embed-text-v1.5 (768d local) to text-embedding-v4 (1024d API)
- [x] Add checkpoint/resume to ingestion pipeline
- [x] Improve KG entity matching with jieba segmentation and colloquial alias map

## Phase 2.8: System Prompt Redesign
*Goal: Professional Chinese TCM clinical reference prompt.*
- [x] Rewrite system prompt to Chinese with 辨证论治 framework guidance
- [x] Add cite-then-explain instruction (quote original text before analysis)
- [x] Unify arena prompts (both sides use same DEFAULT_SYSTEM_PROMPT)
- [x] Make prompt .env-configurable via SYSTEM_PROMPT_OVERRIDE

## Phase 3: Presentation & E2E Polish
*Goal: Finalize UI/UX and ensure a rock-solid demo.*
- [ ] Task 3.1: Enhance KG subgraph visualization with dagre layout and cited node highlighting.
- [ ] Task 3.2: Perform end-to-end verification of the full RAG pipeline.
- [ ] Task 3.3: Prepare final demo documentation and performance metrics.
