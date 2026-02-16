# TCM-Sage UI Roadmap

## Vision

A comprehensive TCM research platform for practitioners and curious learners, featuring AI-powered question answering with traceable citations, classical text library, and visual diagnosis support.

---

## Phase 1: MVP — AI Chat Interface 🔵 NOW

**Goal**: Core chat experience with citations

Features:
- [x] FastAPI backend with streaming SSE
- [ ] Next.js frontend with chat UI
- [ ] Message bubbles with inline citations [1], [2]
- [ ] Clickable citations → expandable source cards
- [ ] Severity badge (Informational / Clinical)
- [ ] Query history in sidebar (localStorage)
- [ ] Light/dark mode toggle
- [ ] Mobile-responsive layout
- [ ] Sample question chips

Backend work:
- [ ] Pass KG source_ref through retriever
- [ ] Add chunk_index to ingested chunks
- [ ] Citation out-of-range verification

---

## Phase 2: Practitioner Dashboard 🟡 NEXT

**Goal**: Landing page with overview and quick actions

Features:
- [ ] Welcome message with user name (if authenticated)
- [ ] Recent queries list
- [ ] Quick action buttons (New Query, Browse Library)
- [ ] Simple stats (queries today, topics)

---

## Phase 3: Knowledge Library 🟢 FUTURE

**Goal**: Browse and search classical texts directly

Features:
- [ ] Book sidebar (Huangdi Neijing, Shen Nong Ben Cao Jing, etc.)
- [ ] Chapter browser with search
- [ ] Full text view with citation highlighting
- [ ] "Cite this passage" copy button
- [ ] Deep link from Chat citations to Library

---

## Phase 4: Patient Case Management 🟢 FUTURE

**Goal**: Save and organize patient-specific queries

Features:
- [ ] Case list with anonymized patient IDs
- [ ] Query history per case
- [ ] Notes and observations
- [ ] Link queries to specific cases

Requires:
- User authentication
- Database backend

---

## Phase 5: Visual Diagnosis (望诊) 🟢 FUTURE

**Goal**: Image upload for tongue/face analysis

Features:
- [ ] Image upload in chat (📎 button)
- [ ] Tongue photo analysis with AI
- [ ] Face diagnosis support
- [ ] Visual observations in response
- [ ] Overlay annotations on images

Requires:
- Vision-capable LLM (Gemini 2.0 / GPT-4o)
- Multipart form handling in API

---

## Phase 6: Analytics & Trends 🟢 FUTURE

**Goal**: Research insights for practitioners

Features:
- [ ] Queries per day chart
- [ ] Top topics auto-tagged
- [ ] Time range filtering
- [ ] Export query history

---

## Design Assets

| Screen | Stitch Design | Implementation Status |
|--------|---------------|----------------------|
| AI Consultation Interface | 🔲 Pending | Phase 1 |
| Practitioner Dashboard | 🔲 Pending | Phase 2 |
| Knowledge Library | 🔲 Pending | Phase 3 |
| Patient Cases | 🔲 Pending | Phase 4 |
| Query History & Trends | 🔲 Pending | Phase 6 |

---

## Notes

- Stitch designs are HTML/CSS mockups — implementation in Next.js + Tailwind
- Mobile-first for Chat and History; Desktop-optimized for Library
- Auth deferred until Patient Cases phase
