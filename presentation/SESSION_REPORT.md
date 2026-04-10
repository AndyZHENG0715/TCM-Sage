# Session Progress Report — April 10, 2026

## What Was Done This Session

### 1. Presentation Slides Built (PRIMARY)
**File:** `presentation/slides.md` (961 lines, ~22 main slides + 8 backup slides)

**Setup:**
- Slidev v52.14 with `seriph` theme installed at `D:\Dev\TCM-Sage\presentation/`
- Carbon icon pack (`@iconify-json/carbon`) for inline icons throughout
- Images copied to `presentation/public/figures/` for proper Slidev static serving
- To run: `cd presentation && npx slidev --port 3030`

**Slide Structure:**
| Section | Slides | Content |
|---------|--------|---------|
| Title | 1 | Name, ID, supervisor, observer |
| The Problem | 2–4 | TCM knowledge intensity, why existing AI fails |
| Why General AI Falls Short | 5–7 | AI投毒, citation mismatch, phantom tools, comparison grid |
| Building TCM-Sage | 8–14 | Personal story, architecture, Phase 1, Phase 2, retrieval, contributions |
| Proving It Works | 15–18 | Arena design, statistics (38/15/6), practitioner feedback |
| Live Demo | 19–20 | Demo outline card (5 features) |
| Discussion | 21–24 | Limitations, future work, conclusion, thank you |
| Backup (Q&A) | 25–31 | RAG vs fine-tuning, sample size, multi-provider, query classification, hallucination story, AI投毒 evidence, crosswalk bridge |

### 2. v-Click Consistency Fix
**Applied rules:**
- **Comparison/grid slides** (Phase 1, TCM-Sage's Approach, Arena): Show everything at once — no v-clicks
- **Sequential/narrative slides** (How I Got Here, Technical Contributions, Conclusion): v-clicks on all items
- **Practitioner Feedback**: Changed from `<v-clicks>` (all at once) to individual `<v-click>` per card (appear one by one)
- **Punchline callouts**: v-click kept on bottom callout boxes — they're the "reveal after content"
- **Backup slides**: No v-clicks (navigated manually during Q&A)

### 3. Report Cross-Check — Factual Corrections
Compared all slide content against `docs/report/chapters/` (Ch3–Ch7).

**Fixed:**
| Slide | Issue | Fix |
|-------|-------|-----|
| Domain-Specific Retrieval | Said "nomic (768d) → DashScope" | Changed to "all-MiniLM (384d) → nomic (768d) → DashScope (1024d)" — matches Ch3's three-phase evolution |
| Arena slide | Said "General LLM with web search" | Added "DuckDuckGo web search" — matches Ch5 §2.1 baseline description |
| Statistical Results | Missing vote breakdown | Added "38 RAG / 15 Plain / 6 Tie" — matches Ch5 §2.2 |

**Verified as correct (no changes needed):**
- 17 texts, 3.72M characters, 12,204 chunks ✓
- 388 clauses (Shanghan Lun), 489 clauses (Jingui Yaolue) ✓
- p = 0.0011, Cohen's d = 0.45 ✓
- SymMap 2.0: 18,450 entities, 21,476 relationships ✓
- Kenny's 4 selling points match Ch6 §2 qualitative feedback ✓
- Phase evolution narrative matches Ch3 §1.1 "planned from inception" framing ✓

### 4. Task Management
- Created reminder task **T-62cd279e**: "Post-presentation: Import TODO.md into GSD + repo cleanup" — ensures cleanup work isn't forgotten after April 13
- Completed task **T-f992773f**: "Fix Slidev reveal consistency in presentation deck"

---

## What Still Needs To Be Done

### Before April 13 (Presentation Day)
1. **Practice the presentation** — run through with Slidev presenter mode (`localhost:3030/presenter`), time yourself aiming for 28-30 minutes
2. **Test live demo** — start backend (`venv\Scripts\python.exe src/api.py`) + frontend (`cd web && npm run dev`) and run through the 5 demo features
3. **Set up ngrok** — test accessing the app from school network; bring own laptop as backup
4. **Speaker notes review** — all slides have presenter notes in `<!-- -->` comments, review and personalize
5. **Q&A prep** — review backup slides, especially RAG vs fine-tuning, sample size justification, and hallucination story

### After April 13 (Shelved Work)
- **T-62cd279e**: Import TODO.md active items → GSD todos, future items → backlog, execute repo cleanup + /gsd-docs-update
- **T-0f5da8e9**: Full repo restructure (delete orphaned files, move legacy artifacts to `.archive/`, update `.env.backup`)
- Various older tasks from previous sessions (most are stale — recommend triaging after presentation)

---

## Known Issues / Open Items
- **Slidev process management**: Do NOT use `Stop-Process -Name "node"` — it kills OpenCode's own process. Just Ctrl+C the terminal running Slidev.
- **Architecture image**: `architecture.png` is 2.2MB — loads fine locally but may be slow if exported to PDF. Consider compressing if PDF export is needed.
- **Font loading**: Google Fonts for Noto Sans SC may take a moment on first load. If presenting offline, consider pre-caching or using local fonts.
