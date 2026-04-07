# TCM-Sage Context Handoff — April 8, 2026

## Purpose
This document captures all incomplete work, abandoned tasks, and audit findings
to preserve continuity across context resets.

---

## 1. Consolidated Audit Report (42 issues)

See the full audit report pasted by the user in conversation. Key highlights:

### Critical (1)
- **C-01**: `src/verifier.py` SelfCritiqueVerifier is dead code. Production uses `verify_answer()` in main.py.

### Major Code Issues (13) — Top Priority
- **M-01**: HybridRetriever in retriever.py is bypassed by API path. ui_backend.py does its own retrieval.
- **M-04**: Stats methodology mismatch — code uses one-sample t-test (ttest_1samp against 0.5), report calls it "paired t-test"
- **M-05**: Answer verification happens AFTER streaming, not before (contradicts MVP doc)
- **M-07**: `duckduckgo-search` not in requirements.txt
- **M-09**: config.py DEFAULT_LLM_MODEL='qwen3:8b' (Ollama) but DEFAULT_LLM_PROVIDER='alibaba' — contradictory
- **M-12**: Reranker batch limit (10) not enforced in API path

### Major Doc Issues (11)
- **D-01 thru D-11**: MVP doc, README, CONFIG.md all describe outdated architecture/numbers

### Minor Issues (17)
- Dead code, stale docstrings, duplicate README paragraphs, etc.

---

## 2. Tasks Started But Abandoned

### 2a. RAG Absence Hallucination Prompt
- **Problem**: LLM says "第82条可能已在历史中遗失" when RAG can't find something
- **Attempted**: Added prompt instruction twice, both times over-corrected (LLM said "资料里没有" even when related info existed)
- **Current state**: Removed the instruction entirely. System prompt has NO guidance on this.
- **Plan**: Need to find a balanced wording. Test with queries OUTSIDE the 17-text coverage.
- **Trigger**: The 伤寒论 clause issue (original trigger) is fixed, so need new test cases.

### 2b. Measurement Conversion Verification
- **Added**: Comprehensive 古今度量衡换算 table to DEFAULT_SYSTEM_PROMPT in main.py
- **NOT tested**: Never restarted backend and asked "麻黄汤的完整药物组成和剂量"
- **Expected**: Should now say 1两≈13.8g instead of 3g

### 2c. Hedging Principles Verification
- **Added**: 【辨证谨慎性原则】to system prompt
- **NOT tested**: User's TCM friend was supposed to re-test the pediatric case
- **The specific case**: 8-month-old with adenovirus pneumonia — system was too assertive about 脾虛 and 热郁于营

### 2d. Frontend Verification (3 items)
- zh-Hant default: Changed in context.tsx but not browser-tested
- ErrorBoundary: Created component, wrapped pages, but never triggered an error
- Arena timeout: Added 60s timeout but never simulated a hang

### 2e. Agent-Updated Docs (unverified)
- docs/CONFIG.md — agent updated but quality not checked
- .planning/codebase/ARCHITECTURE.md — agent updated but quality not checked
- web/README.md — agent replaced boilerplate but quality not checked

### 2f. Main Branch Behind
- feature/premium-ui is 20+ commits ahead of main
- Last merge was earlier today but continued committing after
- Need to merge again before submission

### 2g. LaTeX Never Compiled
- Report was never run through pdflatex/xelatex
- Unknown if it compiles successfully
- Known risks: Unicode characters, tabularx column mismatches, missing figure files

### 2h. Arena Temporal Data Not Verified
- Ch5 daily breakdown was written by hand, not extracted from arena_votes.jsonl
- The analysis script had PowerShell encoding issues and was deleted
- Need to verify: Apr 3 (3 votes), Apr 4 (27 votes), Apr 5 (8 votes), Apr 6 (18 votes) = 56

### 2i. Uncited Competitor Claims
- HuatuoGPT 88.1%, Qihuang <15% usage, iFlytek 107 clinics — all uncited in report
- Oracle review flagged this as IMPORTANT
- Options: add citations or soften to "reportedly"

### 2j. 国家医学大纲 Quality Check
- User asked me to evaluate if it's useful or just political
- Never researched this

---

## 3. Report Status

### Files
- Main: docs/school/FYP_Final_Report.tex (Ch1 + abstract + acknowledgments + appendix)
- Chapters: docs/school/chapters/chapter2.tex through chapter7.tex
- Figures: docs/school/figures/ (2 arena PNGs with data labels)
- Mermaid: docs/school/figures/architecture.mmd (needs rendering to PNG)
- Bibliography: docs/school/references.bib

### Content Added This Session
- Query Classification + dual temperature strategy (Ch3, Ch4)
- 8 LLM provider support (Ch3, Ch4)
- Reranker pipeline details (Ch4)
- TCM-specific embedding prefixes (Ch4)
- Kenny's 4 selling points (Ch6)
- Practitioner feedback from Guangdong Provincial Hospital (Ch5, Ch6, Ch7)
- Mahuang Tang dosage error as honest limitation (Ch5, Ch6)
- 桔梗汤 novel recommendation case study (Ch5, Ch6)
- Corpus detail table in Appendix (17 texts with chunk counts)

### Remaining PLACEHOLDERs
- Ch1:104 — System architecture overview (render Mermaid)
- Ch3:21 — System architecture diagram
- Ch3:42 — Chunking strategy comparison
- Ch3:67 — Retrieval pipeline flowchart
- Ch3:102 — Arena evaluation flow
- Ch6:53 — Competitor comparison table

### Known Report Issues (from audit)
- M-04: "paired t-test" should be "one-sample t-test" — or change the code
- Stats numbers may not match JSONL data exactly
- Some competitor claims need citations

---

## 4. System Changes This Session

### Code Changes (committed + pushed)
- src/main.py: 辨证谨慎性原则, 古今度量衡换算表, removed over-cautious RAG absence prompt
- src/api.py: numpy.bool serialization fix for arena stats, arena streaming timeout
- src/ingest.py: duplicate clause ID fix (chapter_hash disambiguation)
- src/retriever.py: reranker wired into hybrid_search() (separate audit session)
- src/embeddings.py: TCM domain-specific prefixes (separate audit session)
- src/config.py: dead nomic references removed (separate audit session)
- web/components/ErrorBoundary.tsx: new component
- web/hooks/useArena.ts: promise error handling
- web/hooks/useChat.ts: promise error handling
- web/i18n/zh-Hant.json: Traditional Chinese translations
- web/i18n/context.tsx: zh-Hant support, default changed to zh-Hant
- web/components/Sidebar.tsx: 3-way language toggle
- web/app/arena/stats/page.tsx: chartjs-plugin-datalabels, button types
- .gitignore: added .planning/, .env, .cursor, .agent, etc.

### Repo Changes
- .planning/ removed from git tracking (still on disk)
- .env removed from tracking
- vectorstore/ removed from tracking
- Vectorstore re-ingested: 12,628 docs (after sqlite3 was accidentally emptied by git operations)

### Current Branch
- feature/premium-ui (ahead of main by ~20 commits)
- All changes pushed to origin

---

## 5. Key People & Context

- **Student**: Andy Zheng, ID 22231153, HKBU CS AI concentration
- **Supervisor**: Dr. Zhang Ce — evaluation methodology guidance
- **Co-marker**: Prof. Wang Juncheng — wants technical depth, key differences from existing LLMs
- **TCM Consultant**: Kenny Woo Shi Nam (胡仕楠), HKBU SCM Year 5, cGPA 3.97/4.0, Guangdong Provincial Hospital of TCM intern
- **Additional Testers**: Doctoral students at Guangdong Provincial Hospital (arranged through family connection)
- **Deadline**: April 8, 2026, 5pm — FINAL REPORT
- **Presentation**: April 10-16, 30 min + 10 min Q&A

---

## 6. Key Decisions Made

- Report framing: "practice-informed research" (prototype first, validate with literature)
- AI disclosure: "polishing" not "editing" in Acknowledgments
- Phase 2 (Hybrid RAG) was planned from the start, NOT a reaction to Phase 1 problems
- Phase 3 is "Production Hybrid RAG" not "Advanced RAG" (still hybrid, not a new paradigm)
- Arena baseline uses generic prompt + DuckDuckGo (this IS by design, not a bug — per M-03 audit finding)
- "Full local deployment" reworded to accurately describe cloud embeddings + local LLM option
- Hallucination framing: core issue is unverifiability/black box, not hallucination per se
- SymMap is modern (not classical) — the TEXTS are classical, KG bridges modern↔classical

---

## 7. Pending Code Changes (do AFTER report submission)

### Must Do (confirmed by user)
- M-04: Change `ttest_1samp` → `ttest_rel` in api.py:835 (paired t-test)
- M-07: Add `duckduckgo-search>=7.0.0` to requirements.txt
- M-08: Fix model name qwen-turbo → qwen-flash in:
  - web/lib/types.ts:84
  - web/hooks/useArena.ts:41
  - web/lib/arenaPrompts.ts:17
- M-13: Add 'general' category to query classifier (skip retrieval for non-medical queries)
  - Also change informational temp from 0.1 → 0.7
  - Document first, then code
- M-02: Delete dead CLI fallback prompt in main.py:730-741
- M-03: Update docs to state arena baseline mirrors common AI platforms (LLM + web search)
- M-06: Wire verify_citation_bounds() into post-generation check
  - Combine with verify_answer() result in metadata
  - User-facing message: '部分引用来源未在检索结果中找到' not technical jargon
- C-01: Delete or annotate verifier.py as experimental/future
- M-09: Delete dead LLM constants from config.py (DEFAULT_LLM_PROVIDER/MODEL/TEMPERATURE)
  - Keep paths and DEFAULT_RETRIEVAL_K only
  - CLI path (main.py) and Streamlit (ui_app.py) can be archived/deprecated
- M-11: Centralize os.getenv into config.py (future, post-FYP)
- M-12: NOT a problem — qwen3-rerank supports 500 docs/request, not 10. Fix AGENTS.md docs.

### Future Work (user approved but after FYP)
- DuckDuckGo tool calling: Let LLM choose search keywords instead of raw user query
  - Current bug: '水蛭性味' returns adult content because DDG splits on /性
  - Quick interim fix possible: prefix search with '中医'
- RAG step display in UI: Show retrieval progress (classifying → retrieving → reranking → generating)
  - SSE status events + frontend display, ArenaPanel already has similar loading messages
- TCMEval-SDT benchmark: User approved running it. Details:
  - GitHub: https://github.com/zhuyan166/TCMEval/tree/main/evaluation/TCMEval-SDT
  - 50 test cases (answers hidden), 200 train cases (with answers, use for self-eval)
  - 4 tasks: clinical info extraction (20%), pathogenesis MCQ (30%), syndrome MCQ (40%), summary (10%)
  - Need wrapper script to format RAG output as MCQ answers
  - Token cost: ~415K tokens total, ~¥4-8 RMB
  - Time: 2-3 hours (script + run + analysis)
  - Strategy: run on 50 training cases. High score → include in report. Low score → discuss as limitation.
  - Proposal mentioned this benchmark, so better to have tried than not
  - DO THIS AFTER REPORT FIXES ARE DONE
- Adaptive Retrieval merged with classifier: 'general' category skips retrieval entirely

## 8. Report Fixes Still Needed

### Critical (before submission Apr 8 5pm)
- [ ] Declaration Page — standard 'I declare this is my own work' page
- [ ] Ch7 Goal Completion Statement — compare proposed aims (from Project Statement) with actual delivery
- [ ] Acknowledgments AI Disclosure — update with specific tools, scope, level of use
- [ ] Appendix System Setup Guide — clone, install, configure, run
- [ ] References format — change from numeric [1] to author-year (Lewis et al., 2020)
- [ ] Re-render Mermaid architecture diagram (updated version with classifier + multi-provider)
- [ ] 4 UI screenshots (main chat, arena, KG explorer, settings)
- [ ] Replace all PLACEHOLDER figures
- [ ] Full section-by-section review with user for accuracy

### Important Corrections from User
- Student concentration is CST (Computing and Software Technologies), NOT AI
- Proposed aims from Project Statement: evidence-synthesis tool, glass box, hybrid retrieval, citation traceability, user testing
- Evaluation: proposal said TCMEval-SDT + Likert scale surveys — actually did Arena T-Test + qualitative feedback
  - Had a Google Form (1-7 rating) but only 2 responses during early bad version, decided not to use
- Testers: Kenny Woo Shi Nam (core, consented to full name), 广东省中医院 doctoral students/practitioners, 1 HKU TCM student
  - 张新昂 is also a doctoral student. Most doctoral students are practicing TCM physicians.
  - **Privacy: Only Kenny consented to full name. All others use pseudonyms or role descriptions only (e.g., 'Dr. Zhang', 'a doctoral student at...')**
- Kenny's contributions: 17 book selection, extensive testing, feedback, recruited HKBU SCM + 广东省中医院大德道分院 testers
- Dr. Zhang (pseudonym for 张新昂): future direction (中西结合, modern clinical masters, specific text recommendations)
- Hybrid RAG was planned from start; reranker was NOT planned, emerged from scaling needs
- 'query the entire corpus in seconds' = concept description, not about specific book count

## 9. Key Design Decisions (for future reference)
- Arena baseline generic prompt: DELIBERATE design choice, mirrors how typical AI platforms work (LLM + web search)
- verify_answer() SUPPORTED/UNSUPPORTED chosen over complex SelfCritiqueVerifier for performance: 1 LLM call vs 3
- Post-streaming verification: SSE architecture cannot pre-verify, this is by design not a bug
- config.py LLM constants are dead code — main.py and ui_backend.py read env vars independently
- CLI and Streamlit paths are legacy, web UI is the production interface
