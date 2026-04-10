# TCM-Sage Context Handoff — Updated April 8, 2026 (Post-Submission)

## Purpose
This document captures all context needed to continue work across sessions.
Last updated: April 8, 2026, after Final Report submission.

---

## 1. Current Status

### ✅ SUBMITTED: Final Report (April 8, 2026, 5pm)
- PDF compiled via Prism AI (XeLaTeX), 42 pages, 6.36MB
- Uploaded to FYP system + BUMoodle
- Turnitin version needed separately (strip title/declaration/acknowledgments/ToC/appendices)
- Signed FYP consent form submitted

### Upcoming Deadlines
| Date | Deliverable |
|---|---|
| **Apr 10-16** | **Oral presentation + live demo** (30 min + 10 min Q&A) |
| **Apr 23** | Complete final report (PDF), Abstract, Poster (PDF+PNG), Slides (PDF), Demo video (MP4), Source code + setup guide (ZIP) |

---

## 2. File Structure (reorganized Apr 8)

```
docs/
├── report/                    # Final Report LaTeX source + compiled PDF
│   ├── FYP_Final_Report.tex
│   ├── FYP_Final_Report.pdf   # Compiled by Prism AI
│   ├── references.bib         # 12 verified citations
│   ├── chapters/              # Ch2-Ch7
│   └── figures/               # 10 PNGs + architecture.mmd
├── submissions/               # Past deliverables (progress reports, project statement)
├── school/                    # School-provided handbooks only
├── research/                  # Literature review notes
├── project/                   # Project planning docs
├── HANDOFF.md                 # This file
├── TODO.md
└── CONFIG.md
```

---

## 3. Pending Code Changes (post-submission, pre-presentation)

### Must Do
- [ ] M-07: Add `duckduckgo-search>=7.0.0` to requirements.txt
- [ ] M-08: Fix model name `qwen-turbo` → `qwen-flash` in:
  - web/lib/types.ts:84
  - web/hooks/useArena.ts:41
  - web/lib/arenaPrompts.ts:17
- [ ] M-13: Add 'general' category to query classifier (skip retrieval for non-medical queries)
  - Also change informational temp from 0.1 → 0.7
- [ ] M-02: Delete dead CLI fallback prompt in main.py:730-741
- [ ] C-01: Delete or annotate verifier.py as experimental/future
- [ ] M-09: Delete dead LLM constants from config.py
- [ ] M-06: Wire verify_citation_bounds() into post-generation check

### Already Done (this session)
- [x] M-04: `ttest_1samp` → `ttest_rel` (paired t-test) in api.py
- [x] Arena stats print mode (`?print=true`) for white-background chart export
- [x] Mermaid diagram restyled for academic white background
- [x] All 5 PLACEHOLDERs replaced (3 TikZ + 2 includegraphics)
- [x] LaTeX preamble: XeLaTeX + xeCJK + TikZ
- [x] Custom title page matching handbook format
- [x] Section order fixed (References before Appendices)
- [x] Abstract restored (accidentally deleted during reorder)
- [x] Gezhi Yulun removed from Ch1 table (not in corpus)
- [x] 88.1% moved from HuatuoGPT to Qihuang 2.0 (correct attribution)
- [x] Qihuang institution: "Shanghai University of TCM" → "multiple medical and research institutions"
- [x] MedRAG bib: wrong author Luo → correct author Xiong (arXiv:2402.13178)
- [x] KG classics bib: missing author added (Xiang, Lin, Cai, Jiang)
- [x] CNR bib annotation trimmed
- [x] Ch7 name synced in report structure section
- [x] HANDOFF.md: CST concentration, 59 votes

### Future Work (post-FYP)
- DuckDuckGo tool calling: Let LLM choose search keywords
- RAG step display in UI (SSE status events)
- TCMEval-SDT benchmark (optional, could strengthen presentation)
- Adaptive Retrieval: 'general' category skips retrieval entirely

---

## 4. Presentation Prep Notes

### Format
- 30 min presentation + 10 min Q&A
- Live demo required (per handbook)
- Assessed by supervisor (Dr. Zhang Ce) and observer (Prof. Wang Juncheng)

### Prof. Wang's Known Focus
- Wants technical depth
- Key differences from existing LLMs
- Mid-point feedback: "articulate why TCM-Sage is different from ChatGPT"

### Suggested Demo Flow
1. Main chat: ask a TCM question → show citation panel → click through to source
2. Show clause-level retrieval (ask about 伤寒论 specific clause)
3. Show KG explorer (search an entity, expand relationships)
4. Arena: run a blind comparison, show stats page
5. Settings: show multi-provider support, local deployment option

### Key Talking Points
- Glass-box vs black-box (evidence synthesis, not AI doctor)
- Crosswalk bridge novelty (2025 Chinese Medicine paper confirms "largely unexplored")
- Arena stats: 59 votes, p=0.0011, Cohen's d=0.45
- Three-phase evolution (Naive → Hybrid → Production Hybrid)
- 17 texts, 12,204 chunks, clause-level for Shanghan Lun/Jingui Yaolue

### Likely Q&A Topics
- Why not fine-tune instead of RAG? (Answer: transparency, no training data needed, easy corpus expansion)
- Sample size of 59 — is it enough? (Answer: p<0.01 with medium effect size; quality of testers matters)
- How does it handle queries outside the 17 texts? (Answer: LLM falls back to parametric knowledge, verifier flags as unsupported)
- What about patient privacy? (Answer: fully local deployment option, no data leaves the machine)

---

## 5. Key People & Context

- **Student**: ZHENG Zian (Andy), ID 22231153, HKBU CS **CST** concentration
- **Supervisor**: Dr. Zhang Ce — evaluation methodology guidance
- **Co-marker**: Prof. Wang Juncheng — wants technical depth, key differences from existing LLMs
- **TCM Consultant**: Kenny Woo Shi Nam (胡仕楠), HKBU SCM Year 5, cGPA 3.97/4.0
- **Additional Testers**: Doctoral students at Guangdong Provincial Hospital of TCM
- **Privacy**: Only Kenny consented to full name. All others use pseudonyms or role descriptions.

---

## 6. Key Design Decisions (for reference)

- Arena baseline generic prompt: DELIBERATE design choice, mirrors typical AI platforms (LLM + web search)
- verify_answer() SUPPORTED/UNSUPPORTED chosen over SelfCritiqueVerifier for performance: 1 LLM call vs 3
- Post-streaming verification: SSE architecture cannot pre-verify, this is by design
- config.py LLM constants are dead code — main.py and ui_backend.py read env vars independently
- CLI and Streamlit paths are legacy, web UI is the production interface
- Report framing: "practice-informed research" (prototype first, validate with literature)
- Phase 2 (Hybrid RAG) was planned from the start, NOT a reaction to Phase 1 problems
- Hallucination framing: core issue is unverifiability/black box, not hallucination per se
- SymMap is modern (not classical) — the TEXTS are classical, KG bridges modern↔classical

---

## 7. Unverified Changes (test before presentation)

- Measurement conversion table in system prompt (1两≈13.8g) — never tested live
- 辨证谨慎性原则 hedging in system prompt — never tested with pediatric case
- zh-Hant default language — changed in code but not browser-tested
- ErrorBoundary component — created but never triggered
- Arena 60s timeout — added but never stress-tested

---

## 8. Branch Status

- **Branch**: feature/premium-ui (ahead of main by ~30 commits)
- **Report changes**: NOT committed/pushed (intentional — avoid AI trail in git history)
- **Code changes**: Most committed and pushed
- **Vectorstore**: 12,628 docs in ChromaDB (re-ingested this session)
- **Arena**: 59 votes live (38 RAG / 15 Plain / 6 Tie)

---

## 9. April 23 Deliverables Checklist

- [ ] Complete final report PDF (with all appendices) — have this
- [ ] Abstract (separate text) — extract from report
- [ ] Poster (PDF + PNG)
- [x] Presentation slides (PDF) — **built with Slidev** (`presentation/slides.md`, 22 main + 8 backup slides)
- [ ] Demo video (MP4)
- [ ] Source code + database + system setup guide (ZIP)
  - Exclude: `.git/`, `.planning/`, `vectorstore/`, `.env`, `node_modules/`
  - Include: `data/source/`, `data/graph/`, setup guide (Appendix B content)

---

## 10. Open Problems (carried forward from previous sessions)

### RAG Absence Hallucination Prompt
- **Original trigger**: User asked about 伤寒论 Clause 82 → RAG couldn't find it (pre-clause-chunking) → LLM hallucinated "第82条可能已在历史中遗失"
- **Original trigger is now FIXED**: Clause-level chunking means all 388 伤寒论 clauses are individually indexed. That specific query works.
- **Underlying problem still exists**: When a query genuinely falls OUTSIDE the 17-text corpus, the LLM has no guidance on what to do. It may hallucinate or fabricate classical-sounding text.
- **Attempted fix**: Added prompt instruction twice, both times over-corrected (LLM said "资料里没有" even when related info DID exist in retrieved context)
- **Current state**: Instruction removed entirely. System prompt has ZERO guidance on out-of-corpus queries.
- **Testing challenge**: Hard to find good test cases because we need queries that:
  1. A TCM practitioner would reasonably ask
  2. Are genuinely NOT answerable from the 17 texts
  3. The LLM would otherwise try to answer (not obviously off-topic)
- **Candidate test queries** (untested):
  - Texts we don't have: “医宗金鉴怎么讲路极法的” (医宗金鉴 is not in corpus)
  - Modern concepts: “中药注射液的临床应用” (modern, no classical basis)
  - Cross-domain: “中医骨伤科的设置标准” (orthopedics, explicitly out of scope)
### Agent-Updated Docs (unverified quality)
- docs/CONFIG.md — agent updated but quality not checked
- .planning/codebase/ARCHITECTURE.md — agent updated but quality not checked
- web/README.md — agent replaced boilerplate but quality not checked

### 国家医学大纲 Quality Check
- User asked to evaluate if it's useful or just political — never researched

---

## 11. Important Context Rules

### Privacy
- **Only Kenny Woo Shi Nam consented to full name** in report/presentation
- All others use pseudonyms or role descriptions (e.g., 'Dr. Zhang', 'doctoral students at Guangzhou University of Chinese Medicine')
- 张新昂 → referred to as 'Dr. Zhang' (pseudonym)

### Tester Details (for Q&A prep)
- Kenny: 17 book selection, extensive testing, feedback, recruited testers from HKBU SCM + 广东省中医院大德道分院
- Dr. Zhang (张新昂): doctoral student, future direction advice (中西结合, modern clinical masters, specific text recommendations)
- Evaluation: proposal said TCMEval-SDT + Likert surveys — actually did Arena + qualitative feedback
  - Had a Google Form (1-7 rating) but only 2 responses during early bad version, decided not to use

### Report Framing Constraints
- Student concentration is **CST** (Computing and Software Technologies), NOT AI
- Phase 3 is "Production Hybrid RAG" not "Advanced RAG" (still hybrid, not a new paradigm)
- Arena baseline generic prompt is BY DESIGN — don't explain it as a limitation
- 88.1% belongs to **Qihuang 2.0**, not HuatuoGPT
- Qihuang 2.0 institution: "ECNU in collaboration with multiple medical and research institutions" (not Shanghai University of TCM for v2.0)

### Code Architecture Notes
- M-01: HybridRetriever in retriever.py is bypassed by API path. ui_backend.py does its own retrieval (duplicated but functional).
- M-05: Answer verification happens AFTER streaming, not before — this is by design (SSE architecture).
- M-12: NOT a problem — qwen3-rerank supports 500 docs/request, not 10. AGENTS.md docs are wrong.
