# TCM-Sage — Remaining Tasks & Future Plans

Last updated: April 7, 2026 (post-Oracle review)

## CRITICAL — Must Fix Before Submission (April 8, 5pm)

### Report Fixes (from Oracle review)
- [ ] **Student ID**: Fill in real ID (currently [REDACTED]) in FYP_Final_Report.tex:32
- [ ] **Placeholders**: Replace all [PLACEHOLDER] figures — they render as empty boxes in PDF
  - FYP_Final_Report.tex:104, chapter3.tex:21/42/67/102, chapter6.tex:53
- [ ] **Ch5 vote total mismatch**: Daily breakdown (lines 59-63) totals 50 votes not 56 — missing 6 ties
  - Fix: add tie counts to each date row, or add a note explaining ties are excluded from daily breakdown
- [ ] **Ch5 stats methodology**: Line 40 says encoding is "1 for win, -1 for loss, 0 for tie"
  - This may not match the actual API code which uses 1/0/0.5 scoring against 0.5 baseline
  - Fix: verify the actual scoring method in api.py and match the description
- [ ] **LaTeX compile risk**: chapter4.tex:100 has raw Unicode 【Formatting Requirements】
  - Fix: replace with ASCII or escape: \texttt{[Formatting Requirements]}
- [ ] **Ye Tianshi error**: chapter7.tex:30 lists him as "modern clinical master" — he's Qing dynasty
  - Wenre Lun is already in the corpus (FYP_Final_Report.tex:68)
  - Fix: remove from future work modern masters list, or reword
- [ ] Student reviews all 7 chapters for voice/tone
- [ ] Show draft to Dr. Zhang Ce for feedback
- [ ] Compile LaTeX → PDF, verify formatting
- [ ] Submit by 5pm April 8

### Report Improvements (Important but not blocking)
- [ ] **Uncited claims about competitors** — examiner will ask for sources:
  - Qihuang <15% usage, 320B params (Ch1:82, Ch2:35-39, Ch6:9)
  - iFlytek 107 clinics, 9800 diagnoses
  - Fix: add proper citations or soften to "reportedly"
- [ ] **"Full local deployment" overstated** — current code uses DashScope cloud embeddings
  - Fix: reword to "local LLM inference is supported; full local pipeline requires local embedding model"
- [ ] **Citation Panel description wrong** — Ch4:125 says Cytoscape.js mini-view
  - Actually: CitationPanel uses KGViewer with @xyflow/react, not Cytoscape
  - Cytoscape is only on the dedicated /kg/ explorer page
- [ ] **Shanghan Lun clause count**: report says 388, some project docs say 398 — verify which is correct
- [ ] **AI filler language to soften/remove**:
  - Ch2:7 "dominant architecture" → "widely adopted architecture"
  - Ch2:61 "gold standard" → "widely recognized benchmark"
  - Ch2:65 "robust mathematical foundation" → "statistical foundation"
  - Ch4:8 "modern full-stack architecture designed for high-performance..." → simplify
  - Ch4:107 "current state-of-the-art" → "a representative general-purpose configuration"
- [ ] **Terminology consistency**:
  - T-Test → t-test (lowercase) everywhere
  - Xunfei vs iFlytek — pick one
  - markdown → Markdown
  - Black Box / Glass Box capitalization
- [ ] **Soften strong claims**:
  - Ch5:102 "Impact on the Field" → too strong for 56 votes
  - Ch6:25 "held consistently" → "was observed across"
  - Ch1:89 "clear gap in the market" → "apparent gap"
- [ ] **Bibliography entries to complete**:
  - chatbotarena2024: add access date, fuller metadata
  - luo2024medrag: add arXiv ID

### System
- [ ] Test measurement conversion prompt fix — ask "麻黄汤的完整药物组成和剂量" and verify 1两≈13.8g
- [ ] Verify Traditional Chinese (zh-Hant) is default language in frontend
- [ ] Verify all error boundaries work (crash a component intentionally)
- [ ] Continue collecting Arena votes (currently 56, more is better)

## Post-Submission (Before Presentation — April 10-16)

### Presentation Prep
- [ ] 30 min presentation + 10 min Q&A
- [ ] Storytelling arc: First thought → Issues found → Solutions made → Proof it works (T-Test)
- [ ] Live demo: show main chat, Citation Panel, KG Explorer, Arena
- [ ] Prepare for Prof. Wang's likely question: "What's the key difference from existing LLMs?"
- [ ] Prepare Arena stats slide: p=0.0018, d=0.47, 56+ votes

### System Polish
- [ ] Tune the "RAG absence hallucination" prompt (LLM says "文献已遗失" when RAG can't find something)
  - Need to find a new trigger case since 伤寒论 clause issue is already fixed
  - Test with queries about topics definitely NOT in the 17 texts
- [ ] Fix Arena vote analysis script (for presentation slides)
- [ ] Consider adding more Arena sample questions

## Future Development (Post-FYP)

### Corpus Expansion (from practitioner feedback)
- [ ] 张锡纯《医学衷中参西录》— modern Chinese-Western integration classic
- [ ] 经方大师: 柳渡舟, 冯世伦 works
- [ ] 叶天士 works (Wenbing school)
- [ ] 张景岳《景岳全书》(partially included already?)
- [ ] Modern acupuncture: 石学敏《实用针灸学》, 董氏奇穴, 承淡安, 薄智云腹针, 泽田健《针灸真髓》
- [ ] 《针灸大成》, 《一针疗法》
- [ ] 《本草药征》
- [ ] Chinese-Western integrated medicine references
- [ ] Evaluate national medical curriculum (国家医学大纲) for quality — use if good, skip if purely political

### Platform Direction
- [ ] Kenny wants to recommend to HKBU School of Chinese Medicine
- [ ] Explore case competition opportunities
- [ ] Consider platform model: each school/hospital maintains own knowledge base
- [ ] Patient-friendly mode: simplified TCM explanations alongside classical citations

### Technical Improvements
- [ ] Stack TCM fine-tuned model (e.g., HuatuoGPT) as base + RAG on top
- [ ] Expanded evaluation: larger sample, senior practitioners, dimension-specific scoring
- [ ] DashScope enable_search: try with qwen3.5-flash model name
- [ ] Arena auto-scroll left/right panel speed equalization

## Known Issues (Non-Blocking)
- Hydration warning from Dark Reader browser extension (cosmetic, non-blocking)
- ChromaDB `$and` filter requires specific syntax (documented in code)
- Arena DuckDuckGo search fails silently on network errors (graceful degradation by design)
- KG Explorer Cytoscape listeners lack cleanup on unmount (minor memory leak)
