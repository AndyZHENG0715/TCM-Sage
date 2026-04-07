# TCM-Sage — Remaining Tasks & Future Plans

Last updated: April 8, 2026 (post-audit session)

## Already Fixed (this session)
- [x] Student ID: 22231153
- [x] Ch5 daily vote total: added ties to each row (sums to 56)
- [x] Ch5 stats methodology: corrected to 1/0.5/0 encoding against 0.5 baseline
- [x] LaTeX Unicode brackets: replaced with ASCII
- [x] Ye Tianshi: replaced with Zhang Jingyue in Ch7 future work
- [x] AI filler language: cleaned across Ch2/Ch4
- [x] Terminology: t-test lowercase, iFlytek consistent
- [x] "clear gap" softened to "apparent gap"
- [x] Local deployment wording: accurately describes cloud embeddings + local LLM support
- [x] Citation Panel: corrected to xyflow/React Flow (not Cytoscape)
- [x] Shanghan Lun clause count: verified 388 is correct
- [x] Arena charts: added data labels to bar+pie charts
- [x] Appendix: filled with 17-text corpus detail table
- [x] Acknowledgments: added (Kenny, Guangdong practitioners, AI disclosure)
- [x] Query Classification + dual temperature: added to Ch3 and Ch4
- [x] Multi-provider support (8 providers): added to Ch3 and Ch4
- [x] Reranker pipeline wiring: reflected in Ch4 (now actually integrated)
- [x] TCM-specific embedding prefixes: added to Ch4
- [x] Kenny's qualitative feedback (4 selling points): added to Ch6
- [x] Reranker fixed in code: wired into hybrid_search() (separate audit session)
- [x] Embedding prefixes fixed in code: TCM domain-specific (separate audit session)
- [x] Dead nomic config removed from config.py (separate audit session)

## CRITICAL — Must Do Before Submission (April 8, 5pm)

### Report
- [ ] **Re-render architecture Mermaid diagram** — updated version at docs/school/figures/architecture.mmd
  - Now includes Query Classification, dual temperature, multi-provider support
- [ ] **Take 4 UI screenshots** and add to docs/school/figures/:
  - Main chat with Citation Panel open
  - Arena dual-panel view with vote buttons
  - KG Explorer (cytoscape graph)
  - Settings Panel (provider/model/temperature)
- [ ] **Replace remaining PLACEHOLDER** in Ch1 (FYP_Final_Report.tex:104) with rendered architecture.png
- [ ] **Replace PLACEHOLDERs** in Ch3 (chapter3.tex) with appropriate figures
- [ ] Student reviews all 7 chapters for voice/tone
- [ ] Show draft to Dr. Zhang Ce for feedback
- [ ] Compile LaTeX → PDF, verify formatting
- [ ] Submit by 5pm April 8
- [ ] **TCMEval-SDT benchmark** — run 50 training cases if time permits (see HANDOFF.md for details)

### Report Improvements (Important but not blocking)
- [x] ~~Uncited claims~~ — DONE: Qihuang 32B with ECNU citation, iFlytek with CNR citation, <15% deleted
- [ ] **Bibliography** — complete entries for chatbotarena2024, luo2024medrag

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
