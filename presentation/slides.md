---
theme: seriph
title: "TCM-Sage: Glass-Box Evidence Synthesis for Classical TCM"
info: |
  Final Year Project Presentation
  ZHENG Zian (Andy) · 22231153 · CST
  Department of Computer Science, Hong Kong Baptist University
author: ZHENG Zian (Andy)
keywords: TCM, RAG, Knowledge Graph, Evidence Synthesis
presenter: true
download: true
exportFilename: TCM-Sage-FYP-Presentation
export:
  format: pdf
  timeout: 60000
transition: slide-left
aspectRatio: 16/9
fonts:
  sans: "Noto Sans SC, Inter, Roboto, Arial"
  serif: "Noto Serif SC, Georgia"
  mono: "Fira Code, Consolas"
  provider: google
colorSchema: light
defaults:
  layout: default
---

# TCM-Sage <carbon-chemistry class="inline text-blue-400" />

<div class="text-xl text-gray-500 mt-2">
A Glass-Box Evidence-Synthesis System for Classical TCM
</div>
<div class="text-sm text-gray-400 mt-1">
via Hybrid RAG and Knowledge Graph Integration
</div>

<div class="mt-12 grid grid-cols-2 gap-4 text-sm">
<div>

**ZHENG Zian, Andy** · 22231153 · CST

Department of Computer Science

Hong Kong Baptist University

</div>
<div class="text-right">

Supervisor: **Dr. ZHANG Ce**

Observer: **Prof. WANG Juncheng**

April 13, 2026 (17:10-17:55)

</div>
</div>

<div class="abs-bl mx-14 my-8 flex gap-2 text-sm text-gray-400">
  <carbon-presentation-file /> Oral Presentation & Live Demo
</div>

<!--
Welcome everyone. My name is Andy Zheng, and today I'll present my FYP project, TCM-Sage — a glass-box evidence synthesis system for classical Traditional Chinese Medicine.
-->

---
layout: section
transition: fade
---

# <carbon-intent-request-active class="inline" /> The Problem

Why do TCM practitioners need a new kind of AI tool?

---

# <carbon-book class="inline text-blue-500" /> TCM: A Knowledge-Intensive Domain

<br>

Traditional Chinese Medicine is built on **thousands of years** of classical literature.

<v-clicks>

- <carbon-document-multiple-01 class="inline text-blue-400" /> Practitioners must master texts spanning from 黃帝內經 (c. 200 BCE) to 溫病條辨 (1798 CE)
- <carbon-time class="inline text-orange-400" /> This requires **years of study** — book knowledge + clinical experience
- <carbon-search class="inline text-green-400" /> AI can learn from these texts and retrieve the precise details **in seconds**

</v-clicks>

<br>

<v-click>

<div class="p-4 rounded-xl bg-blue-50 border border-blue-200">
  <carbon-idea class="inline text-blue-500" /> <strong>Core Idea:</strong> What if AI could serve as an intelligent assistant that retrieves exactly the right classical passage when a practitioner needs it?
</div>

</v-click>

<!--
TCM is incredibly knowledge-intensive. Practitioners need to master texts spanning over 2000 years. This takes years of study and clinical experience.

AI can learn from all these texts and retrieve exactly what you need in seconds. That's what TCM-Sage does.
-->

---

# <carbon-help class="inline text-purple-500" /> Why Not Just Ask Existing AI?

Several TCM-specialized LLMs already exist — Qihuang (岐黄), HuatuoGPT, TCMChat...

<v-clicks>

<div class="mt-4 p-4 rounded-xl bg-red-50 border border-red-200">

**None** provide passage-level citation tracing to classical sources.

- <carbon-warning class="inline text-red-500" /> You get an answer, but **where did it come from?**
- <carbon-view-off class="inline text-red-500" /> The practitioner **cannot verify** against the original text
- <carbon-locked class="inline text-red-500" /> It's a **black box** — plausible-sounding, but unverifiable

</div>

</v-clicks>

<br>

<v-click>

<div class="p-4 rounded-xl bg-green-50 border border-green-200">
  <carbon-view class="inline text-green-600" /> TCM-Sage is a <strong>glass box</strong>: every claim is traceable to a specific passage in a specific classical text, with knowledge graph context linking related concepts.
</div>

</v-click>

<!--
You might ask, why not just use one of the existing TCM LLMs? Several exist — Qihuang, HuatuoGPT, TCMChat. But none of them tell you WHERE the answer came from at the passage level. That's the fundamental gap TCM-Sage fills.
-->

---
layout: section
transition: fade
---

# <carbon-warning-alt class="inline" /> Why General AI Falls Short

The limitations of web search for domain knowledge

---

# <carbon-cloud-alerting class="inline text-red-500" /> The Trust Problem with AI + Web Search

<v-clicks>

<div class="grid grid-cols-2 gap-4 mt-4">

<div class="p-3 rounded-lg bg-red-50 border border-red-100">
  <div class="font-bold text-red-600"><carbon-coronavirus class="inline" /> 1. AI投毒 (AI Poisoning)</div>
  <div class="text-sm mt-1">Fake content published online gets recommended by AI within <strong>minutes</strong></div>
  <div class="text-xs text-gray-500 mt-1">→ 茗感神经 experiment (2025): fabricated milk tea brand → AI recommended it as #1</div>
</div>

<div class="p-3 rounded-lg bg-orange-50 border border-orange-100">
  <div class="font-bold text-orange-600"><carbon-unlink class="inline" /> 2. Citation Mismatch</div>
  <div class="text-sm mt-1">AI cites articles where only the <strong>title</strong> is relevant — content is unrelated</div>
</div>

<div class="p-3 rounded-lg bg-yellow-50 border border-yellow-100">
  <div class="font-bold text-yellow-700"><carbon-bot class="inline" /> 3. Phantom Tool Calls</div>
  <div class="text-sm mt-1">LLM <strong>claims</strong> it searched online but never actually invoked the search tool</div>
</div>

<div class="p-3 rounded-lg bg-purple-50 border border-purple-100">
  <div class="font-bold text-purple-600"><carbon-theater class="inline" /> 4. Graceful Fabrication</div>
  <div class="text-sm mt-1">When AI can't find something, it invents <strong>plausible-sounding excuses</strong> instead of admitting "I don't know"</div>
</div>

</div>

</v-clicks>

<div class="mt-3 text-xs text-gray-400 text-center italic">These are not hypothetical risks — each has documented real-world evidence (see Appendix: AI投毒 Evidence).</div>

<!--
Let me explain why general AI with web search is insufficient. AI poisoning is real, citation mismatches are common, LLMs sometimes fake their search actions, and when they can't find something, they fabricate plausible excuses.
-->

---

# <carbon-compare class="inline text-blue-500" /> TCM-Sage's Approach

<div class="grid grid-cols-2 gap-6 mt-6">
<div class="p-5 rounded-xl bg-red-50 border border-red-200">

### <carbon-close-filled class="inline text-red-500" /> General LLM + Web Search

- <carbon-warning class="inline text-red-400" /> Searches unreliable internet sources
- <carbon-coronavirus class="inline text-red-400" /> Vulnerable to AI Poisoning
- <carbon-unknown class="inline text-red-400" /> Cannot guarantee source authenticity
- <carbon-arrow-down class="inline text-red-400" /> Answer quality depends on what's online

</div>
<div class="p-5 rounded-xl bg-green-50 border border-green-200">

### <carbon-checkmark-filled class="inline text-green-600" /> TCM-Sage (RAG)

- <carbon-book class="inline text-green-500" /> Retrieves from **curated classical texts** verified over millennia
- <carbon-direct-link class="inline text-green-500" /> Context injected **directly** into the prompt
- <carbon-search-locate class="inline text-green-500" /> LLM **synthesizes** — never searches the web
- <carbon-connect class="inline text-green-500" /> Every claim traceable + **KG** linking related concepts

</div>
</div>

<v-click>

<div class="mt-4 p-3 rounded-lg bg-blue-50 border border-blue-200 text-center text-sm">
  <carbon-information class="inline text-blue-500" /> When RAG provides context, the LLM does not hallucinate. The failure mode is RAG not finding what the user needs — not the AI making things up.
</div>

</v-click>

<div class="mt-2 text-xs text-gray-400 text-center italic">This architectural choice directly addresses the trust problem from the previous slide.</div>

<!--
General LLMs search the internet — unreliable, poisonable, unverifiable. TCM-Sage retrieves from curated classical texts. When RAG provides context, the LLM doesn't hallucinate.
-->

---
layout: section
transition: slide-up
---

# <carbon-build-tool class="inline" /> Building TCM-Sage

From a single text to a production system

---

# <carbon-user-avatar class="inline text-blue-500" /> How I Got Here

<br>

<v-clicks>

<div class="flex items-start gap-3 mb-4">
  <div class="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center flex-shrink-0 mt-1"><carbon-idea class="text-blue-500" /></div>
  <div>Originally selected <strong>AI-powered course selection</strong> as my FYP topic</div>
</div>

<div class="flex items-start gap-3 mb-4">
  <div class="w-8 h-8 rounded-full bg-green-100 flex items-center justify-center flex-shrink-0 mt-1"><carbon-partnership class="text-green-600" /></div>
  <div>My dad suggested pivoting to <strong>Traditional Chinese Medicine</strong> — he had connections to practitioners</div>
</div>

<div class="flex items-start gap-3 mb-4">
  <div class="w-8 h-8 rounded-full bg-purple-100 flex items-center justify-center flex-shrink-0 mt-1"><carbon-group class="text-purple-500" /></div>
  <div>I also had <strong>friends studying TCM</strong> who confirmed my interest</div>
</div>

<div class="flex items-start gap-3 mb-4">
  <div class="w-8 h-8 rounded-full bg-orange-100 flex items-center justify-center flex-shrink-0 mt-1"><carbon-checkmark class="text-orange-600" /></div>
  <div>Asked Dr. Zhang Ce to change topic — <strong>approved</strong></div>
</div>

<div class="flex items-start gap-3">
  <div class="w-8 h-8 rounded-full bg-red-100 flex items-center justify-center flex-shrink-0 mt-1"><carbon-machine-learning class="text-red-500" /></div>
  <div>The thread throughout: <strong>AI</strong> — just the application domain changed</div>
</div>

</v-clicks>

<!--
I originally chose AI-powered course selection. My dad suggested TCM, I had friends studying TCM too, and Dr. Zhang approved the change. The common thread was always AI.
-->

---

# <carbon-flow class="inline text-blue-500" /> System Architecture

<img src="/figures/architecture.png" class="h-102 mx-auto rounded-lg shadow-lg" />

<div class="text-center mt-3 text-sm text-gray-600">

**Query → Classification → Hybrid Retrieval (Vector + KG) → Reranking → Generation → Verification**

</div>

<div class="text-center mt-2 text-xs text-gray-400 italic">Key design principle: every retrieval step is transparent and verifiable — not a black box.</div>

<!--
Here's the system architecture. Query classification determines the LLM temperature. Hybrid retrieval runs vector search and knowledge graph search. Results are reranked, then the LLM generates with citations and verifies.
-->

---

# <carbon-milestone class="inline text-blue-500" /> Phase 1: Foundation
<span class="text-sm text-gray-400">September 2025 — January 2026</span>

<div class="grid grid-cols-2 gap-6 mt-4">
<div>

### <carbon-tools class="inline text-green-500" /> What I Built


- <carbon-document class="inline text-blue-400" /> **1 text**: 黃帝內經 (Huangdi Neijing)
- <carbon-terminal class="inline text-blue-400" /> CLI prototype → **Streamlit** web UI
- <carbon-diagram class="inline text-blue-400" /> Hand-built KG entities (for stability)
- <carbon-layers class="inline text-blue-400" /> 3-layer AI extraction pipeline — ready but unverified

</div>
<div>

### <carbon-presentation-file class="inline text-orange-500" /> Mid-Point Demo (Jan 2026)
- Demoed Streamlit prototype
- CLI mentioned as early-stage prototype

<br>

<div class="p-3 rounded-lg bg-orange-50 border border-orange-200 mt-2">
  <div class="font-bold text-sm"><carbon-user class="inline text-orange-600" /> Prof. Wang's Challenge:</div>
  <div class="text-sm italic mt-1">"The student is encouraged to think about the key difference between it and existing LLM models."</div>
</div>

</div>
</div>

<!--
In Phase 1, I built the foundation with one text, a Streamlit prototype, and a hand-built knowledge graph. Prof. Wang challenged me to articulate what makes this different from existing LLMs. That shaped everything next.
-->

---

# <carbon-rocket class="inline text-purple-500" /> Phase 2: Responding to the Challenge
<span class="text-sm text-gray-400">February — March 2026</span>

<v-clicks>

<div class="p-3 rounded-lg bg-blue-50 border border-blue-100 mb-3">
  <div class="font-bold"><carbon-data-structured class="inline text-blue-600" /> Knowledge Graph Pivot</div>
  <div class="text-sm mt-1">3-layer AI pipeline was ready → Dr. Zhang advised <strong>credible academic source</strong> → Pivoted to <strong>SymMap 2.0</strong> (18,450 entities, 21,476 relationships) + <strong>crosswalk bridge</strong> for ancient ↔ modern terms</div>
</div>

<div class="p-3 rounded-lg bg-green-50 border border-green-100 mb-3">
  <div class="font-bold"><carbon-application-web class="inline text-green-600" /> UI Pivot</div>
  <div class="text-sm mt-1">Streamlit → <strong>Next.js 16 production frontend</strong> — inspired by Kenny's vision of building a <strong>platform</strong></div>
</div>

<div class="p-3 rounded-lg bg-purple-50 border border-purple-100">
  <div class="font-bold"><carbon-star-filled class="inline text-purple-600" /> The Breakthrough: 17 Foundational Texts</div>
  <div class="text-sm mt-1"><strong>Kenny</strong> (HKBU SCM Year 5, cGPA 3.97) helped select the texts. 1 → <strong>17 classical TCM texts</strong> (3.72M characters). Even concepts from 黃帝內經 were explained <strong>better</strong> with the full theoretical foundation.</div>
</div>

</v-clicks>

<!--
After the mid-point, three major pivots: KG to SymMap 2.0 with crosswalk bridge, UI to Next.js, and the breakthrough — 17 foundational texts.
-->

---

# <carbon-data-refinery class="inline text-teal-500" /> Domain-Specific Retrieval

<v-clicks>

<div class="grid grid-cols-3 gap-4 mt-6">

<div class="p-4 rounded-xl bg-blue-50 border border-blue-200 text-center">
  <carbon-text-link-analysis class="text-3xl text-blue-500 mb-2" />
  <div class="font-bold text-sm">Clause-Level Chunking</div>
  <div class="text-xs mt-2 text-left">Regular embeddings couldn't handle "傷寒論第82條" searches. <strong>388</strong> + <strong>489</strong> clauses individually indexed.</div>
</div>

<div class="p-4 rounded-xl bg-green-50 border border-green-200 text-center">
  <carbon-upgrade class="text-3xl text-green-500 mb-2" />
  <div class="font-bold text-sm">Embedding Upgrade</div>
  <div class="text-xs mt-2 text-left">all-MiniLM (384d) → nomic (768d) → <strong>DashScope text-embedding-v4</strong> (1024d) with TCM domain-specific prefixes.</div>
</div>

<div class="p-4 rounded-xl bg-purple-50 border border-purple-200 text-center">
  <carbon-list-numbered class="text-3xl text-purple-500 mb-2" />
  <div class="font-bold text-sm">Reranking</div>
  <div class="text-xs mt-2 text-left"><strong>qwen3-rerank</strong> in hybrid search pipeline. Most relevant passages first.</div>
</div>

</div>

</v-clicks>

<!--
Three domain-specific optimizations: clause-level chunking, embedding upgrade with TCM-specific prefixes, and a reranker.
-->

---

# <carbon-trophy class="inline text-yellow-500" /> Technical Contributions

<div class="text-sm text-gray-400 mt-2">These five points summarize the project's core technical contributions.</div>

<v-clicks>

<div class="space-y-3 mt-4">

<div class="flex items-start gap-3">
  <div class="w-7 h-7 rounded-full bg-blue-500 text-white flex items-center justify-center flex-shrink-0 text-sm font-bold">1</div>
  <div><strong>Hybrid RAG Pipeline for Classical Chinese</strong><br><span class="text-sm text-gray-500">Vector search + SymMap 2.0 KG + crosswalk bridge for ancient ↔ modern terms</span></div>
</div>

<div class="flex items-start gap-3">
  <div class="w-7 h-7 rounded-full bg-green-500 text-white flex items-center justify-center flex-shrink-0 text-sm font-bold">2</div>
  <div><strong>Domain-Specific Retrieval Optimizations</strong><br><span class="text-sm text-gray-500">Clause-level chunking, formula-aware canonical retrieval, source authority boosting</span></div>
</div>

<div class="flex items-start gap-3">
  <div class="w-7 h-7 rounded-full bg-purple-500 text-white flex items-center justify-center flex-shrink-0 text-sm font-bold">3</div>
  <div><strong>RAG Context Engineering</strong><br><span class="text-sm text-gray-500">Pattern Priming (markdown context → markdown output), post-context instruction anchoring</span></div>
</div>

<div class="flex items-start gap-3">
  <div class="w-7 h-7 rounded-full bg-orange-500 text-white flex items-center justify-center flex-shrink-0 text-sm font-bold">4</div>
  <div><strong>Arena Blind A/B Evaluation</strong><br><span class="text-sm text-gray-500">Dual-streaming evaluation with paired t-test and Cohen's d validation</span></div>
</div>

<div class="flex items-start gap-3">
  <div class="w-7 h-7 rounded-full bg-red-500 text-white flex items-center justify-center flex-shrink-0 text-sm font-bold">5</div>
  <div><strong>Practitioner Validation</strong><br><span class="text-sm text-gray-500">Real-world testing by TCM students and practitioners from 3 institutions</span></div>
</div>

</div>

</v-clicks>

<!--
Five technical contributions: hybrid RAG, domain-specific retrieval, context engineering, blind evaluation, and practitioner validation.
-->

---
layout: section
transition: fade
---

# <carbon-chart-evaluation class="inline" /> Proving It Works

Arena Blind Evaluation & Statistical Results

---

# <carbon-compare class="inline text-blue-500" /> Arena: Blind A/B Evaluation

<div class="grid grid-cols-2 gap-6 mt-2">
<div>


- <carbon-idea class="inline text-blue-400" /> Idea born in a **catch-up meeting** with Dr. Zhang
- <carbon-growth class="inline text-green-400" /> Grew into full implementation inspired by **LMSYS Chatbot Arena**
- <carbon-split-screen class="inline text-purple-400" /> Two responses streamed **side-by-side**, identity hidden
- <carbon-checkmark-filled class="inline text-green-500" /> **Model A**: TCM-Sage (RAG-enhanced)
- <carbon-close-outline class="inline text-red-400" /> **Model B**: General AI (LLM + web search)
- <carbon-user class="inline text-orange-400" /> User votes **without knowing** which is which


</div>
<div>

<img src="/figures/ui-arena.png" class="rounded-lg shadow-lg" />

</div>
</div>

<!--
I built an Arena blind evaluation inspired by LMSYS Chatbot Arena. Two responses streamed side by side, identity hidden. Users vote without knowing which is RAG-enhanced.
-->

---

# <carbon-chart-bar class="inline text-green-500" /> Statistical Results

<div class="grid grid-cols-2 gap-6 mt-1">
<div>

### <carbon-group class="inline text-blue-400" /> 59 Blind Votes (38 RAG / 6 Tie)

From real TCM practitioners and students:
- <carbon-education class="inline" /> HKBU School of Chinese Medicine
- <carbon-education class="inline" /> HKU TCM students
- <carbon-hospital class="inline" /> Doctoral students at 廣東省中醫藥大學
- <carbon-hospital class="inline" /> Practitioners at 廣東省中醫院

<div class="mt-1 p-3 rounded-lg bg-green-50 border border-green-200">

| Metric | Value |
|--------|-------|
| **Paired t-test** | **p = 0.0011** |
| **Cohen's d** | **0.45** (medium) |
| **Conclusion** | Significant preference for RAG |

</div>

<div class="mt-2 text-xs text-gray-400 italic">Significance (p-value) + practical impact (Cohen's d) + domain-valid testers together validate the result.</div>

</div>
<div>

<img src="/figures/arena-win-rate.png" class="rounded-lg shadow mb-3" />
<img src="/figures/arena-vote-distribution.png" class="rounded-lg shadow" />

</div>
</div>

<!--
59 blind votes from real practitioners. Paired t-test p = 0.0011, Cohen's d = 0.45. Statistically significant preference for RAG.
-->

---

# <carbon-star class="inline text-yellow-500" /> Practitioner Feedback

Qualitative feedback from **Kenny Woo Shi Nam** (HKBU SCM Year 5, cGPA 3.97/4.0):

<div class="grid grid-cols-2 gap-4 mt-6">

<v-click>

<div class="p-4 rounded-xl bg-blue-50 border border-blue-200">
  <carbon-connect-target class="text-2xl text-blue-500 mb-2" />
  <div class="font-bold text-sm">Focused & Relevant</div>
  <div class="text-xs mt-1">Focused answers with adequate elaboration and valid arguments — no irrelevant details</div>
</div>

</v-click>

<v-click>

<div class="p-4 rounded-xl bg-green-50 border border-green-200">
  <carbon-checkmark-outline class="text-2xl text-green-500 mb-2" />
  <div class="font-bold text-sm">Credible Sources</div>
  <div class="text-xs mt-1">Extracts and integrates credible sources from TCM classics</div>
</div>

</v-click>

<v-click>

<div class="p-4 rounded-xl bg-purple-50 border border-purple-200">
  <carbon-direction-merge class="text-2xl text-purple-500 mb-2" />
  <div class="font-bold text-sm">TCM Thinking</div>
  <div class="text-xs mt-1">Guides practitioners how to think from TCM perspectives — treatment approaches, medica recommendations</div>
</div>

</v-click>

<v-click>

<div class="p-4 rounded-xl bg-orange-50 border border-orange-200">
  <carbon-settings-adjust class="text-2xl text-orange-500 mb-2" />
  <div class="font-bold text-sm">Domain-Tuned</div>
  <div class="text-xs mt-1">Tuned for the TCM domain</div>
</div>

</v-click>

</div>

<v-click>

<div class="mt-4 text-center text-sm text-gray-500">
  <carbon-arrow-right class="inline" /> Kenny has expressed interest in recommending the system to <strong>HKBU School of Chinese Medicine</strong>
</div>

</v-click>

<!--
Kenny highlighted four selling points: focused answers, credible sources, TCM thinking guidance, and domain-specific tuning.
-->

---
layout: section
transition: slide-up
---

# <carbon-screen class="inline" /> Live Demo

---
layout: center
---

# <carbon-play-filled class="inline text-green-500" /> Live Demonstration

<div class="grid grid-cols-1 gap-3 mt-8 max-w-lg mx-auto">

<div class="flex items-center gap-3 p-3 rounded-lg bg-blue-50 border border-blue-200">
  <div class="w-8 h-8 rounded-full bg-blue-500 text-white flex items-center justify-center font-bold">1</div>
  <div><strong>Main Chat</strong> — TCM question → streaming + citation panel</div>
</div>

<div class="flex items-center gap-3 p-3 rounded-lg bg-green-50 border border-green-200">
  <div class="w-8 h-8 rounded-full bg-green-500 text-white flex items-center justify-center font-bold">2</div>
  <div><strong>Clause Retrieval</strong> — "傷寒論第82條" → exact match</div>
</div>

<div class="flex items-center gap-3 p-3 rounded-lg bg-purple-50 border border-purple-200">
  <div class="w-8 h-8 rounded-full bg-purple-500 text-white flex items-center justify-center font-bold">3</div>
  <div><strong>KG Explorer</strong> — Entity relationships in SymMap 2.0</div>
</div>

<div class="flex items-center gap-3 p-3 rounded-lg bg-orange-50 border border-orange-200">
  <div class="w-8 h-8 rounded-full bg-orange-500 text-white flex items-center justify-center font-bold">4</div>
  <div><strong>Arena Stats</strong> — T-test results & charts</div>
</div>

<div class="flex items-center gap-3 p-3 rounded-lg bg-red-50 border border-red-200">
  <div class="w-8 h-8 rounded-full bg-red-500 text-white flex items-center justify-center font-bold">5</div>
  <div><strong>Settings</strong> — Multi-provider LLM switching</div>
</div>

</div>

<div class="mt-4 text-xs text-gray-400 text-center italic">
  <carbon-warning-alt class="inline" /> If live connectivity is unstable, I will continue with prepared screenshots and video of the same workflow.
</div>

<!--
I'll demonstrate five features. Let me switch to the browser.
[SWITCH TO BROWSER]
-->

---
layout: section
transition: fade
---

# <carbon-chat class="inline" /> Discussion

---

# <carbon-warning class="inline text-orange-500" /> Limitations

<div class="text-sm text-gray-400 mt-1 mb-2">All three limitations are acknowledged with current status and a clear next-step plan.</div>

<v-clicks>

<div class="space-y-3 mt-4">

<div class="p-3 rounded-lg bg-orange-50 border border-orange-200">
  <div class="font-bold text-sm"><carbon-error class="inline text-orange-600" /> RAG Absence Hallucination</div>
  <div class="text-xs mt-1">When retrieval fails, LLM fabricates excuses (e.g., "此文献可能已在历史中遗失"). Prompt fix attempted → over-corrected → removed. Original trigger resolved by clause-level chunking; underlying issue <strong>open but non-reproducible</strong>.</div>
</div>

<div class="p-3 rounded-lg bg-yellow-50 border border-yellow-200">
  <div class="font-bold text-sm"><carbon-catalog class="inline text-yellow-700" /> Corpus Scope</div>
  <div class="text-xs mt-1">17 texts cover foundational theory well. Modern clinical practice material still <strong>limited</strong> (practitioner feedback).</div>
</div>

<div class="p-3 rounded-lg bg-blue-50 border border-blue-200">
  <div class="font-bold text-sm"><carbon-chart-bar class="inline text-blue-600" /> Sample Size</div>
  <div class="text-xs mt-1">59 votes — small but statistically significant (p < 0.01). Quality of testers (real practitioners) matters alongside quantity.</div>
</div>

</div>

</v-clicks>

<!--
Limitations: RAG absence hallucination when retrieval fails, corpus scope limited to foundational theory, and sample size of 59 votes.
-->

---

# <carbon-growth class="inline text-green-500" /> Future Work

<div class="grid grid-cols-2 gap-4 mt-6">

<v-clicks>

<div class="p-4 rounded-xl bg-blue-50 border border-blue-200">
  <carbon-add-filled class="text-2xl text-blue-500 mb-2" />
  <div class="font-bold text-sm">Corpus Expansion</div>
  <div class="text-xs mt-1">Practitioner-recommended: 张锡纯《医学衷中参西录》, works by 刘渡舟, 冯世伦, modern acupuncture texts</div>
</div>

<div class="p-4 rounded-xl bg-green-50 border border-green-200">
  <carbon-cloud-app class="text-2xl text-green-500 mb-2" />
  <div class="font-bold text-sm">Platform Development</div>
  <div class="text-xs mt-1">Kenny interested in HKBU SCM recommendation. Each institution maintains own knowledge base.</div>
</div>

<div class="p-4 rounded-xl bg-purple-50 border border-purple-200">
  <carbon-user-favorite class="text-2xl text-purple-500 mb-2" />
  <div class="font-bold text-sm">Patient-Friendly Mode</div>
  <div class="text-xs mt-1">Simplified explanations alongside classical citations for non-specialist users</div>
</div>

<div class="p-4 rounded-xl bg-orange-50 border border-orange-200">
  <carbon-machine-learning class="text-2xl text-orange-500 mb-2" />
  <div class="font-bold text-sm">Fine-Tuned Base Model</div>
  <div class="text-xs mt-1">Stack TCM-specialized LLM (e.g., HuatuoGPT) + RAG pipeline for compounding improvements</div>
</div>

</v-clicks>

</div>

<!--
Future directions: corpus expansion, platform development, patient-friendly mode, and fine-tuned base model integration.
-->

---

# <carbon-checkmark-outline class="inline text-green-500" /> Conclusion

<v-clicks>

<div class="space-y-3 mt-6">

<div class="flex items-start gap-3">
  <carbon-checkmark-filled class="text-green-500 flex-shrink-0 mt-1" />
  <div>Built a <strong>working evidence-synthesis system</strong> for classical Traditional Chinese Medicine</div>
</div>

<div class="flex items-start gap-3">
  <carbon-checkmark-filled class="text-green-500 flex-shrink-0 mt-1" />
  <div>Fills a gap neither general LLMs nor existing TCM AI products fill: <strong>transparent, citation-backed retrieval</strong></div>
</div>

<div class="flex items-start gap-3">
  <carbon-checkmark-filled class="text-green-500 flex-shrink-0 mt-1" />
  <div><strong>17 classical texts</strong>, <strong>12,200+ chunks</strong>, clause-level precision for 傷寒論 and 金匱要略</div>
</div>

<div class="flex items-start gap-3">
  <carbon-checkmark-filled class="text-green-500 flex-shrink-0 mt-1" />
  <div><strong>SymMap 2.0</strong> knowledge graph with crosswalk bridge for ancient ↔ modern term mapping</div>
</div>

<div class="flex items-start gap-3">
  <carbon-checkmark-filled class="text-green-500 flex-shrink-0 mt-1" />
  <div>Statistically validated: <strong>p = 0.0011</strong>, tested by <strong>real TCM practitioners</strong></div>
</div>

<div class="flex items-start gap-3">
  <carbon-checkmark-filled class="text-green-500 flex-shrink-0 mt-1" />
  <div><strong>Glass box, not black box</strong> — every answer is verifiable</div>
</div>

<div class="mt-3 text-sm text-blue-600 text-center font-medium">Glass-box verifiability — where every answer traces back to a classical source — is what sets TCM-Sage apart.</div>

</div>

</v-clicks>

<!--
TCM-Sage fills a gap no existing tool addresses. 17 classical texts, clause-level precision, knowledge graph with crosswalk bridge, statistically validated by real practitioners. Glass box, not black box.
-->

---
layout: center
class: text-center
transition: fade
---

# Thank You <carbon-favorite class="inline text-red-400" />

<br>

<div class="text-2xl">Questions?</div>

<br>

**ZHENG Zian (Andy)** · 22231153

Supervisor: Dr. ZHANG Ce · Observer: Prof. WANG Juncheng

<div class="abs-bl mx-14 my-8 text-sm text-gray-400">
  Department of Computer Science, Hong Kong Baptist University
</div>

<!--
Thank you for listening. I'm happy to take any questions.
-->

---
layout: section
---

# <carbon-notebook class="inline" /> Appendix

---

# Why RAG, Not Fine-Tuning?

| | <carbon-search-locate class="inline" /> RAG | <carbon-machine-learning class="inline" /> Fine-Tuning |
|---|---|---|
| **Transparency** | Can see retrieved sources | Black box |
| **Training data** | None needed | Requires instruction-tuning pairs |
| **Corpus updates** | Add text files, re-ingest | Retrain the model |
| **Base LLM** | Swappable (8 providers) | Locked to one model |
| **Classical Chinese** | Works with any LLM | May degrade general capabilities |

<div class="mt-4 p-3 rounded-lg bg-green-50 border border-green-200 text-sm">
  <carbon-information class="inline text-green-500" /> RAG provides the transparency that TCM practitioners need to <strong>verify</strong> AI-generated advice against original texts.
</div>

<!--
Backup cue: use when asked why not fine-tune a TCM model instead — walk through each table row.
-->

---

# Sample Size: Is 59 Enough?

<div class="space-y-3 mt-6">

<div class="flex items-start gap-3">
  <carbon-checkmark-filled class="text-green-500 flex-shrink-0 mt-1" />
  <div><strong>p = 0.0011</strong> — well below 0.05 threshold, even below 0.01</div>
</div>

<div class="flex items-start gap-3">
  <carbon-checkmark-filled class="text-green-500 flex-shrink-0 mt-1" />
  <div><strong>Cohen's d = 0.45</strong> — medium effect size, meaningful practical significance</div>
</div>

<div class="flex items-start gap-3">
  <carbon-checkmark-filled class="text-green-500 flex-shrink-0 mt-1" />
  <div>Testers were <strong>real TCM practitioners and students</strong>, not random participants</div>
</div>

<div class="flex items-start gap-3">
  <carbon-checkmark-filled class="text-green-500 flex-shrink-0 mt-1" />
  <div>Consistent with LMSYS Chatbot Arena vote-based methodology</div>
</div>

</div>

<!--
Backup cue: use when asked to defend sample size — lead with p-value, then effect size, then tester quality.
-->

---

# Multi-Provider Architecture

<div class="mt-6 text-center">

**8 LLM Providers Supported:**

<div class="flex flex-wrap justify-center gap-3 mt-4">
  <div class="px-3 py-1 rounded-full bg-blue-100 text-blue-700 text-sm">DashScope (Qwen)</div>
  <div class="px-3 py-1 rounded-full bg-green-100 text-green-700 text-sm">OpenAI</div>
  <div class="px-3 py-1 rounded-full bg-orange-100 text-orange-700 text-sm">Anthropic</div>
  <div class="px-3 py-1 rounded-full bg-red-100 text-red-700 text-sm">Google</div>
  <div class="px-3 py-1 rounded-full bg-purple-100 text-purple-700 text-sm">OpenRouter</div>
  <div class="px-3 py-1 rounded-full bg-yellow-100 text-yellow-700 text-sm">Together AI</div>
  <div class="px-3 py-1 rounded-full bg-gray-100 text-gray-700 text-sm">Ollama</div>
  <div class="px-3 py-1 rounded-full bg-teal-100 text-teal-700 text-sm">LMStudio</div>
</div>

</div>

<div class="grid grid-cols-2 gap-4 mt-6">
<div class="p-3 rounded-lg bg-blue-50 border border-blue-100 text-sm">
  <carbon-settings class="inline text-blue-500" /> Switch providers via Settings UI — no code changes
</div>
<div class="p-3 rounded-lg bg-green-50 border border-green-100 text-sm">
  <carbon-locked class="inline text-green-500" /> <strong>Local deployment</strong>: Ollama/LMStudio for complete privacy
</div>
</div>

---

# Query Classification & Dual Temperature

<div class="grid grid-cols-2 gap-6 mt-8">

<div class="p-4 rounded-xl bg-blue-50 border border-blue-200 text-center">
  <carbon-information class="text-3xl text-blue-500 mb-2" />
  <div class="font-bold">Informational</div>
  <div class="text-sm mt-1">"什么是六经辨证？"</div>
  <div class="text-xs text-gray-500 mt-2">→ Higher temperature for richer explanation</div>
</div>

<div class="p-4 rounded-xl bg-red-50 border border-red-200 text-center">
  <carbon-medication class="text-3xl text-red-500 mb-2" />
  <div class="font-bold">Prescriptive</div>
  <div class="text-sm mt-1">"麻黄汤的剂量？"</div>
  <div class="text-xs text-gray-500 mt-2">→ Lower temperature for clinical precision</div>
</div>

</div>

<div class="mt-4 text-sm text-center text-gray-500">
Classification runs <strong>first</strong> (separate smaller model) → determines temperature → then retrieval + generation
</div>

---

# RAG Hallucination: The Full Story

<div class="space-y-2 mt-4 text-sm">

<div class="flex items-start gap-2">
  <div class="w-5 h-5 rounded-full bg-red-500 text-white flex items-center justify-center flex-shrink-0 text-xs">1</div>
  <div>Kenny searched <strong>"傷寒論第82條"</strong> → system returned <strong>Clause 2</strong> instead (wrong)</div>
</div>
<div class="flex items-start gap-2">
  <div class="w-5 h-5 rounded-full bg-orange-500 text-white flex items-center justify-center flex-shrink-0 text-xs">2</div>
  <div>Implemented clause-level chunking → found correct clause, but <strong>only one at a time</strong></div>
</div>
<div class="flex items-start gap-2">
  <div class="w-5 h-5 rounded-full bg-yellow-500 text-white flex items-center justify-center flex-shrink-0 text-xs">3</div>
  <div><strong>"第2條和第82條"</strong> → found Clause 2, <strong>missed</strong> Clause 82</div>
</div>
<div class="flex items-start gap-2">
  <div class="w-5 h-5 rounded-full bg-green-500 text-white flex items-center justify-center flex-shrink-0 text-xs">4</div>
  <div><strong>"第82條和第2條"</strong> → found 82 correctly (query-ordering bias revealed)</div>
</div>
<div class="flex items-start gap-2">
  <div class="w-5 h-5 rounded-full bg-blue-500 text-white flex items-center justify-center flex-shrink-0 text-xs">5</div>
  <div>Implemented <strong>multi-clause search</strong> → fixed</div>
</div>
<div class="flex items-start gap-2">
  <div class="w-5 h-5 rounded-full bg-purple-500 text-white flex items-center justify-center flex-shrink-0 text-xs">6</div>
  <div>Deeper issue: when retrieval misses, LLM fabricates <em>"此文献可能已在历史中遗失"</em></div>
</div>
<div class="flex items-start gap-2">
  <div class="w-5 h-5 rounded-full bg-red-500 text-white flex items-center justify-center flex-shrink-0 text-xs">7</div>
  <div>Prompt fix → <strong>over-corrected</strong> → removed. Issue <strong>open but non-reproducible</strong></div>
</div>

</div>

<!--
Backup cue: use when asked about hallucination — walk steps 1–7; step 7: issue open but non-reproducible since clause-level chunking resolved the original trigger.
-->

---

# AI投毒 Evidence

<div class="grid grid-cols-3 gap-4 mt-6">

<div class="p-4 rounded-xl bg-red-50 border border-red-200">
  <div class="font-bold text-sm text-red-600">茗感神经 Experiment</div>
  <div class="text-xs text-gray-500">AI新榜, Sep 2025</div>
  <div class="text-xs mt-2">Fabricated milk tea brand, 2 fake articles. Within <strong>10 minutes</strong>, AI recommended it as #1.</div>
</div>

<div class="p-4 rounded-xl bg-orange-50 border border-orange-200">
  <div class="font-bold text-sm text-orange-600">Salesforce Research</div>
  <div class="text-xs text-gray-500">Mar 2026</div>
  <div class="text-xs mt-2">"Poisoning the Well" — <strong>~80% attack success rate</strong> on AI search agents.</div>
</div>

<div class="p-4 rounded-xl bg-purple-50 border border-purple-200">
  <div class="font-bold text-sm text-purple-600">虎嗅 Investigation</div>
  <div class="text-xs text-gray-500">Mar 2026</div>
  <div class="text-xs mt-2">"当8家顶级AI集体说谎" — fake product across <strong>8 AI platforms</strong>, all fell for it.</div>
</div>

</div>

<div class="mt-4 p-3 rounded-lg bg-green-50 border border-green-200 text-center text-sm">
  <carbon-security class="inline text-green-500" /> TCM-Sage's curated classical corpus is <strong>immune</strong> to this attack vector.
</div>

<!--
Backup cue: use when asked for evidence of AI poisoning risks — three independent sources, all 2025–2026.
-->

---

# Crosswalk Bridge

<div class="grid grid-cols-2 gap-6 mt-6">

<div>

### <carbon-warning class="inline text-orange-500" /> Problem
- Same word can mean **different things** across 2000 years
- Modern "感冒" ≠ classical "伤寒"
- Both relate to cold/flu but have different clinical meanings

</div>
<div>

### <carbon-checkmark class="inline text-green-500" /> Solution
- **SymMap 2.0** provides modern terminology (18,450 entities)
- **Crosswalk bridge** maps modern → classical entities
- jieba segmentation + colloquial alias mapping
- A 2025 paper confirmed this is **"largely unexplored"**

</div>
</div>

<!--
Backup cue: use when asked about terminology mismatch between classical and modern Chinese medical terms.
-->
