# COMP4898 Final Year Project: Second Progress Report

**Project Title:** TCM-Sage: An Evidence-Synthesis Tool for Traditional Chinese Medicine  
**Student:** Andy Zheng  
**Date:** January 5, 2026  
**Reporting Period:** October 2025 – December 2025 (Phase 2)  

---

## 1. Executive Summary

This report outlines the significant progress achieved during Phase 2 (MVP Implementation) of the "TCM-Sage" project. Following the foundational research and data pipeline construction in Phase 1, the primary objective of this period was to develop a functional **Minimum Viable Product (MVP)** capable of end-to-end Retrieval-Augmented Generation (RAG).

During this period, the project successfully transitioned from a theoretical architecture to a working software system. The core RAG pipeline is now fully operational, featuring a novel **Hybrid Retrieval** engine that combines vector semantic search with a Knowledge Graph (KG) to address the specific ambiguities of Classical Chinese Medicine texts. Furthermore, the system incorporates a **Reflective Generator** architecture, implementing real-time query routing and self-correction mechanisms to ensure the safety and faithfulness of generated advice. All planned milestones for the MVP have been met, and the system is ready for the mid-point demonstration scheduled for January 7–9, 2026.

## 2. Detailed Achievements in Phase 2

### 2.1. Core RAG Pipeline & Multi-Provider Architecture
The fundamental "nervous system" of TCM-Sage has been successfully implemented in Python. The system now ingests user queries, retrieves relevant context from the *Huangdi Neijing* (Yellow Emperor's Inner Canon), and synthesizes evidence-based answers.
* **Vector Store Integration:** The full text of the *Huangdi Neijing* was programmatically cleaned, split into chapters, and chunked into semantic units. These chunks were embedded using `all-MiniLM-L6-v2` and stored in a persistent local **ChromaDB** vector store, enabling sub-second semantic retrieval.
* **Multi-Provider Flexibility:** To ensure cost-efficiency and future-proofing, the system was engineered with a provider-agnostic layer. It seamlessly switches between **Alibaba Cloud Model Studio** (Qwen-Turbo), **OpenAI** (GPT-4o), and **Google** (Gemini) via simple environment variable configuration. This was critical for development, allowing the use of Alibaba's free token tier for heavy testing while retaining access to GPT-4o for final benchmarking.

### 2.2. Architectural Pivot: Hybrid Retrieval Engine
A major technical achievement this phase was the implementation of a **Hybrid Retriever**. Early testing revealed that pure vector search struggled with specific herb-symptom relationships (e.g., distinguishing between a "headache" mentioned in a metaphor versus a clinical treatment).
* **Ensemble Context Aggregation:** Instead of a complex mathematical merging of vector scores, the system now employs an "Ensemble Context" strategy. It retrieves two distinct streams of data:
    1.  **Semantic Text Chunks (Vector):** Broad context and theoretical explanations.
    2.  **Structured Facts (Knowledge Graph):** Precise entity relationships (e.g., *Headache — TREATS — Chuanxiong*).
* These streams are presented to the Large Language Model (LLM) as distinct sections ("=== Text Passages ===" and "=== Knowledge Graph Facts ==="). This "Glass Box" approach allows the LLM to explicitly reason over the structured data, significantly improving the precision of prescriptive answers.

### 2.3. The Reflective Generator: Safety & Self-Correction
Given the medical nature of the project, "hallucination" (generation of false information) is a critical safety risk. To mitigate this, a **Reflective Generator** architecture was implemented, inspired by the "Self-RAG" framework.
* **Intelligent Query Routing:** The system first classifies every incoming user query as either **"Informational"** (general concepts) or **"Prescriptive"** (clinical advice).
    * *Informational* queries use a higher temperature (0.1) for creative, fluent explanations.
    * *Prescriptive* queries force a temperature of 0.0 to maximize determinism and factual adherence.
* **Self-Correction Loop:** A post-generation verification module was added to the pipeline. After generating an answer, the system freezes the output and prompts a secondary, lightweight LLM instance to audit the response. It asks: *"Does the Proposed Answer contain any factual claims NOT supported by the Context?"* If the verifier detects unsupported claims, a warning flag (`⚠️ [Self-Correction Warning]`) is automatically appended to the user output. This functionality ensures that users are alerted to potential hallucinations in real-time.

## 3. Challenges Encountered & Solutions

### 3.1. Ambiguity in Classical Terminology
**Challenge:** The *Huangdi Neijing* uses archaic terminology where a single character can mean a symptom, an organ, or a cosmological concept depending on context. Standard vector embeddings often failed to capture these precise nuances.
**Solution:** This necessitated the shift to **Hybrid Retrieval**. By manually mapping key clinical entities (Symptoms, Herbs, Formulas) into a graph structure, we provided the system with a "ground truth" pathway. Even if the semantic search misses a subtle connection, the explicit graph edge guarantees the correct herb-symptom pair is retrieved.

### 3.2. Knowledge Graph Construction Strategy
**Challenge:** The original plan called for fully automated Information Extraction (IE) using LLMs to build the Knowledge Graph during Phase 2. However, initial experiments showed that automated extraction on Classical Chinese text requires significant prompt engineering and validation to avoid populating the graph with noise.
**Solution:** A strategic decision was made to prioritize **Reliability over Scale** for the MVP. We adopted a **Manual JSON Curation** strategy for the Phase 2 demo. We hand-verified a "Golden Path" dataset focusing on common ailments (Headache, Insomnia). This ensures the Mid-Point Demo is flawless and trustworthy. The development of the automated extraction pipeline has been rescheduled to Phase 3, where it will be the primary focus.

## 4. Updated Project Plan (Phase 3 & 4)

With the MVP complete, the focus for the next reporting period (January – February 2026) shifts from "Core Architecture" to "Scale and User Experience."

### Phase 3: Enhancement (Jan – Feb 2026)
* **Automated Knowledge Graph Construction:** This is the most significant remaining technical task. I will develop Python scripts utilizing Large Language Models to systematically parse the 81 chapters of the *Su Wen*, extracting entities (Herbs, Symptoms) and relationships (Treats, Causes) into the graph structure automatically. This will scale the graph from dozens of nodes to thousands.
* **Web Interface Implementation:** The current Command-Line Interface (CLI) will be replaced with a web-based dashboard using **Streamlit**. Key features will include a "Reasoning Visualization" panel that displays exactly which text chunks and graph nodes were used to generate the answer, enhancing the "Explainability" of the tool.

### Phase 4: Evaluation (Mar 2026)
* **Quantitative Benchmarking:** The system will be tested against a "Golden Set" of 20 predefined clinical questions. We will measure **Response Latency** (Target: <5s) and **Citation Faithfulness** (Target: >4.5/5 on human evaluation).
* **Pilot Testing:** Small-scale user testing will be conducted with 3-5 students from the HKBU School of Chinese Medicine to gather qualitative feedback on user trust and system usability.

## 5. Conclusion

Phase 2 has been highly productive. The transition from a theoretical proposal to a working code base involves overcoming significant complexity in retrieving information from classical texts. By implementing the Hybrid Retriever and Self-Correction module, TCM-Sage has evolved into a safety-conscious, evidence-backed assistant that meets the core requirements of the project proposal. I am confident in the current state of the system for the upcoming mid-point presentation and geared up for the scaling challenges in Phase 3.