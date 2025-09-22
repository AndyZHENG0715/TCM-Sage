# Self-RAG: Learning to Retrieve, Generate, and Critique

- **Full Title:** SELF-RAG: LEARNING TO RETRIEVE, GENERATE, AND CRITIQUE THROUGH SELF-REFLECTION
- **Authors:** Akari Asait, Zeqiu Wut, Yizhong Wang, Avirup Sil, Hannaneh Hajishirzits
- **Year:** 2023
- **Link:** http://arxiv.org/abs/2310.11511
- **Tags:** #RAG, #Self-Correction, #Framework

---

## 1. Abstract / Key Idea

> Despite their remarkable capabilities, large language models (LLMs) often produce responses containing factual inaccuracies due to their sole reliance on the parametric knowledge they encapsulate. Retrieval-Augmented Generation (RAG), an ad hoc approach that augments LMs with retrieval of relevant knowledge, decreases such issues. However, indiscriminately retrieving and incorporating a fixed number of retrieved passages, regardless of whether retrieval is necessary, or passages are relevant, diminishes LM versatility or can lead to unhelpful response generation. We introduce a new framework called Self-Reflective Retrieval-Augmented Generation (Self-RAG) that enhances an LM's quality and factuality through retrieval and self-reflection. Our framework trains a single arbitrary LM that adaptively retrieves passages on-demand, and generates and reflects on retrieved passages and its own generations using special tokens, called reflection tokens. Generating reflection tokens makes the LM controllable during the inference phase, enabling it to tailor its behavior to diverse task requirements. Experiments show that Self-RAG (7B and 13B parameters) significantly outperforms state-of-the-art LLMs and retrieval-augmented models on a diverse set of tasks. Specifically, Self-RAG outperforms ChatGPT and retrieval-augmented Llama2-chat on Open-domain QA, reasoning and fact verification tasks, and it shows significant gains in improving factuality and citation accuracy for long-form generations relative to these models.

*This paper introduces SELF-RAG, a framework that trains a language model to decide for itself when to retrieve information, and to critique its own generated answers for relevance and factual accuracy using special "reflection tokens".*

---

## 2. Key Contributions & Findings

- *Contribution 1: Adaptive, On-Demand Retrieval. Unlike standard RAG models that always retrieve a fixed number of documents, SELF-RAG learns to decide when retrieval is actually necessary for a given prompt, making the process more efficient and relevant.*
- *Contribution 2: Self-Reflection via "Reflection Tokens". The core innovation is training the LLM to generate special tokens to critique its own output. It evaluates if retrieved passages are relevant (`ISREL`), if its own statements are supported by the evidence (`ISSUP`), and how useful the overall response is (`ISUSE`).*
- *Contribution 3: Controllable and Customizable Inference. The reflection tokens allow for fine-tuned control over the model's output at inference time. By adjusting weights for different critique tokens, a user can tailor the model’s behavior—for example, prioritizing factual accuracy and citations over fluency, without needing to retrain the model.*
- *Contribution 4: State-of-the-Art Performance. The paper demonstrates that SELF-RAG significantly outperforms both standard LLMs (like Llama2) and traditional RAG models across a wide range of tasks, including question answering, reasoning, and long-form generation, particularly in improving factuality and citation accuracy.*

---

## 3. Relevance to TCM-Sage

*This paper is highly relevant to the core challenges of the TCM-Sage project, particularly regarding practitioner trust and the nuanced nature of TCM knowledge.*

- **Concept:** Self-Correction/Reflection
  - **Application:** *This is vital for building practitioner trust and overcoming the "black box" problem. By implementing a self-reflection mechanism like the one described, TCM-Sage can validate its own generated answers against the retrieved TCM texts. The `[ISSUP]` (Is Supported) token is a perfect analog for the evidence-backed, transparent system the project aims to build, providing a layer of internal fact-checking that is crucial for clinical adoption.*
- **Concept:** Adaptive Retrieval
  - **Application:** *TCM queries vary greatly in complexity. A simple question about a single herb's properties may not require extensive retrieval, whereas a complex differential diagnosis needs to draw from multiple classical sources. SELF-RAG's on-demand retrieval confirms that building a system that can intelligently decide when to search the corpus is a more effective approach than constant, indiscriminate retrieval. This directly addresses the challenge of data fragmentation by ensuring the model only pulls in evidence when necessary.*
- **Concept:** Controllable Inference for Nuanced Topics
  - **Application:** *TCM is not always about a single "correct" answer; it often involves multiple schools of thought. The ability to customize SELF-RAG's output at inference time is a powerful concept for TCM-Sage. The system could be designed to allow practitioners to adjust settings to, for example, prioritize evidence from a specific classical text or to surface conflicting viewpoints more prominently, thereby aligning the tool’s output with the nuanced and often subjective nature of TCM diagnostics.*

---

## 4. Important Quotes or Figures

> We introduce a new framework called Self-Reflective Retrieval-Augmented Generation (SELF-RAG) that enhances an LM's quality and factuality through retrieval and self-reflection.

> Generating reflection tokens makes the LM controllable during the inference phase, enabling it to tailor its behavior to diverse task requirements.

![Figure 1: Overview of SELF-RAG](<Asai et al.2024 Self RAG Figure 1.png>)
**Figure 1: Overview of SELF-RAG**
*This figure is the most important visual in the paper. It clearly contrasts the linear, one-size-fits-all process of standard RAG with SELF-RAG's dynamic, multi-step approach:*
1.  **Standard RAG:** *Retrieves a fixed set of documents and generates a single answer, which may be contradictory or based on irrelevant information.*
2.  **SELF-RAG:**
    - **Step 1: Retrieve on demand.** *First, it decides if retrieval is even needed.*
    - **Step 2: Generate in parallel.** *It generates multiple potential responses based on different retrieved passages.*
    - **Step 3: Critique and select.** *It uses reflection tokens to score each generated segment for quality and factuality, then selects the best one to continue, effectively building the best possible answer piece by piece.*

*This iterative "generate, then critique" loop is the key takeaway and directly maps to the needs of an evidence-synthesis tool like TCM-Sage.*

---
