# Gao et al. 2024 RAG Survey

- **Full Title:** Retrieval-Augmented Generation for Large Language Models: A Survey
- **Authors:** Gao, Yunfan; Xiong, Yun; Gao, Xinyu; Jia, Kangxiang; Pan, Jinliu; Bi, Yuxi; Dai, Yi; Sun, Jiawei; Wang, Meng; Wang, Haofen
- **Year:** 2024
- **Link:** https://www.proquest.com/working-papers/retrieval-augmented-generation-large-language/docview/2903732828/se-2?accountid=11440
- **Tags:** #Large language model, #retrieval-augmented generation, #natural language processing, #information retrieval

---

## 1. Abstract / Key Idea

> Large Language Models (LLMs) showcase impressive capabilities but encounter challenges like hallucination, outdated knowledge, and non-transparent, untraceable reasoning processes. Retrieval-Augmented Generation (RAG) has emerged as a promising solution by incorporating knowledge from external databases. This enhances the accuracy and credibility of the generation, particularly for knowledge-intensive tasks, and allows for continuous knowledge updates and integration of domain-specific information. RAG synergistically merges LLMs' intrinsic knowledge with the vast, dynamic repositories of external databases. This comprehensive review paper offers a detailed examination of the progression of RAG paradigms, encompassing the Naive RAG, the Advanced RAG, and the Modular RAG. It meticulously scrutinizes the tripartite foundation of RAG frameworks, which includes the retrieval, the generation and the augmentation techniques. The paper highlights the state-of-the-art technologies embedded in each of these critical components, providing a profound understanding of the advancements in RAG systems. Furthermore, this paper introduces up-to-date evaluation framework and benchmark. At the end, this article delineates the challenges currently faced and points out prospective avenues for research and development.

*This survey provides a comprehensive overview of Retrieval-Augmented Generation (RAG), detailing its evolution from simple to highly modular paradigms and systematically breaking down the core techniques for retrieval, generation, and augmentation.*

---

## 2. Key Contributions & Findings

- *Contribution 1: The paper categorizes the evolution of RAG into three distinct paradigms: Naive RAG, Advanced RAG, and Modular RAG, providing a clear framework for understanding its development.*
- *Contribution 2: It thoroughly analyzes the key technologies within the core components of RAG systems: "Retrieval," "Generation," and "Augmentation," offering a deep dive into state-of-the-art methods for each stage.*
- *Contribution 3: The survey summarizes the current evaluation methods for RAG, covering a wide range of tasks, datasets, and benchmarks, which addresses a common gap in RAG research*
- *Contribution 4: It outlines current challenges and future directions, such as handling long contexts, improving robustness, and developing production-ready RAG systems.*

---

## 3. Relevance to TCM-Sage

*This paper is highly relevant as it provides a direct technical roadmap for building and enhancing the proposed RAG architecture of TCM-Sage.*

- **Concept:** Advanced RAG / Modular RAG
  - **Application:** *The project's proposed "Hybrid Knowledge Representation" using a vector database and a knowledge graph directly aligns with the Advanced and Modular RAG paradigms. The survey's discussion of query transformation and hybrid retrieval techniques provides proven methods to address the "Semantic Ambiguity" challenge in TCM terminology.*
- **Concept:** Self-Correction/Reflection (Self-RAG)
  - **Application:** *This is vital for overcoming the "Practitioner Trust and 'Black Box' Aversion" problem. By implementing a self-reflection mechanism like Self-RAG, where the model critiques its own output against retrieved TCM texts, TCM-Sage can provide an internal layer of fact-checking and transparency, making the system's outputs more trustworthy for clinical practitioners.*
- **Concept:** Evaluation Frameworks
  - **Application:** *The paper's detailed breakdown of RAG evaluation—including metrics for retrieval quality (Context Relevance) and generation quality (Answer Faithfulness)—provides a structured methodology for the project's planned quantitative and qualitative evaluations.  This confirms that measuring retrieval accuracy and user trust are standard, critical steps for validating a RAG system.*
---

## 4. Important Quotes or Figures

> (Right) Modular RAG inherits and develops from the previous paradigm, showcasing greater flexibility overall. This is evident in the introduction of multiple specific functional modules and the replacement of existing modules. The overall process is not limited to sequential retrieval and generation; it includes methods such as iterative and adaptive retrieval.

*The figure below visually summarizes the evolution of RAG, which is a core concept of the paper.*

![Figure 3: Comparison between the three paradigms of RAG](<Gao et al. 2024 RAG Survey Figure 3.png>)
---