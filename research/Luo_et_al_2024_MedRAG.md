# Luo_et_al_2024_MedRAG

- **Full Title:** MedRAG: Enhancing Retrieval-augmented Generation with Knowledge Graph-Elicited Reasoning for Healthcare Copilot
- **Authors:** Xuejiao Zhao, Siyan Liu, Su-Yin Yang, and Chunyan Miao
- **Year:** 2025
- **Link:** https://dl.acm.org/doi/pdf/10.1145/3696410.3714782
- **Tags:** #RAG, #Knowledge_Graph, #HealthcareAI, #Decision_Support

---

## 1. Abstract / Key Idea

> Retrieval-augmented generation (RAG) is a well-suited technique for retrieving privacy-sensitive Electronic Health Records (EHR). It can serve as a key module of the healthcare copilot, helping reduce misdiagnosis for healthcare practitioners and patients. However, the diagnostic accuracy and specificity of existing heuristic-based RAG models used in the medical domain are inadequate, particularly for diseases with similar manifestations. This paper proposes MedRAG, a RAG model enhanced by knowledge graph (KG)-elicited reasoning for the medical domain that retrieves diagnosis and treatment recommendations based on manifestations. MedRAG systematically constructs a comprehensive four-tier hierarchical diagnostic KG encompassing critical diagnostic differences of various diseases. These differences are dynamically integrated with similar EHRs retrieved from an EHR database, and reasoned within a large language model. This process enables more accurate and specific decision support, while also proactively providing follow-up questions to enhance personalized medical decision-making. MedRAG is evaluated on both a public dataset DDXPlus and a private chronic pain diagnostic dataset (CPDD) collected from Tan Tock Seng Hospital, and its performance is compared against various existing RAG methods. Experimental results show that, leveraging the information integration and relational abilities of the KG, our MedRAG provides more specific diagnostic insights and outperforms state-of-the-art models in reducing misdiagnosis rates.

*This paper introduces MedRAG, a specialized RAG framework that leverages a structured medical knowledge graph to improve the accuracy and specificity of AI-assisted diagnostics, especially when dealing with diseases that have overlapping symptoms.*

---

## 2. Key Contributions & Findings

- *Contribution 1: The authors developed two comprehensive, four-tier hierarchical diagnostic knowledge graphs (one for chronic pain, another from a public dataset) that map out diseases and their critical diagnostic differences. This structured approach allows for more precise differentiation between similar conditions.*
- *Contribution 2: They proposed a novel RAG architecture, MedRAG, which integrates the knowledge graph to elicit deeper reasoning from the LLM. This method significantly improves the accuracy and specificity of diagnostic suggestions and allows the system to proactively generate clarifying follow-up questions when patient information is ambiguous.*
- *Contribution 3: The paper provides extensive experimental validation on two datasets, demonstrating that MedRAG outperforms several state-of-the-art RAG models in diagnostic accuracy.It also confirms the framework's adaptability across various open-source and closed-source LLMs.*

---

## 3. Relevance to TCM-Sage

*This paper is highly relevant to the TCM-Sage project, as both aim to build a trustworthy, evidence-based RAG tool for a complex medical domain.*

- **Concept:** Knowledge Graph for Semantic Disambiguation
  - **Application:** *TCM diagnosis is highly nuanced, with different patterns often sharing similar symptoms. MedRAG's success in using a Knowledge Graph to differentiate diseases with similar manifestations provides a direct blueprint for TCM-Sage. By creating a TCM terminology knowledge graph, as planned in the project statement, will make it possible to map the complex relationships between symptoms, patterns, and formulas, directly addressing the core challenge of "Semantic Ambiguity."*
- **Concept:** Proactive Questioning to Guide Diagnosis
  - **Application:** *A key innovation in MedRAG is its ability to generate targeted follow-up questions when initial information is insufficient. This is a powerful feature for a clinical assistant. Implementing a similar "Proactive Diagnostic Questioning Mechanism" in TCM-Sage would allow the tool to guide practitioners toward a more precise diagnosis by asking for key differentiating details, enhancing the tool's practical utility.*
- **Concept:** Building Practitioner Trust through Explainability
  - **Application:** *The TCM-Sage project correctly identifies practitioner trust and aversion to "black box" systems as a major hurdle. MedRAG's architecture, which grounds its output in both a knowledge graph and retrieved documents, reinforces the "glass-box" design philosophy central to TCM-Sage. This paper's methodology offers a proven approach for ensuring that all synthesized information is transparent and traceable back to a source, which is critical for clinical adoption.*

---

## 4. Important Quotes or Figures

> "However, the diagnostic accuracy and specificity of existing heuristic-based RAG models used in the medical domain are inadequate, particularly for diseases with similar manifestations."

*This quote perfectly encapsulates the core problem that both MedRAG and the TCM-Sage project are designed to solve.* 

![Figure 1: (b) MedRAG](<Luo_et_al_2024_MedRAG Figure 1(b).png>)

*This figure effectively contrasts the vague and inaccurate outputs of standard RAG models with MedRAG's ability to provide accurate, specific, and actionable diagnostic suggestions, including follow-up questions.*

---