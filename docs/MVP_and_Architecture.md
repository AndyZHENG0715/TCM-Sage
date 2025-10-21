# TCM-Sage: MVP Definition & System Architecture

## 1. Minimum Viable Product (MVP) Definition

* **Goal:** To provide a TCM practitioner with a trustworthy assistant that can answer clinical questions by synthesizing information from a set of classical TCM texts.
* **Core Feature:** A user can ask a question in natural language (e.g., "What are the indications for the formula *Gui Pi Tang* according to classical texts?") and receive an accurate answer with direct citations pointing to the source text.
* **Data Scope:** For the MVP, the project will use a single, foundational classical text to ensure quality. The **Huangdi Neijing (黄帝内经)** is an ideal starting point.

## 2. System Architecture

* **Overall Paradigm:** The system will be based on a **Modular RAG** architecture. This paradigm was chosen for its flexibility, which allows for the combination of multiple techniques necessary for a complex domain like TCM.
* **Key Components:** The system will consist of three core modules that work in a coordinated loop:
    1.  **The Ingestion & Indexing Pipeline:** The offline process for preparing the knowledge base.
    2.  **The Hybrid Retriever:** The "search engine" responsible for finding relevant information.
    3.  **The Reflective Generator:** The "brain" that synthesizes the answer and ensures it is trustworthy.

## 3. Component Deep-Dive

### 3.1 The Hybrid Retriever

* **Goal:** To overcome "Semantic Ambiguity" in TCM by finding the most relevant passages of text, even if a user's query uses different terminology.
* **Methodology:** The implementation will be a **hybrid retrieval** strategy combining two methods, as validated by recent surveys in the field:
    1.  **Vector Search:** The text of the *Huangdi Neijing* will be embedded into a vector database. This allows the system to find passages that are *semantically similar* to the user's query.
    2.  **Knowledge Graph (KG) Search:** A small KG will be built containing key TCM entities (Herbs, Symptoms, Formulas). When a query contains a known entity, the KG can be used to find related concepts and expand the search, as demonstrated in the MedRAG paper for improving diagnostic accuracy.

### 3.2 The Knowledge Graph

* **Goal:** To explicitly map the complex relationships within TCM, providing a structured way to navigate the knowledge base.
* **MVP Content:** The initial KG will focus on mapping a small set of core entities from the *Huangdi Neijing*.
    * **Nodes:** `Symptoms`, `Herbs`, `Formulas`, `Five Elements`, `Yin/Yang`.
    * **Edges (Relationships):** `TREATS`, `IS_COMPOSED_OF`, `IS_ASSOCIATED_WITH`.

### 3.3 The Reflective Generator

* **Goal:** To generate a factually accurate, evidence-backed answer and build practitioner trust by being transparent (a "glass-box" design).
* **Methodology:** The generator will be implemented using a framework inspired by **Self-RAG**.
    1.  **Adaptive Retrieval:** The model will first determine if a query requires retrieval or if it's a simple conversational query.
    2.  **Generate & Critique:** The LLM will generate an answer based on the retrieved text and then critique its own answer to verify:
        * **Support:** Is the answer supported by the source text?
        * **Relevance:** Is the answer relevant to the question?
    3.  **Controllable Inference:** A routing mechanism will be implemented. A smaller, faster model will classify the query's clinical severity.
        * **Low Severity (Informational):** The query will be passed to the Generator with a higher temperature for a more nuanced, descriptive answer.
        * **High Severity (Diagnostic/Prescriptive):** The query will be passed to the Generator with a temperature of **0.0** to ensure maximum adherence to the source text.

## 4. Implementation Status

### 4.1 Core RAG Chain Implementation

The basic RAG pipeline has been successfully implemented in `src/main.py` with the following features:

* **Vector Store Integration**: The system loads the ChromaDB vector store created during the ingestion phase and performs semantic similarity search using the `all-MiniLM-L6-v2` embedding model.

* **Retrieval Configuration**: The retriever is configured to return the top 5 most relevant document chunks by default, with configurable parameters for different use cases.

* **Prompt Template Design**: The system uses a carefully designed prompt template that:
  - Instructs the LLM to answer in the same language as the question
  - Emphasizes evidence-based responses from the source text only
  - Requires proper citation of source chapters
  - Maintains focus on the Huangdi Neijing content

* **Citation Strategy**: The system formats retrieved documents with clear source attribution, enabling the LLM to provide accurate citations to specific chapters of the Huangdi Neijing.

* **Error Handling**: Comprehensive error handling for missing API keys, vector store files, and API failures ensures robust operation.

### 4.2 Current Capabilities

The MVP can currently:
- Load the persistent ChromaDB knowledge base
- Perform semantic similarity search on classical TCM texts
- Generate evidence-backed answers using multiple LLM providers (Alibaba Cloud Model Studio, OpenAI, Google AI Studio, Anthropic, OpenRouter, Together AI)
- Provide source citations from the Huangdi Neijing
- Handle configuration and runtime errors gracefully
- Switch between different LLM providers via simple environment variable configuration
- Support Chinese language queries with proper Unicode handling on Windows
- Use cost-effective Alibaba Cloud Model Studio with 1M free tokens for new users

### 4.3 Next Steps

The immediate next steps for Phase 2 include:
- Building an interactive Command-Line Interface (CLI)
- Implementing query routing for controllable inference
- Adding support for different query types and severity levels
