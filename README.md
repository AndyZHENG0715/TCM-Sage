# TCM-Sage: An Evidence-Synthesis Tool for TCM

**TCM-Sage** is an evidence-synthesis tool for Traditional Chinese Medicine (TCM) practitioners. This project aims to empower practitioners by providing explainable, evidence-backed insights from the vast corpus of classical TCM literature using a Retrieval-Augmented Generation (RAG) architecture.

## Project Background

The vast body of TCM knowledge, spanning thousands of years of literature, represents both a profound asset and a significant operational challenge. Manually searching for analogous historical cases or cross-referencing symptoms during a patient consultation is impractical. This project leverages a Large Language Model (LLM) not as a decision-maker, but as an intelligent clinical reference assistant. By creating an explainable, evidence-backed tool, TCM-Sage empowers practitioners to query the entire corpus of TCM literature in seconds, helping them validate hypotheses and deliver informed, evidence-based care.

## System Architecture

The system is built on a Modular RAG paradigm to handle the complexities of classical Chinese texts.

1. **Knowledge Base:** The current knowledge base is the full classical text of the *Huangdi Neijing (黃帝内經)*. The text has been programmatically cleaned, chunked, and embedded into a persistent **ChromaDB vector store**.

2. **Hybrid Retriever:** The retriever combines semantic vector search with a future Knowledge Graph (KG) to resolve the ambiguity of classical terminology.

3. **Reflective Generator:** A two-layer "glass-box" generator inspired by Self-RAG ensures trustworthy answers:

    - **Query Routing:** A small, fast LLM pre-classifies query severity to apply either a creative (high temperature) or strict (zero temperature) generation setting.

    - **Self-Critique:** The main LLM generates an answer and then validates it against the retrieved source text, providing a direct citation to the source chapter.

## Current Status

**Phase 2: MVP Implementation & Core Logic**

The project has successfully completed Phase 1 (Research & Data Preparation). The immediate next step is to build the core application logic in `src/main.py` to create a functional command-line MVP.

## Project Roadmap & Timeline

This plan is aligned with the official submission deadlines.

* **Phase 1: Research & Scoping (Apr 2025 - Sep 2025) - COMPLETED**

    - [x] Literature Review, Architecture Design, Data Pipeline Construction.

    - [x] First Progress Report submitted.

* **Phase 2: MVP Implementation (Oct 2025 - Dec 2025)**

    - [ ] Implement the core RAG chain (`src/main.py`).

    - [ ] Build a functional Command-Line Interface (CLI).

    - [ ] Submit Source Code: End of Nov, End of Dec.

* **Phase 3: Mid-Point Review & Enhancement (Jan 2026 - Feb 2026)**

    - [ ] Prepare and deliver Mid-point Presentation & Demo (Jan 7-9).

    - [ ] Submit Second Progress Report (Jan 5).

    - [ ] Begin Knowledge Graph development and web UI implementation.

    - [ ] Submit Source Code: End of Jan, End of Feb.

* **Phase 4: Pilot Testing & Evaluation (Mar 2026)**

    - [ ] Conduct internal quantitative evaluation (Latency, Precision, Faithfulness).

    - [ ] Conduct pilot testing with TCM practitioners to gather qualitative feedback.

    - [ ] Submit Source Code: End of Mar.

* **Phase 5: Final Submission (Apr 2026)**

    - [ ] Submit Final Report (Apr 8).

    - [ ] Deliver Final Presentation & Demo (Apr 10-16).

    - [ ] Submit complete Project Archive (Apr 23).

## Setup and Installation

1. **Clone the repository:**

    ```bash
    git clone [Your GitHub Repository URL]
    cd tcm-sage
    ```

2. **Create and activate a Python virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Set up Environment Variables:**

    - Create a file named .env in the root directory.
    - Add your OpenAI API key to this file:
    ```bash
    OPENAI_API_KEY="your-api-key-here"
    ```


## How to Run the Code

1. **Build the Knowledge Base (Run only once):**
    This script will clean the source text, create chunks, generate embeddings, and build the vector store.

    ```bash
    python src/ingest.py
    ```

2. **Test the Retriever (Optional):**
    This script runs a sample query to verify the vector store is working correctly.
    ```bash
    python src/test_retriever.py
    ```
3. **Run the Main Application (Future):**
    This will start the main RAG application.
    ```bash
    python src/main.py
    ```
