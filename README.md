# TCM-Sage: An Evidence-Synthesis Tool for TCM

**TCM-Sage** is an evidence-synthesis tool for Traditional Chinese Medicine (TCM) practitioners. This project aims to empower practitioners by providing explainable, evidence-backed insights from the vast corpus of classical TCM literature using a Retrieval-Augmented Generation (RAG) architecture.

## Project Background

The vast body of TCM knowledge, spanning thousands of years of literature, represents both a profound asset and a significant operational challenge. Manually searching for analogous historical cases or cross-referencing symptoms during a patient consultation is impractical. This project leverages a Large Language Model (LLM) not as a decision-maker, but as an intelligent clinical reference assistant. By creating an explainable, evidence-backed tool, TCM-Sage empowers practitioners to query the entire corpus of TCM literature in seconds, helping them validate hypotheses and deliver informed, evidence-based care.

## System Architecture

The system is built on a Modular RAG paradigm to handle the complexities of classical Chinese texts.

1. **Knowledge Base:** The current knowledge base is the full classical text of the *Huangdi Neijing (黃帝内經)*. The text has been programmatically cleaned, chunked, and embedded into a persistent **ChromaDB vector store**.

2. **Hybrid Retriever:** The retriever combines semantic vector search with a future Knowledge Graph (KG) to resolve the ambiguity of classical terminology.

3. **Reflective Generator:** A two-layer "glass-box" generator inspired by Self-RAG ensures trustworthy answers:

    - **Query Routing:** ✅ **IMPLEMENTED** - A small, fast LLM pre-classifies query severity to apply either a creative (higher temperature) or strict (zero temperature) generation setting based on clinical severity.

    - **Self-Critique:** The main LLM generates an answer and then validates it against the retrieved source text, providing a direct citation to the source chapter.

## Current Status

**Phase 2: MVP Implementation & Core Logic** ✅ **COMPLETED**

The project has successfully completed Phase 1 (Research & Data Preparation) and Phase 2 (MVP Implementation). The core application logic in `src/main.py` is now fully functional with intelligent query classification and routing capabilities.

## Project Roadmap & Timeline

This plan is aligned with the official submission deadlines.

- **Phase 1: Research & Scoping (Apr 2025 - Sep 2025) - COMPLETED**

  - [x] Literature Review, Architecture Design, Data Pipeline Construction.

  - [x] First Progress Report submitted.

- **Phase 2: MVP Implementation (Oct 2025 - Dec 2025)** ✅ **COMPLETED**

  - [x] Implement the core RAG chain (`src/main.py`).

  - [x] Implement multi-provider LLM support with Alibaba Cloud Model Studio integration.

  - [x] Build a functional Command-Line Interface (CLI).

  - [x] **Implement intelligent query classification and routing system.**

  - [ ] Submit Source Code: End of Nov, End of Dec.

- **Phase 3: Mid-Point Review & Enhancement (Jan 2026 - Feb 2026)**

  - [ ] Prepare and deliver Mid-point Presentation & Demo (Jan 7-9).

  - [ ] Submit Second Progress Report (Jan 5).

  - [ ] Begin Knowledge Graph development and web UI implementation.

  - [ ] Submit Source Code: End of Jan, End of Feb.

- **Phase 4: Pilot Testing & Evaluation (Mar 2026)**

  - [ ] Conduct internal quantitative evaluation (Latency, Precision, Faithfulness).

  - [ ] Conduct pilot testing with TCM practitioners to gather qualitative feedback.

  - [ ] Submit Source Code: End of Mar.

- **Phase 5: Final Submission (Apr 2026)**

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

    - Create a `.env` file in the project root directory
    - Configure your LLM provider and API key (see `docs/CONFIG.md` for detailed instructions):

    ```bash
    # For Alibaba Cloud Model Studio (1M free tokens for new users - recommended)
    LLM_PROVIDER=alibaba
    DASHSCOPE_API_KEY="your-alibaba-dashscope-api-key-here"
    ```

    For other providers (Google AI Studio, OpenAI, Anthropic, OpenRouter, Together AI), see `docs/CONFIG.md` for setup instructions.

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

3. **Run the Main Application:**
    This will start the interactive RAG application. You can ask questions about Traditional Chinese Medicine and get evidence-backed answers with citations from the Huangdi Neijing.

    ```bash
    python src/main.py
    ```

    The application will prompt you to enter questions. Type 'exit' to quit the program.

4. **Launch the Prototype UI (Optional, for demos):**
    A lightweight Streamlit interface is available for showcasing the system to stakeholders. It reuses the same backend pipeline without modifying the CLI flow.

    ```bash
    streamlit run src/ui_app.py
    ```

    The UI displays the detected query severity, temperature routing, and maintains a session history. Use the sidebar to review the active configuration. Remember to configure your `.env` file before launching to avoid API errors.

## Key Features

### 🧠 **Intelligent Query Classification**

TCM-Sage automatically analyzes each user query to determine its clinical severity:

- **Informational Queries**: General knowledge questions (e.g., "陰陽是什麼？") use higher temperature for creative explanations
- **Prescriptive Queries**: Medical advice questions (e.g., "頭痛如何治療？") use zero temperature for maximum accuracy

### 🔄 **Dynamic Response Generation**

The system uses three optimized LLM instances:

- **Classifier LLM**: Fast, lightweight model for query analysis
- **Informational LLM**: Main model with configurable temperature for general questions
- **Prescriptive LLM**: Main model with zero temperature for medical advice

### 📚 **Evidence-Based Answers**

All responses are backed by direct citations from the Huangdi Neijing, ensuring practitioners can verify information sources.

### 🌐 **Multi-Provider Support**

Switch between different LLM providers seamlessly:

- Alibaba Cloud Model Studio (1M free tokens)
- Google AI Studio (Free tier)
- OpenAI, Anthropic, OpenRouter, Together AI

### 🕸️ **Knowledge Graph (Hybrid Retrieval)**

Optional hybrid retrieval combines vector search with a TCM knowledge graph:

- **Entity Types**: Symptoms, Herbs, Formulas
- **Relationships**: TREATS, CONTAINS, ASSOCIATED_WITH
- Query "頭痛" returns both text passages AND related herbs/formulas from the graph
- Enable with `HYBRID_RETRIEVAL_ENABLED=true` in `.env`

## Multi-Provider LLM Support

TCM-Sage supports multiple LLM providers for flexibility and cost management:

- **Alibaba Cloud Model Studio** (Recommended - 1M free tokens for new users)
- **Google AI Studio** (Free tier available)
- **OpenAI** (GPT-4o, GPT-4o-mini)
- **Anthropic** (Claude 3.5 Sonnet, Claude 3 Haiku)
- **OpenRouter** (Access to multiple providers)
- **Together AI** (Open source models)

Switch providers by simply changing `LLM_PROVIDER=google` in your `.env` file. See [CONFIG.md](CONFIG.md) for detailed setup instructions.

## Configuration

The main application (`src/main.py`) includes configurable parameters:

- **LLM Provider**: Choose from supported providers (default: alibaba)
- **Model Selection**: Specify a particular model or use provider defaults
- **Retrieval Count (`k`)**: Number of most relevant document chunks to retrieve (default: 5). Increase for broader context, decrease for faster responses.
- **Query Classification**: Configure classifier model and temperature settings for different query types
- **Temperature Settings**:
  - `LLM_TEMPERATURE`: For informational queries (default: 0.1, can increase for creativity)
  - `PRESCRIPTIVE_TEMPERATURE`: For medical queries (default: 0.0 for maximum accuracy)

See `docs/CONFIG.md` for detailed configuration options and setup instructions.
