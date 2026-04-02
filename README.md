# TCM-Sage: An Evidence-Synthesis Tool for TCM

**TCM-Sage** is an evidence-synthesis tool for Traditional Chinese Medicine (TCM) practitioners. This project aims to empower practitioners by providing explainable, evidence-backed insights from the vast corpus of classical TCM literature using a Retrieval-Augmented Generation (RAG) architecture.

## Project Background

The vast body of TCM knowledge, spanning thousands of years of literature, represents both a profound asset and a significant operational challenge. Manually searching for analogous historical cases or cross-referencing symptoms during a patient consultation is impractical. This project leverages a Large Language Model (LLM) not as a decision-maker, but as an intelligent clinical reference assistant. By creating an explainable, evidence-backed tool, TCM-Sage empowers practitioners to query the entire corpus of TCM literature in seconds, helping them validate hypotheses and deliver informed, evidence-based care.

## System Architecture

The system is built on a Modular RAG paradigm to handle the complexities of classical Chinese texts.

1. **Knowledge Base:** The current knowledge base is the full classical text of the *Huangdi Neijing (黃帝内經)*. The text has been programmatically cleaned, chunked, and embedded into a persistent **ChromaDB vector store**.

2. **Hybrid Retriever:** The retriever combines semantic vector search with an in-memory **Knowledge Graph (KG)** built with **NetworkX** to resolve the ambiguity of classical terminology.

3. **Reflective Generator:** A two-layer "glass-box" generator inspired by Self-RAG ensures trustworthy answers:

    - **Query Routing:** A small, fast LLM pre-classifies query severity to apply either a creative (higher temperature) or strict (zero temperature) generation setting based on clinical severity.

    - **Self-Critique:** The main LLM generates an answer and then validates it against the retrieved source text, providing a direct citation to the source chapter.

## Tech Stack

- **Frontend:** Next.js 16, React 19, TailwindCSS, Lucide React, XYFlow (for KG visualization)
- **Backend:** FastAPI (Python 3.10+), LangChain, Uvicorn
- **Vector Database:** ChromaDB
- **Knowledge Graph:** NetworkX
- **Embeddings:** Nomic Embed Text v1.5 (HuggingFace)
- **LLM Support:** Alibaba DashScope (Qwen), Google Gemini, OpenAI, Anthropic

## Current Status (Apr 2026)

The project is in active stabilization + knowledge graph migration:

- ✅ Next.js 16 web interface and FastAPI backend are both production-usable for local development.
- ✅ Hybrid retrieval (vector + KG) is running, with ongoing migration from legacy graph data to SymMap-based KG data.
- ✅ Multi-provider LLM support is available (Alibaba/Qwen, Gemini, OpenAI, Anthropic).
- 🔄 Prompt quality tuning, verification hardening, and end-to-end evaluation are still in progress.

For detailed planning artifacts and phase tracking, see `.planning/ROADMAP.md`.

## Setup and Installation

1. **Clone the repository:**
    ```bash
    git clone [Your GitHub Repository URL]
    cd tcm-sage
    ```

2. **Create a Python virtual environment (required):**
    ```bash
    python -m venv venv
    ```

3. **Install backend dependencies using the project venv:**
    ```bash
    # Windows
    venv\Scripts\python.exe -m pip install -r requirements.txt

    # macOS / Linux
    venv/bin/python -m pip install -r requirements.txt
    ```

4. **Install frontend dependencies:**
    ```bash
    cd web
    npm install
    cd ..
    ```

5. **Set up environment variables:**
    - Copy `.env.example` to `.env`.
    - Configure your provider credentials and retrieval settings.
    - Minimal example:
    ```bash
    LLM_PROVIDER=alibaba
    DASHSCOPE_API_KEY="your-api-key-here"
    ```

## How to Run the Code

1. **Build or refresh the vector knowledge base (run once after source updates):**
    ```bash
    # Windows
    venv\Scripts\python.exe src/ingest.py

    # macOS / Linux
    venv/bin/python src/ingest.py
    ```

2. **Start the backend API (`http://127.0.0.1:8000`):**
    ```bash
    # Windows
    venv\Scripts\python.exe src/api.py

    # macOS / Linux
    venv/bin/python src/api.py
    ```

3. **Start the frontend dev server (`http://localhost:3000`):**
    ```bash
    cd web
    npm run dev
    ```

4. **Run the CLI application (optional):**
    ```bash
    # Windows
    venv\Scripts\python.exe src/main.py

    # macOS / Linux
    venv/bin/python src/main.py
    ```

5. **Run lightweight verification scripts (optional):**
    ```bash
    # Citation formatting / reconstruction checks
    venv\Scripts\python.exe src/test_citations.py

    # SymMap KG retrieval sanity checks
    venv\Scripts\python.exe scripts/verify_symmap_retrieval.py
    ```

## Key Features

### 🧠 **Intelligent Query Classification**
TCM-Sage analyzes each query to determine clinical severity, routing it to optimized LLM instances with tailored temperature settings.

### 🕸️ **Knowledge Graph Visualization**
A modern, interactive graph viewer allows practitioners to explore relationships between symptoms, herbs, formulas, and related entities through the integrated KG pipeline.

### 📚 **Evidence-Based Answers**
All responses are backed by direct, verifiable citations from the source corpus, presented in a dedicated citation panel with source reconstruction.

### 🌐 **Multi-Provider Support**
Seamlessly switch between Alibaba Cloud, Google, OpenAI, and Anthropic for maximum flexibility and availability.

## Configuration
See `docs/CONFIG.md` for detailed configuration options including retrieval parameters, model selection, and graph depth settings.
