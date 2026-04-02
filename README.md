# TCM-Sage: An Evidence-Synthesis Tool for TCM

**TCM-Sage** is an evidence-synthesis tool for Traditional Chinese Medicine (TCM) practitioners. This project aims to empower practitioners by providing explainable, evidence-backed insights from the vast corpus of classical TCM literature using a Retrieval-Augmented Generation (RAG) architecture.

## Project Background

The vast body of TCM knowledge, spanning thousands of years of literature, represents both a profound asset and a significant operational challenge. Manually searching for analogous historical cases or cross-referencing symptoms during a patient consultation is impractical. This project leverages a Large Language Model (LLM) not as a decision-maker, but as an intelligent clinical reference assistant. By creating an explainable, evidence-backed tool, TCM-Sage empowers practitioners to query the entire corpus of TCM literature in seconds, helping them validate hypotheses and deliver informed, evidence-based care.

## System Architecture

The system is built on a Modular RAG paradigm to handle the complexities of classical Chinese texts.

2. **Knowledge Base:** 17 classical TCM texts (3.72M characters) including the four canonical works (《黄帝内经》《伤寒论》《金匮要略》《神农本草经》), 本草纲目, 备急千金要方, 金元四大家著作, and 温病学 texts. Texts are chunked with sentence-aware splitting and embedded into a persistent **ChromaDB** vector store using **DashScope text-embedding-v4** (1024 dimensions).

2. **Hybrid Retriever:** The retriever combines semantic vector search with the **SymMap 2.0 Knowledge Graph** (18,450 entities, 21,476 relationships) built with **NetworkX** and connected via a crosswalk bridge for entity resolution.

3. **Reflective Generator:** A two-layer "glass-box" generator inspired by Self-RAG ensures trustworthy answers:

    - **Query Routing:** A small, fast LLM pre-classifies query severity to apply either a creative (higher temperature) or strict (zero temperature) generation setting based on clinical severity.

    - **Self-Critique:** The main LLM generates an answer and then validates it against the retrieved source text, providing a direct citation to the source chapter.

## Tech Stack

- **Frontend:** Next.js 16, React 19, TailwindCSS, Lucide React, @xyflow/react (KG visualization), Chart.js (arena statistics)
- **Backend:** FastAPI (Python 3.10+), LangChain, Uvicorn
- **Vector Database:** ChromaDB (1024-dim, text-embedding-v4)
- **Knowledge Graph:** SymMap 2.0 via NetworkX + crosswalk bridge
- **Embeddings:** DashScope text-embedding-v4 (Alibaba Cloud, 1024 dimensions)
- **LLM Support:** Alibaba DashScope (Qwen), Google Gemini, OpenAI, Anthropic

## Current Status (Apr 2026)

The project is in active development approaching FYP presentation:

- ✅ Next.js 16 web interface and FastAPI backend are production-usable.
- ✅ 17 classical TCM texts ingested (3.72M characters, 11,522 chunks) with DashScope text-embedding-v4.
- ✅ Hybrid retrieval (vector + SymMap 2.0 KG) with jieba-enhanced entity matching.
- ✅ Arena blind A/B evaluation system for comparing RAG vs plain LLM responses.
- ✅ Chinese system prompt with 辨证论治 framework and cite-then-explain guidance.
- ✅ Multi-provider LLM support (Alibaba/Qwen, Gemini, OpenAI, Anthropic).
- 🔄 KG subgraph visualization and arena statistical analysis in progress.

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
A modern, interactive graph viewer powered by @xyflow/react renders subgraph neighborhoods around cited entities with dagre layout, allowing practitioners to explore relationships between symptoms, herbs, formulas, and related entities from the SymMap 2.0 knowledge graph.

### 📚 **Evidence-Based Answers**
All responses are backed by direct, verifiable citations from the 17-text classical corpus. The system quotes original text verbatim before explaining, and presents citations in a dedicated panel with full paragraph viewing and source reconstruction.

### ⚖️ **Arena Blind Evaluation**
A blind A/B comparison system where TCM practitioners evaluate RAG-enhanced responses against plain LLM responses without knowing which is which. Results are analyzed with paired T-Test for statistical proof of RAG effectiveness.

### 🌐 **Multi-Provider Support**
Seamlessly switch between Alibaba Cloud, Google, OpenAI, and Anthropic for maximum flexibility and availability.

## Configuration
See `docs/CONFIG.md` for detailed configuration options including retrieval parameters, model selection, and graph depth settings.
