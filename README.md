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

- **Frontend:** Next.js 16, React 19, TailwindCSS, Lucide React, @xyflow/react (KG visualization), Chart.js (arena statistics), react-chartjs-2
- **Backend:** FastAPI (Python 3.13), LangChain, Uvicorn
- **Vector Database:** ChromaDB (1024-dim, text-embedding-v4)
- **Knowledge Graph:** SymMap 2.0 via NetworkX + crosswalk bridge
- **Embeddings:** DashScope text-embedding-v4 (Alibaba Cloud, 1024 dimensions)
- **Reranker:** DashScope qwen3-rerank for retrieval re-scoring
- **Chinese NLP:** jieba segmentation for KG entity matching
- **LLM Support:** Alibaba DashScope (Qwen), Google Gemini, OpenAI, Anthropic

## Current Status (Apr 2026)

The project is in active development approaching FYP presentation:

- ✅ Next.js 16 web interface and FastAPI backend are production-usable.
- ✅ 17 classical TCM texts ingested (3.72M characters, 12,204 chunks) with DashScope text-embedding-v4.
- ✅ Clause-level chunking for 伤寒论 (398条) and 金匮要略 (489条) with contextual headers.
- ✅ Hybrid retrieval (vector + SymMap 2.0 KG) with jieba-enhanced entity matching.
- ✅ Reranker (qwen3-rerank) for improved retrieval relevance ordering.
- ✅ Arena blind A/B evaluation with T-Test statistical analysis and downloadable charts.
- ✅ Chinese system prompt with 辨证论治 framework, cite-then-explain, and explicit elaboration instructions.
- ✅ Markdown-formatted RAG context (Pattern Priming) producing 3.7x longer, well-structured answers.
- ✅ Multi-provider LLM support (Alibaba/Qwen, Gemini, OpenAI, Anthropic).
- ✅ Welcome modal, notice banner, Google Form feedback integration.
- ✅ KG subgraph explorer page (`/kg/[entityId]`) with interactive graph visualization.

## Setup and Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/AndyZHENG0715/TCM-Sage.git
    cd TCM-Sage
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
A blind A/B comparison system where TCM practitioners evaluate RAG-enhanced responses against plain LLM responses without knowing which is which. Includes a statistics page (`/arena/stats`) with paired T-Test analysis, win rate charts (downloadable as PNG), and per-query results table.

### 📊 **Arena Statistics & T-Test**
Live statistical analysis of arena votes with downloadable bar charts and pie charts. Computes t-statistic, p-value, Cohen's d effect size, and significance interpretation for FYP presentation.

### 🌐 **Multi-Provider Support**
Seamlessly switch between Alibaba Cloud, Google, OpenAI, and Anthropic for maximum flexibility and availability.

## Configuration
See [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) for detailed configuration options including provider setup, retrieval parameters, model selection, and graph depth settings.

## Project Structure

```
TCM-Sage/
├── src/                    # Python RAG core (FastAPI, LangChain, retriever, arena)
│   ├── api.py              # Thin FastAPI route layer (SSE, CORS, health)
│   ├── source_context.py   # Citation source context and raw book lookup
│   ├── arena_stream.py     # Arena dual-panel SSE scheduler
│   ├── arena_stats.py      # Arena vote statistics and t-test analysis
│   ├── main.py             # CLI entry point + LLM factory, prompts, classification
│   ├── retriever.py        # HybridRetriever — vector + graph ensemble
│   ├── graph_builder.py    # TCMKnowledgeGraph — NetworkX loader, traversal
│   ├── ingest.py           # Build vector index + chunks.json
│   ├── arena.py            # Arena blind A/B evaluation + vote storage
│   ├── embeddings.py       # DashScope text-embedding-v4 + qwen3-rerank
│   └── config.py           # Central paths and defaults
├── web/                    # Next.js 16 frontend
│   ├── app/                # Routes: chat, arena/, arena/stats, kg/, source/
│   ├── components/         # UI components (CitationPanel, KGViewer, etc.)
│   ├── hooks/              # React hooks (useChat, useArena, useSettings)
│   ├── i18n/               # Chinese/English UI translations
│   └── lib/                # API client, markdown renderer, shared types
├── data/
│   ├── source/             # 17 classical TCM .txt corpus (UTF-8)
│   ├── processed/          # chunks.json + ingest_checkpoint.json
│   ├── graph/symmap/       # SymMap v2.0 KG JSON + raw xlsx
│   ├── graph/crosswalk/    # RAG↔SymMap entity bridge (approved/pending CSV)
│   └── feedback/           # Arena votes (arena_votes.jsonl)
├── scripts/                # Utility and test scripts
├── vectorstore/            # ChromaDB persistence (generated)
├── presentation/           # Slidev FYP presentation
├── docs/                   # Project documentation
│   ├── ARCHITECTURE.md     # System architecture
│   ├── CONFIGURATION.md    # Configuration reference
│   ├── GETTING-STARTED.md  # First-time setup guide
│   ├── DEVELOPMENT.md      # Developer guide
│   ├── TESTING.md          # Testing guide
│   ├── API.md              # API reference
│   └── report/             # FYP Final Report (LaTeX + PDF)
└── requirements.txt        # Python dependencies
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
