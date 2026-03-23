# Project Context

## Purpose

**TCM-Sage** is an evidence-synthesis tool for Traditional Chinese Medicine (TCM) practitioners. The project aims to empower practitioners by providing explainable, evidence-backed insights from the vast corpus of classical TCM literature using a Retrieval-Augmented Generation (RAG) architecture.

**Goals:**

- Enable practitioners to query the entire corpus of TCM literature in seconds
- Provide explainable, evidence-backed answers with direct citations to source texts
- Support clinical decision-making with intelligent query classification (informational vs. prescriptive)
- Maintain strict accuracy for medical/prescriptive queries through temperature routing

## Tech Stack

### Core

- **Python 3.x** - Primary programming language
- **LangChain** - RAG orchestration framework (langchain, langchain-core, langchain-community)
- **ChromaDB** - Persistent vector store for semantic search
- **Sentence Transformers** - Embeddings generation (`all-MiniLM-L6-v2`)

### LLM Providers (Multi-Provider Support)

- Alibaba Cloud Model Studio (DashScope) - Default/Recommended
- OpenAI (GPT-4o, GPT-4o-mini)
- Google AI Studio (Gemini 2.5 Pro)
- Anthropic (Claude 3.5 Sonnet)
- OpenRouter, Together AI

### UI

- **Streamlit** - Prototype UI for demos (`src/ui_app.py`)
- **CLI** - Primary interface (`src/main.py`)

### Supporting Libraries

- `python-dotenv` - Environment variable management
- `pydantic` / `pydantic-settings` - Configuration validation
- `torch` / `transformers` - ML model support

## Project Conventions

### Code Style

- **Standard**: PEP 8 (Python official style guide)
- **Docstrings**: Google-style docstrings with Args, Returns, and Raises sections
- **Type Hints**: Use type hints for function parameters and return values
- **Naming**:
  - `snake_case` for functions and variables
  - `PascalCase` for classes
  - `UPPER_SNAKE_CASE` for constants
- **Imports**: Group imports by standard library, third-party, then local modules

### Architecture Patterns

- **Modular RAG Paradigm**: Separation of ingestion, retrieval, and generation concerns
- **Provider Abstraction**: Factory pattern for LLM instantiation (`create_llm()`)
- **Query Routing**: Classifier-based routing for severity-appropriate response generation
- **Glass-box Design**: Transparent answer generation with mandatory source citations
- **Environment-based Configuration**: All config via `.env` files, never hardcoded

### Testing Strategy

- **Retriever Testing**: `src/test_retriever.py` for vector store verification
- **Manual Testing**: Interactive CLI loop for end-to-end testing
- **Planned Evaluation** (Phase 4):
  - Quantitative: Latency, Precision, Faithfulness metrics
  - Qualitative: Pilot testing with TCM practitioners

### Git Workflow

- **Environment Files**: Never commit `.env` files (use `.env.example` as template)
- **Commit Messages**: Descriptive, present-tense commit messages
- **Branching**: Feature branches for new capabilities

## Domain Context

This project operates in the **Traditional Chinese Medicine (TCM)** domain. Key knowledge:

- **Primary Source**: Huangdi Neijing (黃帝内經) - foundational classical text
- **Language**: Classical Chinese text with bilingual support (Chinese/English queries)
- **Terminology**: Specialized TCM concepts (陰陽, 五行, 經絡, etc.) require precise handling
- **Query Types**:
  - **Informational**: "What is 陰陽?" → General knowledge, higher temperature acceptable
  - **Prescriptive**: "How to treat 頭痛?" → Medical advice, zero temperature required

## Important Constraints

### Technical

- Vector store must be pre-built via `src/ingest.py` before running main app
- Embeddings model must match between ingestion and retrieval (`all-MiniLM-L6-v2`)
- Windows platform requires UTF-8 encoding fixes for Chinese text output

### Academic/Timeline

- Academic project with defined submission deadlines (see README roadmap)
- Phase 2 (MVP) complete; Phases 3-5 pending (Jan 2026 - Apr 2026)

### Regulatory/Ethical

- System is a **clinical reference assistant**, NOT a decision-maker
- All medical queries require strict accuracy (zero temperature)
- Answers must include source citations for verification

## External Dependencies

### LLM APIs

| Provider | API Key Variable | Default Model |
|----------|------------------|---------------|
| Alibaba Cloud | `DASHSCOPE_API_KEY` | `qwen3-14b` |
| OpenAI | `OPENAI_API_KEY` | `gpt-4o` |
| Google | `GOOGLE_API_KEY` | `gemini-2.5-pro` |
| Anthropic | `ANTHROPIC_API_KEY` | `claude-3-5-sonnet-20241022` |
| OpenRouter | `OPENROUTER_API_KEY` | `openai/gpt-4o` |
| Together AI | `TOGETHER_API_KEY` | `meta-llama/Llama-3.1-8B-Instruct-Turbo` |

### Local Resources

- ChromaDB vector store at `vectorstore/chroma/`
- Source texts in `data/` directory
- Research documents in `research/` directory
