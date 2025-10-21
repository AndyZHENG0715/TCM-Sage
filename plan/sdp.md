# TCM-Sage: Software Development Plan

** Version:** 1.0
** Date:** October 17, 2025

## 1. Introduction

This document provides a detailed technical overview and implementation plan for the TCM-Sage project. It is intended for developers and contributors to understand the system architecture, technology stack, data pipeline, and development roadmap. It complements the high-level summary in `README.md`.

## 2. Technology Stack

The project relies on the following core technologies and libraries:

* **Language:** Python 3.10+

* **Core Framework:** LangChain

* **Embedding Model:** `all-MiniLM-L6-v2` (via SentenceTransformers)

* **Vector Store:** ChromaDB (local, persistent storage)

* **Primary LLM (Generator):** OpenAI `gpt-4o` (or similar high-capability model)

* **Routing LLM (Classifier):** A smaller, faster model like `Gemini 2.5 Flash` or a local `Llama-3-8B-Instruct`.

* **LangChain Integration:** `langchain-openai` for OpenAI API integration

* **Environment Management:** venv

* **Package Management:** `pip` and `requirements.txt`

## 3. Project Structure

The repository is organized to separate data, source code, and documentation.

```bash
tcm-sage/                             # Root directory
│
├── data/
│   ├── source/
│   │   └── huangdi_neijing.txt      # Raw, original text file
│   ├── cleaned/
│   │   └── cleaned_huangdi_neijing.txt # Programmatically cleaned text
│   └── processed/
│       └── chunks.json                # Final, structured chunks with metadata
│
├── notebooks/                         # Jupyter notebooks for experimentation (e.g., KG development)
│
├── research/                          # Literature review and summary notes
│
├── src/
│   ├── ingest.py                    # Data pipeline: Clean -> Chunk -> Embed -> Store
│   ├── test_retriever.py            # Utility script to verify the vector store
│   └── main.py                      # Core application logic (RAG chain)
│
├── vectorstore/
│   └── chroma/                        # Persisted ChromaDB database files
│
├── .env                             # For storing API keys (not committed to Git)
├── .gitignore                       # Standard Python .gitignore
├── DEVELOPMENT_PLAN.md              # This document
├── Evaluation_Plan.md               # Metrics and pilot testing strategy
├── LICENSE
├── MVP_and_Architecture.md
├── Project_Plan.md                  # High-level project timeline
└── README.md                          # Main project overview and setup guide
```

## 4. System Architecture & Data Flow

### 4.1 Data Ingestion Pipeline (`ingest.py`)

This is an offline, one-time process.

1. **Load Raw Text:** Reads `data/source/huangdi_neijing.txt`.

2. **Clean Text:** Programmatically removes modern translations (`参考译文`), table of contents, and formatting artifacts. Saves to `data/cleaned/`.

3. **Split by Chapter:** The cleaned text is split into a list, where each item is the full text of one chapter.

4. **Chunk with Metadata:** The script iterates through each chapter.

    - It extracts the chapter title (e.g., "上古天真论篇第一").

    - It uses `RecursiveCharacterTextSplitter` to chunk the chapter's content.

    - For each chunk, it attaches the chapter title as a `metadata` field (e.g., `{"source": "上古天真论篇第一"}`).

5. **Store Chunks:** The final list of structured chunks is saved to `data/processed/chunks.json`.

6. **Embed & Persist:**

    - The content of each chunk is fed into the `all-MiniLM-L6-v2` embedding model.

    - The resulting vectors, along with their corresponding text and metadata, are saved to the persistent ChromaDB vector store in the `vectorstore/chroma` directory.

### 4.2 Core Application Logic (`main.py`)

This is the online, real-time process that runs when a user submits a query.

1. **Load Knowledge Base:** The application starts by loading the persistent ChromaDB vector store and the embedding model into memory.

2. **Instantiate Retriever:** A retriever object is created from the vector store (`vectorstore.as_retriever()`).

3. **User Query:** The system accepts a user's question.

4. **Retrieval:** The retriever uses the embedding model to convert the user's question into a vector and performs a similarity search in ChromaDB, returning the top-k most relevant text chunks (including their metadata).

5. **Chain Execution (LangChain Expression Language):**

    - The retrieved documents are formatted into a single string of context.

    - This context, along with the original user question, is inserted into a predefined prompt template.

    - The complete prompt is sent to the LLM (e.g., `gpt-4o`).

    - The LLM's response is captured by an output parser.

6. **Output:** The final, evidence-backed answer is returned to the user.

## 5. Phase 2 Implementation Plan

This section details the concrete steps for building the MVP as outlined in the `Project_Plan.md`.

### Task 2.1: Implement the Core RAG Chain (`src/main.py`) ✅ COMPLETED
* **Objective:** Create a runnable script that executes the full RAG pipeline for a single, hardcoded query.

* **Sub-tasks:**

    - [x] Create `src/main.py`.

    - [x] Implement loading of the vector store and retriever.

    - [x] Define the prompt template that instructs the LLM to use the context and cite sources.

    - [x] Instantiate the LLM (`ChatOpenAI`).

    - [x] Construct the chain using LangChain Expression Language (LCEL).

    - [x] Add a simple `if __name__ == "__main__":` block to invoke the chain with a test query (e.g., "`阴阳是什么？`") and print the result.

* **Implementation Details:**
    - Added comprehensive error handling for missing API keys and vector store files
    - Configured retriever with k=5 for optimal context retrieval
    - Implemented proper document formatting with source metadata
    - Set model temperature to 0.1 for factual, evidence-based responses

### Task 2.2: Build a Functional Command-Line Interface (CLI)

* **Objective:** Turn `main.py` into an interactive application.

* **Sub-tasks:**

    - [ ] Add a `while True:` loop to the main function to continuously prompt the user for input.

    - [ ] Use Python's `input()` function to accept a query from the command line.

    - [ ] Add logic to exit the loop (e.g., if the user types "exit" or "quit").

    - [ ] Format the output nicely in the console, clearly separating the "Answer" from the "Sources".

### Task 2.3: Implement Query Routing (Controllable Inference)

* **Objective:** Add the first layer of the "Reflective Generator."

* **Sub-tasks:**

    - [ ] Create a new function, e.g., `get_query_severity(query)`.

    - [ ] Inside this function, define a simple prompt that asks a smaller LLM (`Gemini 2.5 Flash`) to classify the query as either "informational" or "prescriptive".

    - [ ] In the main RAG chain logic, call this function first.

    - [ ] Based on the result, instantiate the main LLM (`ChatOpenAI`) with either a default temperature (e.g., 0.7) or a temperature of 0.0.

This document will be updated as the project progresses through its phases.
