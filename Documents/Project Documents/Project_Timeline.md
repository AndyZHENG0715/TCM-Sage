# TCM-Sage: Detailed Project Plan & Milestones

This document outlines the official timeline, key stages, and deliverables for the TCM-Sage project, incorporating feedback from the First Progress Report review.

## Phase 1: Project Definition & Foundational Research (Complete)

* **Duration:** April 2025 - September 2025

* **Objective:** Define the project scope, conduct foundational research, and prepare the initial data pipeline.

* **Official Deadlines Met:**

    - `Apr 28`: Project topic preference form submitted.

    - `Jun 9`: Formal project statement submitted.

    - `Jun 23`: Formal project statement approved.

    - `Sep 26`: First Progress Report submitted.

* **Key Achievements:**

    - [x] Completed literature review on RAG, Self-RAG, and KG-RAG.

    - [x] Defined MVP and System Architecture (`MVP_and_Architecture.md`).

    - [x] Established development environment and data ingestion pipeline (`src/ingest.py`).

    - [x] Sourced, cleaned, and processed the Huangdi Neijing into a searchable vector store.

## Phase 2: MVP Implementation & Core Logic (Current Stage)

* **Duration:** October 2025 - December 2025

* **Objective:** Develop a functional command-line MVP that can retrieve information and generate evidence-backed answers.

* **Official Deadlines:**

    - `End of Nov`: First monthly source code submission.

    - `End of Dec`: Second monthly source code submission.

* **Key Tasks:**

    - [ ] (Immediate Next Step) Update ingest.py to include source metadata (chapter names) for accurate citations.

    - [x] Develop the core application script (`src/main.py`) to orchestrate the RAG pipeline.

    - [x] Implement the retriever module to perform similarity searches on the vector store.

    - [x] Implement multi-provider LLM support with Alibaba Cloud Model Studio integration (1M free tokens).

    - [ ] Implement the two-layer Reflective Generator:

        - Build the query-routing mechanism using a small model.

        - Construct the prompt templates for the main generator to perform self-critique and cite sources.

    - [ ] Build a simple Command-Line Interface (CLI) to interact with the MVP.

## Phase 3: Mid-Point Review & Enhancement

* **Duration:** January 2026 - February 2026

* **Objective:** Prepare for the mid-point presentation, demonstrate the MVP, and begin enhancing the system with the Knowledge Graph.

* **Official Deadlines:**

    - `Jan 5`: Second Progress Report submitted.

    - `Jan 7-9`: Mid-point presentation and demonstration.

    - `End of Jan`: Third monthly source code submission.

    - `End of Feb`: Fourth monthly source code submission.

* **Key Tasks:**

    - [ ] Prepare slides and documentation for the mid-point review.

    - [ ] Demonstrate the functional CLI-based MVP.

    - [ ] Begin development of the Knowledge Graph (KG) by designing the schema and writing extraction scripts.

    - [ ] Upgrade the retriever to a Hybrid Retriever that queries both the vector store and the KG.

    - [ ] Begin development of a simple web-based user interface (e.g., using Streamlit).

## Phase 4: Pilot Testing & System Evaluation

* **Duration:** March 2026

* **Objective:** Formally evaluate the system against defined metrics and gather real-world user feedback.

* **Official Deadlines:**

    - `End of Mar`: Final monthly source code submission.

* **Key Tasks:**

    - [ ] Finalize the web-based UI for testing.

    - [ ] Conduct internal performance tests to measure Response Latency, Citation Precision, and Answer Faithfulness as defined in Evaluation_Plan.md.

    - [ ] (Address Supervisor Feedback) Recruit 3-5 pilot testers from the HKBU School of Chinese Medicine.

    - [ ] (Address Supervisor Feedback) Conduct structured pilot testing sessions to gather qualitative feedback and quantitative scores for User Trust and Usability.

    - [ ] Analyze feedback and make final refinements to the system.

## Phase 5: Final Submission & Presentation

* **Duration:** April 2026

* **Objective:** Complete all project deliverables for final submission.

* **Official Deadlines:**

    - `Apr 8`: FYP Report for grading submitted.

    - `Apr 10-16`: Oral presentation and final demonstration.

    - `Apr 23`: Submit complete project archive.

* **Key Tasks:**

    - [ ] Write the complete final report.

    - [ ] Create the poster and presentation slides.

    - [ ] Record the final demonstration video.

    - [ ] Compile and submit the final project archive, including all code, reports, and documentation.
