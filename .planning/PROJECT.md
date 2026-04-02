# TCM-Sage: Stabilization & Standard KG Integration

## Overview
This phase of the TCM-Sage Final Year Project (FYP) focuses on finalizing the core MVP for the presentation deadline (April 13th, 2026). It transitions the project from a volatile prototype to a stable, academically defensible application by replacing the LLM-extracted Knowledge Graph with a standard, recognized TCM dataset and resolving outstanding frontend technical debt.

## Goals
1. **System Stabilization**: Resolve known UI bugs (markdown parsing, duplicate sources) and implement the full-text context toggle.
2. **Academic Rigor**: Integrate a reputable, publicly recognized TCM Knowledge Graph (e.g., TCMID, SymMap, or a public academic KG dataset) to ensure defensible results for the FYP marker.
3. **KG Visualization**: Provide an interactive visual representation of the Knowledge Graph citations using React Flow.

## Constraints & Assumptions
- **Deadline**: April 13th, 2026 (Final Presentation).
- **Architecture**: Keep the existing Next.js frontend and FastAPI/LangChain backend.
- **Graph Storage**: Can use in-memory (NetworkX) or lightweight local DB, depending on the chosen KG size.

## References
- Previous architecture definitions and MVP scoping.
- Lecturer feedback regarding KG validity.
