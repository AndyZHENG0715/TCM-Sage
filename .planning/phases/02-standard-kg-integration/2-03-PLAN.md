---
phase: 2-standard-kg-integration
plan: 2-03
type: execute
wave: 3
depends_on: ["2-02"]
files_modified: ["src/config.py", "src/ui_backend.py", ".env.example"]
autonomous: false
requirements: [Task 2.3, Task 2.4]
must_haves:
  truths:
    - "SymMap KG is loaded by default in the retrieval pipeline"
    - "HybridRetriever fetches facts from SymMap KG correctly"
    - "UI visualization correctly displays SymMap entities and relationships"
  artifacts:
    - path: "src/config.py"
      provides: "Updated KG data path"
    - path: "src/ui_backend.py"
      provides: "SymMap integration"
---

<objective>
Update the project configuration and backend to activate the SymMap-based graph and verify the retrieval functionality.

Purpose: Finalize the pivot to the standard Knowledge Graph.
Output: An integrated system using SymMap 2.0 as the primary graph data source.
</objective>

<execution_context>
@$HOME/.gemini/get-shit-done/workflows/execute-plan.md
@$HOME/.gemini/get-shit-done/templates/summary.md
</execution_context>

<context>
@src/config.py
@src/ui_backend.py
@data/graph/symmap_entities.json
</context>

<tasks>

<task type="auto">
  <name>Task 2.3.1: Update configuration to use SymMap KG</name>
  <files>src/config.py, src/ui_backend.py, .env.example</files>
  <action>
    Update `src/config.py` to add `GRAPH_DATA_PATH = GRAPH_DIR / "symmap_entities.json"`.
    Update `src/ui_backend.py` to use the new path from `config.py` or through the environment variable `GRAPH_DATA_PATH`.
    Update `.env.example` to reflect the new default graph file.
  </action>
  <verify>Check `src/config.py` contains `symmap_entities.json` reference.</verify>
  <done>System is configured to use the SymMap dataset.</done>
</task>

<task type="auto">
  <name>Task 2.3.2: Verify HybridRetriever with SymMap facts</name>
  <files>src/ui_backend.py</files>
  <action>
    Test the `HybridRetriever` by performing a query that involves a SymMap entity (e.g., a specific symptom like "頭痛").
    Verify that the graph facts retrieved from `_search_graph_documents` are correct and correspond to SymMap data.
    Check the output of `src/ui_backend.py`'s `run_query` or use a test script `scripts/verify_symmap_retrieval.py` (if needed).
  </action>
  <verify>Run `python src/ui_backend.py` or equivalent and check retrieved facts match SymMap data.</verify>
  <done>HybridRetriever correctly fetches SymMap facts.</done>
</task>

<task type="checkpoint:human-verify">
  <what-built>Activated SymMap Knowledge Graph in UI</what-built>
  <how-to-verify>
    - Start the UI (e.g., Streamlit if applicable, or check the Next.js frontend if connected).
    - Perform a query related to TCM symptoms or herbs.
    - Inspect the `KGViewer` (Graph Visualization) in the `CitationPanel`.
    - Verify that nodes and edges displayed are from the SymMap dataset.
  </how-to-verify>
  <resume-signal>approved</resume-signal>
</task>

</tasks>

<verification>
Ensure the full pipeline (retrieval -> context formatting -> answering) works correctly with the SymMap dataset.
</verification>

<success_criteria>
The system successfully uses SymMap 2.0 as its Knowledge Graph source, and this is verifiable in the UI.
</success_criteria>

<output>
After completion, create `.planning/phases/02-standard-kg-integration/2-03-SUMMARY.md`
</output>
