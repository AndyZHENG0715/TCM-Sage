---
phase: 2-standard-kg-integration
plan: 2-02
type: execute
wave: 2
depends_on: ["2-01"]
files_modified: ["scripts/import_symmap_kg.py", "data/graph/symmap_entities.json"]
autonomous: true
requirements: [Task 2.2]
must_haves:
  truths:
    - "SymMap data is successfully parsed and converted to entities.json format"
    - "New KG data exists as a JSON file"
  artifacts:
    - path: "scripts/import_symmap_kg.py"
      provides: "Data adapter script"
    - path: "data/graph/symmap_entities.json"
      provides: "New standard Knowledge Graph"
---

<objective>
Develop the data adapter script to parse SymMap data and export it into our project's graph format.

Purpose: Automate the transformation from standard research data to our in-memory graph.
Output: A new graph file `symmap_entities.json` based on the SymMap 2.0 dataset.
</objective>

<execution_context>
@$HOME/.gemini/get-shit-done/workflows/execute-plan.md
@$HOME/.gemini/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/phases/02-standard-kg-integration/SYMMAP_MAPPING.md
@src/graph_builder.py
</context>

<tasks>

<task type="auto">
  <name>Task 2.2.1: Implement SymMap Data Adapter script</name>
  <files>scripts/import_symmap_kg.py</files>
  <action>
    Create `scripts/import_symmap_kg.py`. This script should:
    - Use the mapping defined in `SYMMAP_MAPPING.md`.
    - Accept a directory of SymMap source files (CSVs/TSVs) as input.
    - Parse each file to extract entities (Symptoms, Herbs, etc.) and relationships.
    - Handle duplicates and standard mapping (e.g., entity IDs).
    - Save the resulting graph as `data/graph/symmap_entities.json` in the expected format: `{"entities": [...], "relationships": [...]}`.
  </action>
  <verify>Script exists and is executable via `python scripts/import_symmap_kg.py --help`.</verify>
  <done>Adapter script is implemented.</done>
</task>

<task type="auto">
  <name>Task 2.2.2: Generate symmap_entities.json</name>
  <files>data/graph/symmap_entities.json</files>
  <action>
    Run the adapter script on (real or mock) SymMap data to generate the Knowledge Graph.
    Ensure it generates `data/graph/symmap_entities.json` with a substantial number of nodes and edges (e.g., >100).
    Verify the JSON structure matches what `src/graph_builder.py` expects.
  </action>
  <verify>Check `data/graph/symmap_entities.json` exists and contains 'entities' and 'relationships' keys.</verify>
  <done>Knowledge Graph JSON file is generated.</done>
</task>

</tasks>

<verification>
Verify `data/graph/symmap_entities.json` is a valid JSON and can be loaded by `TCMKnowledgeGraph.load_from_json()`.
</verification>

<success_criteria>
A valid `symmap_entities.json` file exists in `data/graph/`.
</success_criteria>

<output>
After completion, create `.planning/phases/02-standard-kg-integration/2-02-SUMMARY.md`
</output>
