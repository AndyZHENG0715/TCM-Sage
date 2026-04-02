---
phase: 2-standard-kg-integration
plan: 2-01
type: execute
wave: 1
depends_on: []
files_modified: [".planning/phases/02-standard-kg-integration/SYMMAP_MAPPING.md"]
autonomous: true
requirements: [Task 2.1]
must_haves:
  truths:
    - "SymMap 2.0 schema is mapped to NetworkX entity/relationship types"
    - "Download sources and file formats are documented"
  artifacts:
    - path: ".planning/phases/02-standard-kg-integration/SYMMAP_MAPPING.md"
      provides: "Data mapping schema"
---

<objective>
Research and map the SymMap 2.0 dataset to our internal NetworkX-based Knowledge Graph schema.

Purpose: To ensure the academically recognized SymMap dataset can be correctly ingested into our system.
Output: A mapping document defining the conversion between SymMap files and our `entities.json` structure.
</objective>

<execution_context>
@$HOME/.gemini/get-shit-done/workflows/execute-plan.md
@$HOME/.gemini/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/research/STACK.md
@src/graph_builder.py
</context>

<tasks>

<task type="auto">
  <name>Task 2.1.1: Research SymMap 2.0 file structures</name>
  <files>.planning/phases/02-standard-kg-integration/SYMMAP_MAPPING.md</files>
  <action>
    Research the SymMap 2.0 dataset (e.g., Wu et al., 2019) to identify the specific file names and column structures for:
    - Symptoms (SM)
    - Herbs (HM)
    - Ingredients (IM)
    - Targets (TM)
    - Diseases (MM)
    - Relationships (e.g., SM-HM, SM-MM, HM-IM, IM-TM, TM-MM)

    Document the expected file formats (CSV/TSV/SQL) and their core columns (e.g., SM_ID, SM_Name, SM_Pinyin).
  </action>
  <verify>Check `.planning/phases/02-standard-kg-integration/SYMMAP_MAPPING.md` contains a 'Dataset Structure' section.</verify>
  <done>SymMap 2.0 file structure is documented.</done>
</task>

<task type="auto">
  <name>Task 2.1.2: Map SymMap to TCMKnowledgeGraph schema</name>
  <files>.planning/phases/02-standard-kg-integration/SYMMAP_MAPPING.md, src/graph_builder.py</files>
  <action>
    Define how each SymMap entity and relationship maps to the `TCMKnowledgeGraph` schema:
    - Entity mapping (e.g., SM -> Symptom, HM -> Herb, MM -> Disease)
    - Relationship mapping (e.g., SM-HM -> TREATS or INDICATES)
    - Attribute mapping (e.g., Pinyin, English name, Description)

    Check if `src/graph_builder.py` needs additional `ENTITY_TYPES` or `RELATIONSHIP_TYPES` (e.g., "Disease", "Ingredient", "Target", "MAPS_TO"). If so, update the constants in `src/graph_builder.py`.
  </action>
  <verify>
    - `SYMMAP_MAPPING.md` contains a 'Schema Mapping' section.
    - `src/graph_builder.py` constants are updated if needed.
  </verify>
  <done>SymMap mapping is clearly defined and `graph_builder.py` constants are compatible.</done>
</task>

</tasks>

<verification>
Ensure the mapping covers the core "Symptom, Herbs, TCM-Modern Medicine mappings" mentioned in the project instructions.
</verification>

<success_criteria>
A comprehensive mapping document exists that can be used by the adapter script in Wave 2.
</success_criteria>

<output>
After completion, create `.planning/phases/02-standard-kg-integration/2-01-SUMMARY.md`
</output>
