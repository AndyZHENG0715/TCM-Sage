# SymMap 2.0 → TCM-Sage Knowledge Graph Mapping

**Reference:** Wu Y, Zhang F, Yang K, et al. *SymMap: an integrative database of traditional Chinese medicine enhanced by symptom mapping.* Nucleic Acids Research. 2019;47(D1):D1110-D1117. [NAR](https://academic.oup.com/nar/article/47/D1/D1110/5150228)

**Official site:** [http://www.symmap.org](http://www.symmap.org) — browse, search, and bulk download.

---

## Dataset Structure

### Download sources and formats

SymMap distributes **six component types** plus **pairwise relationship** tables. Data are provided from the [download page](http://www.symmap.org/download/) (see *file format* section for column glossaries). Typical delivery is **tab-separated text** or **Excel** (`.xls`/`.xlsx`) exported to CSV for ingestion; encoding is usually **UTF-8**.

| SymMap prefix | Component | Typical SymMap ID pattern | Role |
|---------------|-----------|----------------------------|------|
| **SMHB** | Herb (Chinese medicine) | `SMHBxxxxx` | Medicinal herbs |
| **SMYS** | Syndrome (TCM pattern) | `SMYSxxxxx` | TCM syndromes / pattern names |
| **SMTS** | TCM symptom | `SMTSxxxxx` | Symptoms as used in classical TCM |
| **SMMS** | Modern-medicine symptom | `SMMSxxxxx` | Symptoms aligned to modern/clinical terminology |
| **SMIT** | Ingredient (compound) | `SMITxxxxx` | Chemical / active ingredients in herbs |
| **SMTT** | Target (gene/protein) | `SMTTxxxxx` | Molecular targets |
| **SMDE** | Disease | `SMDExxxxx` | Modern disease terms |

**Key files** (search-term key lists) are also published per component; IDs in those files match the main entity tables.

### Entity table columns (expected / typical)

Exact header names can vary slightly by export; the adapter normalizes common aliases. Typical fields:

| Component | Representative columns (examples) |
|-----------|-------------------------------------|
| **SMTS** (TCM symptom) | SymMap ID, Chinese name, Pinyin, English name / label, optional description, external IDs |
| **SMMS** (MM symptom) | SymMap ID, preferred name, synonyms, UMLS or other cross-refs |
| **SMHB** (Herb) | SymMap ID, Chinese name, Pinyin, Latin name, English name, property / meridian / function text, cross-database IDs (e.g. TCMID, TCMSP) |
| **SMIT** (Ingredient) | SymMap ID, compound name, identifiers (e.g. PubChem), structure-related fields where present |
| **SMTT** (Target) | SymMap ID, gene symbol, protein name, organism |
| **SMDE** (Disease) | SymMap ID, disease name, vocabulary IDs (e.g. MeSH, DOID) where present |
| **SMYS** (Syndrome) | SymMap ID, Chinese / English syndrome name, descriptive text |

### Example header patterns (legacy SM/HM-style ↔ SymMap 2.0)

Papers and older dumps often use **SM** (symptom), **HM** (herb), **IM** (ingredient), **TM** (target), **MM** (disease). SymMap 2.0 bulk files typically use **prefix-based IDs** and headers such as:

| Layer (legacy) | SymMap 2.0 prefix | Example ID column | Example name / text columns |
|----------------|-------------------|--------------------|-----------------------------|
| SM (TCM symptom) | SMTS | `SMTS_ID` / `SymMap_ID` | `SMTS_Chinese_Name`, `SMTS_Pinyin`, `SMTS_English_Name` |
| SM (MM symptom) | SMMS | `SMMS_ID` | `SMMS_Name`, synonym / UMLS columns |
| HM (herb) | SMHB | `SMHB_ID` | `SMHB_Chinese_Name`, `SMHB_Pinyin`, `SMHB_Latin_Name` |
| IM (ingredient) | SMIT | `SMIT_ID` | `SMIT_Name`, PubChem / InChI where present |
| TM (target) | SMTT | `SMTT_ID` | `SMTT_Gene_Symbol`, `SMTT_Protein_Name` |
| MM (disease) | SMDE | `SMDE_ID` | `SMDE_Name`, MeSH / DOID columns |
| Syndrome | SMYS | `SMYS_ID` | `SMYS_Chinese_Name`, `SMYS_English_Name` |

Relationship files are usually pairwise ID columns (e.g. `SMTS_ID` + `SMHB_ID`) or synonym pairs using the same prefixes; **TSV/CSV** with header row is the common bulk format, with **Excel** exports on the download site.

### Relationship tables (pairwise)

SymMap exposes **direct** associations between adjacent entity types (e.g. herb–ingredient, symptom–herb). **Indirect** links (e.g. TCM symptom–disease) may be provided as separate inference tables with **p-value / FDR** columns; the adapter can ingest direct edges by default and optionally filter inferred edges by FDR thresholds.

| Relationship (informal) | Typical endpoints | Notes |
|-------------------------|-------------------|--------|
| TCM symptom ↔ Herb | SMTS ↔ SMHB | Clinical / literature-supported links |
| Herb ↔ Ingredient | SMHB ↔ SMIT | Composition |
| Ingredient ↔ Target | SMIT ↔ SMTT | Molecular action |
| Target ↔ Disease | SMTT ↔ SMDE | Association / relevance |
| Syndrome ↔ Symptom / Herb | SMYS ↔ SMTS / SMHB | Pattern-level links |
| MM symptom ↔ TCM symptom / disease | SMMS ↔ SMTS, SMMS ↔ SMDE | Bridging tables |

**Legacy shorthand (papers / older docs):** SM = symptom, HM = herb, IM = ingredient, TM = target, MM = disease — maps to **SMTS/SMMS, SMHB, SMIT, SMTT, SMDE** above.

---

## Schema Mapping

### Entity mapping (SymMap → `TCMKnowledgeGraph` node `type`)

| SymMap component | Prefix | `TCMKnowledgeGraph` entity `type` | Notes |
|------------------|--------|-----------------------------------|-------|
| TCM symptom | SMTS | `Symptom` | Primary symptom nodes for user queries (e.g. 頭痛) |
| MM symptom | SMMS | `Symptom` | Same type; use `symmap_component: "SMMS"` in attributes when needed |
| Herb | SMHB | `Herb` | |
| Ingredient | SMIT | `Ingredient` | Requires `Ingredient` in `ENTITY_TYPES` |
| Target | SMTT | `Target` | Gene/protein |
| Disease | SMDE | `Disease` | Modern disease |
| Syndrome | SMYS | `Syndrome` | Distinct from single symptoms |

**Node attributes:** Prefer `name` (Chinese where applicable), `name_en`, `pinyin`, `symmap_id`, `symmap_component`, `description`, and optional `source_ref` for provenance.

### Relationship mapping (SymMap → edge `type`)

| SymMap link (intuitive) | Edge `type` | Direction (source → target) |
|---------------------------|-------------|-----------------------------|
| Herb treats / used for TCM symptom | `TREATS` | Herb → Symptom |
| Symptom indicates pattern (if modeled) | `INDICATES` | Symptom → Syndrome |
| Herb contains ingredient | `CONTAINS` | Herb → Ingredient |
| Ingredient acts on target | `TARGETS` | Ingredient → Target |
| Target linked to disease | `ASSOCIATED_WITH` | Target → Disease |
| Symptom–disease (inferred or direct) | `CORRELATES_WITH` | Symptom → Disease |
| MM symptom ↔ TCM symptom | `MAPS_TO` | SMMS → SMTS |

**JSON export shape** (matches `load_from_json`):

```json
{
  "entities": [{ "id": "...", "type": "...", "name": "...", "name_en": "...", ... }],
  "relationships": [{ "source": "...", "target": "...", "type": "TREATS", "description": "...", "source_ref": "SymMap" }]
}
```

### Compatibility

`src/graph_builder.py` **`ENTITY_TYPES`** and **`RELATIONSHIP_TYPES`** include the SymMap-oriented values above so programmatic `add_entity` / `add_relationship` calls stay valid. **`load_from_json`** remains permissive for forward compatibility.
