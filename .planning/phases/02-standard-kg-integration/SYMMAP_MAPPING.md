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
