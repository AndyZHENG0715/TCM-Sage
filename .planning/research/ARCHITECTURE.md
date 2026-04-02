# TCM Knowledge Graph Architecture

**Domain:** Traditional Chinese Medicine (TCM) Informatics
**Researched:** March 2026

## Recommended Knowledge Graph Schema (TCM-MKG Style)

For a defensible FYP, the Knowledge Graph (KG) should follow a **triple-based structure** (Subject-Predicate-Object).

### Component Boundaries

| Component | Responsibility | Examples |
|-----------|---------------|----------|
| **Herb Node** | Identity and Properties | English: *Ginseng Radix et Rhizoma*, Property: *Warm* |
| **Compound Node** | Molecular Data | SMILES, MW, LogP, CAS |
| **Target Node** | Biological Action | UniProt: *P35869*, Symbol: *AHR* |
| **Disease Node** | Clinical Phenotype | ICD-11, MeSH, MeSH: *D009369* (Neoplasms) |
| **Symptom Node** | Clinical Pattern | TCM: *Dry mouth*, MM: *Xerostomia* |

### Data Flow (Integration Pipeline)

1.  **Extraction:** Download CSVs/Triples from Zenodo (TCM-MKG).
2.  **Transformation:** Clean/Normalize entity IDs (e.g., mapping Pinyin names to standardized Latin IDs).
3.  **Loading:** Import into **NetworkX** (Python) for quick prototyping or **Neo4j** for persistence.
4.  **Inference:** Run queries (e.g., "Find all herbs containing ingredients that target PTGS2").

## Patterns to Follow

### Pattern 1: Meta-Mapping (The "SymMap" Pattern)
**What:** Bridge TCM Symptoms to MM Phenotypes through a common Ingredient/Target layer.
**Example:**
```cypher
MATCH (s:TCMSymptom)-[:MAPPED_TO]->(m:MMSymptom)
MATCH (m)-[:ASSOCIATED_WITH]->(d:Disease)
RETURN s, d
```

## Anti-Patterns to Avoid

### Anti-Pattern 1: Flattened Databases
**What:** Storing all properties in a single CSV row.
**Why bad:** Loses relational context (e.g., an ingredient can belong to many herbs).
**Instead:** Use a normalized graph schema.

## Scalability Considerations

| Concern | Small Graph (NetworkX) | Large Graph (Neo4j) |
|---------|------------------------|---------------------|
| **10k Nodes** | Fast (RAM-bound). | Fast (Persistence overhead). |
| **1M Nodes** | Slow/Heavy RAM. | Optimized indexing. |

## Sources

- **TCM-MKG Schema:** [Zeng et al. 2025](https://doi.org/10.1016/j.jpha.2025.101342)
- **Graph Import Patterns:** [Neo4j Documentation](https://neo4j.com/docs/)
