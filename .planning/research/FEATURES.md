# Feature Landscape: TCM Datasets

**Domain:** Traditional Chinese Medicine (TCM) Informatics
**Researched:** March 2026

## Table Stakes (Core Features)

Features expected in any defensible TCM dataset for an FYP.

| Feature | Why Expected | Content Scope | Notes |
|---------|--------------|---------------|-------|
| **Herb Entities** | Fundamental TCM unit. | English/Latin/Chinese names. | Must include properties (e.g., Cold/Hot). |
| **Compounds / Ingredients** | Molecular basis of TCM. | CAS numbers, SMILES, MW. | Crucial for molecular docking/docking analysis. |
| **Biological Targets** | Mechanisms of action. | Gene Symbols, UniProt IDs. | Essential for network pharmacology. |
| **Diseases** | Clinical utility. | ICD-10/ICD-11, MeSH. | Linking traditional treatments to western diseases. |

## Differentiators (Advanced Features)

Features that elevate an FYP from "simple mapping" to "advanced research".

| Feature | Value Proposition | Complexity | Recommended Dataset |
|---------|-------------------|------------|---------------------|
| **Symptom Mapping** | Links "Zheng" (TCM Syndromes) to MM Phenotypes. | High | **SymMap** |
| **Prescription (Formulas)** | Models herbal compatibility (Jun-Chen-Zuo-Shi). | Medium | **TCMID** (via TCM-MKG) |
| **ADME Properties** | OB (Oral Bioavailability) and DL (Drug-likeness). | Low | **TCMSP** |
| **Multilingual Mapping** | Maps between Pinyin and Latin names. | Medium | **TCM-MKG** |

## Anti-Features

Features/datasets to avoid for an FYP.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| **Raw Scraped Text** | Too much noise; lacks academic defensibility. | Use standardized KGs (TCM-MKG). |
| **PDF-only Datasets** | Incredibly difficult to integrate into a Python stack. | Use Zenodo CSV/TSV mirrors. |
| **Static Websites** | Many TCM sites are from early 2010s and unstable. | Rely on recent publications (2024-2025). |

## Feature Dependencies

```
Herb -> Ingredient (Contains)
Ingredient -> Target (Binds/Regulates)
Target -> Disease (Involved In)
Symptom -> TCM Syndrome (Associated With)
```

## MVP Recommendation (FYP Strategy)

For a defensible FYP, focus on **one** clear research angle:

1.  **Angle 1: Network Pharmacology.** Focus on **TCMSP**. (Herb-Ingredient-Target-Disease).
2.  **Angle 2: Clinical Phenotypic Analysis.** Focus on **SymMap**. (Symptom-Syndrome-Western Disease).
3.  **Angle 3: AI / Knowledge Graph Construction.** Focus on **TCM-MKG**. (Comprehensive Integration).

**Recommended Deferral:** Scraping own data. It takes too much time and is less defensible than using established datasets.
