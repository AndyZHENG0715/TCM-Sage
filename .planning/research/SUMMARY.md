# Research Summary: TCM Knowledge Graphs & Datasets

**Domain:** Traditional Chinese Medicine (TCM) Informatics
**Researched:** March 2026
**Overall confidence:** HIGH

## Executive Summary

This research identifies the most "defensible" datasets for a Final Year Project (FYP) in Traditional Chinese Medicine (TCM). TCM research has transitioned from isolated databases (like the original TCMID) to multi-dimensional **Knowledge Graphs (KGs)** that bridge traditional clinical patterns (Zheng) with modern molecular biology. 

The "Big Four" datasets (TCMSP, SymMap, TCMID, TCM-Mesh) remain the industry standards, but the most advanced research in 2024-2025 now uses **integrated meta-graphs** like **TCM-MKG (2025)**. For an FYP, selecting a dataset that adheres to international standards (ICD-11, UMLS, MeSH) is critical for academic credibility and interoperability.

## Key Findings

**Top 3 Recommendations for FYP:**
1.  **TCMSP (TCM Systems Pharmacology):** The gold standard for active ingredient screening and ADME (Absorption, Distribution, Metabolism, and Excretion) properties.
2.  **SymMap (Symptom-to-Molecular Mapping):** The premier choice for clinical phenotypic research, linking TCM symptoms/syndromes to modern medical phenotypes.
3.  **TCM-MKG (TCM Multidimensional Knowledge Graph, 2025):** The strongest "Modern" choice. It integrates over 30 databases (including TCMID and TCM-Mesh) into a standardized triple-based format (Knowledge Graph), mapping concepts to ICD-11 and UMLS.

**Implications for Roadmap:**
- **Phase 1: Knowledge Acquisition.** Download and standardize the dataset. (TCM-MKG is the easiest to integrate).
- **Phase 2: Graph Construction.** Import into Neo4j or NetworkX. (TCM-MKG provides pre-formatted triples).
- **Phase 3: Logic/Analysis.** Implement RAG or GNN algorithms.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack (Datasets) | HIGH | Based on citation counts (>10k for TCMSP) and 2025 publications for TCM-MKG. |
| Features | HIGH | Detailed scope for Herbs, Ingredients, Targets, and Diseases is well-documented. |
| Architecture | MEDIUM | Integration with Neo4j depends on the provided CSV/JSON format quality. |
| Pitfalls | MEDIUM | Data sparsity in older datasets (like TCMID) is a known issue. |

## Gaps to Address

- **Real-time Updates:** TCM datasets are often static; some links (e.g., original TCMID) may be broken. Using Zenodo mirrors is recommended.
- **Multilingual Support:** Most datasets are in English or Chinese; mapping between them (e.g., Pinyin to Latin) requires standardized terminology tables (provided in TCM-MKG).
