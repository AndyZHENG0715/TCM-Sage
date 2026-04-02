# TCM Dataset Technology Stack

**Project:** TCM-Sage (FYP Knowledge Graph Research)
**Researched:** March 2026

## Recommended Stack (Datasets)

### 1. TCMSP (TCM Systems Pharmacology)
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| **TCMSP** | 2.3 (Current) | Molecular Screening | Gold standard for **active ingredients** (OB/DL properties). |
| **Download Source** | Zenodo / GitHub | CSV/TSV | The official site (tcmspw.com) is often slow; bulk CSVs from research repositories are better. |

### 2. SymMap (Symptom Mapping)
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| **SymMap** | 2019/Current | Symptom Mapping | Best for **phenotypic mapping** between TCM and Western Medicine. |
| **Download Source** | symmap.org | SQL / Excel | Directly available for academics via email or simple registration. |

### 3. TCM-MKG (The 2025 Integrated Recommendation)
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| **TCM-MKG** | 2.0 (2025) | Core Knowledge Graph | **Most "Defensible" Choice.** Integrates 30+ databases into standardized triples. |
| **Format** | Zenodo | **JSON/CSV Triples** | Pre-built for Graph Neural Networks (GNNs) and Knowledge Graphs. |

## Supporting Libraries (Integration)

| Library | Purpose | When to Use |
|---------|---------|-------------|
| **NetworkX** | Python Graph Library | Small to medium-sized graph analysis and visualization in Python. |
| **Neo4j** | Graph Database | Large-scale production-ready graph queries (Cypher). |
| **PyG (PyTorch Geometric)** | GNN Framework | If the FYP involves predicting new herb-disease links or link prediction. |
| **UMLS-Py** | Medical Ontology | Mapping TCM terms to international standards (e.g., MeSH, ICD-11). |

## Installation & Integration

```bash
# Recommendation: Use a pre-cleaned Zenodo version of TCM-MKG
# Example: Load into NetworkX
import pandas as pd
import networkx as nx

# Load nodes and edges
nodes = pd.read_csv('tcm_mkg_nodes.csv')
edges = pd.read_csv('tcm_mkg_edges.csv')

G = nx.from_pandas_edgelist(edges, source='subject', target='object', edge_attr='relation')
```

## Sources

- **TCMSP:** [Ru et al. 2014](https://doi.org/10.1186/1758-2946-6-13) (>10k citations)
- **SymMap:** [Wu et al. 2019](https://doi.org/10.1093/nar/gky1074) (>1k citations)
- **TCM-MKG:** [Zeng et al. 2025](https://doi.org/10.1016/j.jpha.2025.101342) (Zenodo: 13763953)
- **TCMID:** [Huang et al. 2018](https://doi.org/10.1093/nar/gkx1089) (Included in TCM-MKG)
