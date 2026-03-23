# Change: Data Pipeline Overhaul & LLM-Based KG Extraction

## Why

The current TCM-Sage has two limitations:
1. **Incomplete source data**: Current 黄帝内经 source is incomplete
2. **Manual KG**: Knowledge Graph was manually curated for testing, lacking provenance and scalability

This change replaces the data pipeline with complete, authoritative TCM sources and automated KG extraction using local LLM.

## What Changes

### Data Sources

Replace current source with complete texts from [xiaopangxia/TCM-Ancient-Books](https://github.com/xiaopangxia/TCM-Ancient-Books):

| File | Description | Size |
|------|-------------|------|
| `437-黄帝内经素问.txt` | 素问 core text | 237KB |
| `431-黄帝内经灵枢集注.txt` | 灵枢 with annotations | 601KB |
| `439-黄帝内经太素.txt` | 太素 recension | 870KB |

### LLM-Based KG Extraction

Use **local Ollama** with Qwen3 model (based on hardware research):

| Model | VRAM | Recommendation |
|-------|------|----------------|
| `qwen3:8b` | ~4.5GB Q4 | ✅ Primary (if fits 6GB) |
| `qwen3:4b` | ~2.6GB Q4 | Fallback option |

**3-Pass Extraction Architecture** (inspired by OpenTCM GraphRAG approach):
```
Pass 1: Entity Extraction → entities[] with evidence spans
Pass 2: Relation Extraction → relations[] using entity IDs only
Pass 3: Self-Critique → verify against original text, add confidence scores
```

### Entity Schema (WHO-Aligned)

Based on WHO International Standard Terminologies on TCM (2022):

| Entity Type | Chinese | WHO Category |
|-------------|---------|--------------|
| `Symptom` | 症状 | Diagnostics |
| `Pattern` | 证候/证型 | Diagnostics (ICD-11 TM1) |
| `Herb` | 中药 | Medicinal Treatment |
| `Formula` | 方剂 | Medicinal Treatment |
| `TreatmentMethod` | 治法 | Therapeutics |
| `Meridian` | 经络 | Acupuncture & Moxibustion |
| `Acupoint` | 腧穴 | Acupuncture (361 standard) |

### Relationship Types

| Relation | Direction | Example |
|----------|-----------|---------|
| `TREATS` | Herb/Formula → Symptom/Pattern | 川芎 TREATS 頭痛 |
| `CONTAINS` | Formula → Herb | 川芎茶調散 CONTAINS 川芎 |
| `INDICATES` | Symptom → Pattern | 舌红苔黄 INDICATES 肝火上炎 |
| `APPLIES_TO` | TreatmentMethod → Pattern | 疏肝清热 APPLIES_TO 肝火上炎 |
| `LOCATED_ON` | Acupoint → Meridian | 合谷 LOCATED_ON 手阳明大肠经 |
| `ASSOCIATED_WITH` | Any ↔ Any | Catch-all |

### Provenance Tracking

Each KG fact stores source reference for citations:
```python
{
  "fact": "川芎 --TREATS--> 頭痛",
  "source_ref": {
    "book": "黄帝内经素问",
    "chapter": "...",
    "char_start": 12450,
    "char_end": 12520
  },
  "confidence": 0.95
}
```

## Impact

### Affected Files
- `src/ingest.py` – ✅ DONE: Multi-source ingestion with char offsets
- `src/kg_extractor.py` – PENDING: 3-pass extraction with self-critique
- `data/source/` – ✅ DONE: New source files
- `data/graph/entities.json` – PENDING: Extracted KG
- `vectorstore/` – ✅ DONE: Rebuilt with 2,400 chunks

### Dependencies
- `ollama` – ✅ Added to requirements.txt
- `tqdm` – Progress bar (already available)

## Research References

- **OpenTCM** (arXiv 2025): GraphRAG for TCM, 3,700+ herbs, 65,000+ references
- **WHO TCM Terminology** (2022): Official categories and ICD-11 TM1
- **TCMID**: REST API for herb→target→disease mappings (future enrichment)
- **AI-HPC-Research-Team/TCM_knowledge_graph**: Similar LLM-based approach

## Open Questions

1. ~~Model selection~~ → **Resolved**: qwen3:8b primary, 4b fallback
2. ~~Entity schema~~ → **Resolved**: 7 types, WHO-aligned
3. **Pending**: Extraction quality validation on sample set
