"""
COMPREHENSIVE KG EXTRACTION AUDIT
Examines every aspect of the extraction data to determine the true state.
"""
import json
from collections import Counter
from datetime import datetime

print("=" * 70)
print("COMPREHENSIVE KG EXTRACTION AUDIT")
print(f"Time: {datetime.now().isoformat()}")
print("=" * 70)

# ==================== 1. LOAD ALL DATA ====================
print("\n" + "=" * 70)
print("SECTION 1: DATA FILES")
print("=" * 70)

with open('data/processed/chunks.json', encoding='utf-8') as f:
    all_chunks = json.load(f)
print(f"chunks.json: {len(all_chunks)} chunks")

with open('data/graph/processed_chunk_ids.json', encoding='utf-8') as f:
    processed_ids = json.load(f)
print(f"processed_chunk_ids.json: {len(processed_ids)} IDs")

with open('data/graph/entities_partial.json', encoding='utf-8') as f:
    kg_data = json.load(f)
entities = kg_data.get('entities', [])
relationships = kg_data.get('relationships', [])
print(f"entities_partial.json: {len(entities)} entities, {len(relationships)} relationships")

# ==================== 2. ENTITY SOURCE ANALYSIS ====================
print("\n" + "=" * 70)
print("SECTION 2: WHERE DO ENTITIES COME FROM?")
print("=" * 70)

# Check source_ref structure
if entities:
    sample_entity = entities[0]
    print(f"\nSample entity structure:")
    for key, value in sample_entity.items():
        if isinstance(value, dict):
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {str(value)[:50]}...")

# Count unique source chunks
source_chunks = []
for e in entities:
    source_ref = e.get('source_ref', {})
    if isinstance(source_ref, dict):
        chunk_id = source_ref.get('chunk_id') or source_ref.get('id')
        if chunk_id:
            source_chunks.append(chunk_id)

unique_sources = set(source_chunks)
print(f"\nUnique source chunk IDs in entities: {len(unique_sources)}")

# If no source_ref, check if entities have other identifying info
if len(unique_sources) == 0:
    print("\nNo chunk_id found in source_ref. Checking other fields...")
    for e in entities[:3]:
        print(f"  Entity: {e}")

# ==================== 3. CHUNK ID COMPARISON ====================
print("\n" + "=" * 70)
print("SECTION 3: CHUNK ID MATCHING")
print("=" * 70)

# Get all chunk IDs from chunks.json
all_chunk_ids = set(c.get('id') for c in all_chunks)
processed_set = set(processed_ids)

print(f"Chunk IDs in chunks.json: {len(all_chunk_ids)}")
print(f"First 3 chunk IDs: {list(all_chunk_ids)[:3]}")
print(f"\nProcessed IDs in tracking file: {len(processed_set)}")
print(f"First 3 processed IDs: {list(processed_set)[:3]}")

overlap = all_chunk_ids & processed_set
print(f"\nOverlap: {len(overlap)} IDs match")

missing = all_chunk_ids - processed_set
if missing:
    print(f"Missing (in chunks.json but not processed): {len(missing)}")
    print(f"  Examples: {list(missing)[:3]}")

# ==================== 4. CHECK FOR EMPTY RESULTS ====================
print("\n" + "=" * 70)
print("SECTION 4: ENTITY QUALITY CHECK")
print("=" * 70)

# Type distribution
types = Counter(e.get('type', 'Unknown') for e in entities)
print("\nEntity type distribution:")
for t, c in types.most_common(10):
    print(f"  {t}: {c}")

# Evidence check (do entities have evidence?)
with_evidence = sum(1 for e in entities if e.get('evidence'))
print(f"\nEntities with evidence: {with_evidence}/{len(entities)}")

# Confidence check
confidences = [e.get('confidence', 0) for e in entities if 'confidence' in e]
if confidences:
    print(f"Confidence scores: min={min(confidences)}, max={max(confidences)}, avg={sum(confidences)/len(confidences):.2f}")

# ==================== 5. EXTRACTION STATS ====================
print("\n" + "=" * 70)
print("SECTION 5: EXTRACTION STATISTICS")
print("=" * 70)

stats = kg_data.get('extraction_stats', {})
if stats:
    for k, v in stats.items():
        print(f"  {k}: {v}")
else:
    print("  No extraction_stats found in file")

# ==================== 6. FINAL ASSESSMENT ====================
print("\n" + "=" * 70)
print("SECTION 6: FINAL ASSESSMENT")
print("=" * 70)

# Calculate actual coverage
if len(unique_sources) > 0:
    coverage = len(unique_sources) / len(all_chunks) * 100
else:
    # If we can't determine source chunks, make a rough estimate
    # Assume entities should average ~8 per chunk
    estimated_chunks = len(entities) / 8
    coverage = estimated_chunks / len(all_chunks) * 100

print(f"\nActual entities extracted: {len(entities)}")
print(f"Actual relationships extracted: {len(relationships)}")
print(f"Chunks with entity data: {len(unique_sources) if len(unique_sources) > 0 else 'UNKNOWN'}")
print(f"Estimated coverage: {coverage:.1f}%")
