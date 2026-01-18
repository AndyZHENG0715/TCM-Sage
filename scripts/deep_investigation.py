"""Fixed investigation using correct field names."""
import json
from datetime import datetime

# Load data
with open('data/graph/processed_chunk_ids.json', encoding='utf-8') as f:
    processed_ids = json.load(f)

with open('data/graph/entities_partial.json', encoding='utf-8') as f:
    entities_data = json.load(f)

with open('data/processed/chunks.json', encoding='utf-8') as f:
    all_chunks = json.load(f)

print("=" * 60)
print("CORRECTED INVESTIGATION: KG Extraction Status")
print("=" * 60)

# Basic counts
print(f"\n1. COUNTS:")
print(f"   Processed chunk IDs in tracking file: {len(processed_ids)}")
print(f"   Total chunks in chunks.json: {len(all_chunks)}")
print(f"   Entities extracted: {len(entities_data.get('entities', []))}")
print(f"   Relationships extracted: {len(entities_data.get('relationships', []))}")

# Use correct 'id' field (not 'chunk_id')
all_chunk_ids = set(c.get('id') for c in all_chunks)
processed_set = set(processed_ids)

print(f"\n2. CHUNK ID ANALYSIS (using 'id' field):")
print(f"   Unique chunk IDs in chunks.json: {len(all_chunk_ids)}")
print(f"   Unique processed IDs: {len(processed_set)}")

# Check overlap
overlap = all_chunk_ids & processed_set
missing_from_processed = all_chunk_ids - processed_set
extra_in_processed = processed_set - all_chunk_ids

print(f"\n3. COVERAGE:")
print(f"   Chunks processed that exist in dataset: {len(overlap)}")
print(f"   Chunks in dataset but NOT processed: {len(missing_from_processed)}")
print(f"   IDs in processed but NOT in dataset: {len(extra_in_processed)}")
print(f"   ACTUAL PROGRESS: {len(overlap)}/{len(all_chunk_ids)} = {len(overlap)/len(all_chunk_ids)*100:.1f}%")

# Show unprocessed if any
if missing_from_processed:
    print(f"\n4. UNPROCESSED CHUNKS (first 10):")
    for cid in list(missing_from_processed)[:10]:
        print(f"   - {cid}")

# Extraction stats
stats = entities_data.get('extraction_stats', {})
print(f"\n5. EXTRACTION STATS (from file):")
for k, v in stats.items():
    print(f"   {k}: {v}")
