"""Check how many chunks produced zero entities (likely failed extractions)."""
import json
from collections import Counter

with open('data/graph/entities_partial.json', encoding='utf-8') as f:
    data = json.load(f)

entities = data['entities']
relationships = data['relationships']

print("=" * 60)
print("EXTRACTION QUALITY CHECK")
print("=" * 60)

# Check source_ref distribution
source_refs = [e.get('source_ref', {}).get('chunk_id', 'unknown') for e in entities]
chunk_counter = Counter(source_refs)

print(f"\n1. Total entities: {len(entities)}")
print(f"   Total relationships: {len(relationships)}")
print(f"   Unique chunks that produced entities: {len(chunk_counter)}")

# Entities per chunk stats
counts = list(chunk_counter.values())
if counts:
    print(f"\n2. Entities per chunk stats:")
    print(f"   Min: {min(counts)}")
    print(f"   Max: {max(counts)}")
    print(f"   Avg: {sum(counts)/len(counts):.1f}")

# What percentage of 2027 chunks have entities?
print(f"\n3. Coverage:")
print(f"   Chunks with entities: {len(chunk_counter)}")
print(f"   Chunks without entities: {2027 - len(chunk_counter)}")
print(f"   Failure rate: {(2027 - len(chunk_counter)) / 2027 * 100:.1f}%")
