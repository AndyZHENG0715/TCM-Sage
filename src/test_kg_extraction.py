"""
Quick test script for KG extractor on 10 sample chunks.
"""
from kg_extractor import extract_kg_batch
import json

import os

# Load first 10 chunks relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHUNKS_PATH = os.path.join(SCRIPT_DIR, '../data/processed/chunks.json')

with open(CHUNKS_PATH, encoding='utf-8') as f:
    chunks = json.load(f)

print(f"Testing with {len(chunks[:10])} sample chunks...\n")

# Run extraction
result = extract_kg_batch(chunks, model='qwen3:8b', num_ctx=4096, limit=10)

# Print results
print(f"\n{'='*60}")
print("Test Results:")
print(f"{'='*60}")
print(f"Entities: {len(result['entities'])}")
print(f"Relationships: {len(result['relationships'])}")
print(f"High-confidence entities: {result['extraction_stats']['high_confidence_entities']}")
print(f"High-confidence relationships: {result['extraction_stats']['high_confidence_relationships']}")

# Show sample entities
if result['entities']:
    print(f"\nSample entities (first 5):")
    for e in result['entities'][:5]:
        print(f"  - {e.get('type')}: {e.get('mention')} (conf: {e.get('confidence', 0):.2f})")

# Show sample relationships
if result['relationships']:
    print(f"\nSample relationships (first 5):")
    for r in result['relationships'][:5]:
        print(f"  - {r.get('head')} --{r.get('relation')}--> {r.get('tail')} (conf: {r.get('confidence', 0):.2f})")
