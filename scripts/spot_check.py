"""
Spot-check 20 random entities from the extracted KG.
Verify that evidence actually appears in source text.
"""
import json
import random

# Load entities
with open('data/graph/entities_partial.json', encoding='utf-8') as f:
    data = json.load(f)

entities = data.get('entities', [])
print(f"Total entities: {len(entities)}")
print(f"Total relationships: {len(data.get('relationships', []))}")
print()

# Get type distribution
from collections import Counter
types = Counter(e.get('type', 'Unknown') for e in entities if 'type' in e)
print("Entity type distribution:")
for t, c in types.most_common():
    print(f"  {t}: {c}")
print()

# Sample 20 random entities with evidence
entities_with_evidence = [e for e in entities if e.get('evidence')]
sample = random.sample(entities_with_evidence, min(20, len(entities_with_evidence)))

print("=" * 60)
print("SPOT-CHECK: 20 Random Entities")
print("=" * 60)
for i, e in enumerate(sample, 1):
    print(f"\n{i}. [{e.get('type')}] {e.get('mention')}")
    print(f"   Evidence: \"{e.get('evidence', 'N/A')[:50]}...\"")
    print(f"   Confidence: {e.get('confidence', 'N/A')}")
    print(f"   Supported: {e.get('supported', 'N/A')}")
