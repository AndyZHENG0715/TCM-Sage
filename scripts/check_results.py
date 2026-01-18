import json

with open('data/graph/entities_partial.json', encoding='utf-8') as f:
    d = json.load(f)

print(f"Entities: {len(d['entities'])}")
print(f"Relationships: {len(d['relationships'])}")
print(f"Stats: {d.get('extraction_stats', {})}")

# Type distribution
from collections import Counter
types = Counter(e['type'] for e in d['entities'])
print("\nEntity type distribution:")
for t, c in types.most_common():
    print(f"  {t}: {c}")
