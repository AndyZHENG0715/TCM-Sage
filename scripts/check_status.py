import json

ids = json.load(open('data/graph/processed_chunk_ids.json', encoding='utf-8'))
e = json.load(open('data/graph/entities_partial.json', encoding='utf-8'))

print(f"Chunks processed: {len(ids)}")
print(f"Total chunks in dataset: 2027")
print(f"Progress: {len(ids)}/2027 = {len(ids)/2027*100:.1f}%")
print(f"Entities: {len(e['entities'])}")
print(f"Relationships: {len(e['relationships'])}")
