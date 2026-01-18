"""Reset processed_chunk_ids.json to only keep Lingshu chunks (which are complete)."""
import json

# Load chunks to identify Lingshu chunks
with open('data/processed/chunks.json', encoding='utf-8') as f:
    chunks = json.load(f)

# Get Lingshu chunk IDs
lingshu_ids = [c['id'] for c in chunks if c.get('metadata', {}).get('book') == '黄帝内经灵枢集注']

print(f"Total chunks: {len(chunks)}")
print(f"Lingshu chunks: {len(lingshu_ids)}")

# Save only Lingshu IDs
with open('data/graph/processed_chunk_ids.json', 'w', encoding='utf-8') as f:
    json.dump(lingshu_ids, f, ensure_ascii=False, indent=2)

print(f"Saved {len(lingshu_ids)} Lingshu chunk IDs to processed_chunk_ids.json")
print("Suwen and Taisu chunks are now marked as 'not processed' for re-extraction")
