"""Quick test to see all 7 entities from first chunk."""
import sys
sys.path.insert(0, 'src')
from kg_extractor import extract_kg_from_chunk
import json

chunks = json.load(open('data/processed/chunks.json', encoding='utf-8'))
r = extract_kg_from_chunk(chunks[0]['content'], chunks[0]['metadata'])

print('=== ALL ENTITIES ===')
for i, e in enumerate(r['entities']):
    print(f"{i+1}. [{e['type']}] {e.get('mention','?')} (conf:{e.get('confidence','?')}, supported:{e.get('supported','?')})")

print('\n=== ALL RELATIONSHIPS ===')
for rel in r['relationships']:
    print(f"  {rel['head']} --{rel['relation']}--> {rel['tail']} (conf:{rel.get('confidence','?')})")
