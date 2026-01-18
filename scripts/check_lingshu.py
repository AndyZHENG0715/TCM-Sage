"""Check if 灵枢集注 (Lingshu) extraction is complete."""
import json
from collections import Counter

# Load chunks
with open('data/processed/chunks.json', encoding='utf-8') as f:
    chunks = json.load(f)

# Load entities
with open('data/graph/entities_partial.json', encoding='utf-8') as f:
    kg_data = json.load(f)

entities = kg_data['entities']

print("=" * 60)
print("灵枢集注 (Lingshu) Extraction Status Check")
print("=" * 60)

# Count chunks per book
chunk_books = Counter()
for c in chunks:
    book = c.get('metadata', {}).get('book', 'unknown')
    chunk_books[book] += 1

print("\n1. CHUNKS BY BOOK:")
for book, count in chunk_books.most_common():
    print(f"   {book}: {count}")

# Count entities per book
entity_books = Counter()
entity_chapters = Counter()
for e in entities:
    source_ref = e.get('source_ref', {})
    book = source_ref.get('book', 'unknown')
    chapter = source_ref.get('chapter', 'unknown')
    entity_books[book] += 1
    if book == '黄帝内经灵枢集注':
        entity_chapters[chapter] += 1

print("\n2. ENTITIES BY BOOK:")
for book, count in entity_books.most_common():
    print(f"   {book}: {count}")

# Check Lingshu coverage
lingshu_chunks = chunk_books.get('黄帝内经灵枢集注', 0)
lingshu_entities = entity_books.get('黄帝内经灵枢集注', 0)

print(f"\n3. 灵枢集注 COVERAGE:")
print(f"   Total Lingshu chunks: {lingshu_chunks}")
print(f"   Entities extracted from Lingshu: {lingshu_entities}")

# Estimate chunks that produced entities (assuming ~4-5 entities per chunk)
estimated_chunks_with_entities = lingshu_entities / 5
print(f"   Estimated chunks with entities: ~{estimated_chunks_with_entities:.0f}")
print(f"   Estimated Lingshu coverage: ~{estimated_chunks_with_entities / lingshu_chunks * 100:.1f}%")

print(f"\n4. CHAPTERS WITH ENTITIES (Lingshu only):")
for chapter, count in entity_chapters.most_common():
    print(f"   {chapter}: {count} entities")

# Total chapters in Lingshu
lingshu_chunk_chapters = set()
for c in chunks:
    if c.get('metadata', {}).get('book') == '黄帝内经灵枢集注':
        lingshu_chunk_chapters.add(c.get('metadata', {}).get('chapter', 'unknown'))

print(f"\n5. TOTAL CHAPTERS IN LINGSHU CHUNKS: {len(lingshu_chunk_chapters)}")
print(f"   Chapters with entities: {len(entity_chapters)}")
print(f"   Chapters without entities: {len(lingshu_chunk_chapters) - len(entity_chapters)}")

if len(lingshu_chunk_chapters) - len(entity_chapters) > 0:
    missing = lingshu_chunk_chapters - set(entity_chapters.keys())
    print(f"\n   MISSING CHAPTERS (first 10):")
    for ch in list(missing)[:10]:
        print(f"     - {ch}")
