import sys, os, pathlib
sys.path.insert(0, 'src')
stdout_reconfigure = getattr(sys.stdout, 'reconfigure', None)
if callable(stdout_reconfigure):
    stdout_reconfigure(encoding='utf-8')
os.chdir('D:/Dev/TCM-Sage')
from dotenv import load_dotenv
load_dotenv()

from langchain_chroma import Chroma
from embeddings import get_embedding_model
from ingest import process_single_source, SentenceAwareChineseTextSplitter
from main import create_llm

print("=" * 60)
print("STEP 1: Remove old 伤寒论/金匮要略 chunks")
print("=" * 60)

embeddings = get_embedding_model()
vs = Chroma(persist_directory='vectorstore/chroma', embedding_function=embeddings)
collection = vs._collection
all_data = collection.get(include=['metadatas'])
all_metadatas = all_data.get('metadatas') or []
all_ids = all_data.get('ids') or []

ids_to_remove = []
for i, meta in enumerate(all_metadatas):
    if meta and meta.get('book') in ('伤寒论', '金匮要略方论') and i < len(all_ids):
        ids_to_remove.append(all_ids[i])

print(f"Removing {len(ids_to_remove)} old chunks...")
for i in range(0, len(ids_to_remove), 100):
    batch = ids_to_remove[i:i+100]
    collection.delete(ids=batch)
print(f"Done. DB count: {collection.count()}")

print()
print("=" * 60)
print("STEP 2: Re-ingest with contextual headers")
print("=" * 60)

source_dir = pathlib.Path('data/source')
splitter = SentenceAwareChineseTextSplitter(chunk_size=500, chunk_overlap=50)

targets = [('伤寒论.txt', '伤寒论'), ('金匮要略方论.txt', '金匮要略方论')]
all_chunks = []
for fname, book_name in targets:
    chunks = process_single_source(source_dir / fname, book_name, splitter)
    all_chunks.extend(chunks)
    print(f"  {book_name}: {len(chunks)} chunks")
    if chunks:
        print(f"  Sample: {chunks[0]['content'][:120]}")

print(f"Total: {len(all_chunks)} chunks")

seen_ids = set()
unique_chunks = []
for c in all_chunks:
    if c['id'] not in seen_ids:
        seen_ids.add(c['id'])
        unique_chunks.append(c)
    else:
        c['id'] = c['id'] + '_dup'
        unique_chunks.append(c)
all_chunks = unique_chunks

BATCH_SIZE = 10
total_batches = (len(all_chunks) + BATCH_SIZE - 1) // BATCH_SIZE
for i in range(0, len(all_chunks), BATCH_SIZE):
    batch = all_chunks[i:i+BATCH_SIZE]
    batch_num = i // BATCH_SIZE + 1
    texts = [c['content'] for c in batch]
    ids = [c['id'] for c in batch]
    metas = [c['metadata'] for c in batch]
    try:
        vs.add_texts(texts=texts, metadatas=metas, ids=ids)
        if batch_num % 10 == 0 or batch_num == total_batches:
            print(f"  Batch {batch_num}/{total_batches}")
    except Exception as e:
        print(f"  Error at batch {batch_num}: {e}")

print(f"DB count after: {collection.count()}")

print()
print("=" * 60)
print("STEP 3: Retrieval quality test")
print("=" * 60)

vs2 = Chroma(persist_directory='vectorstore/chroma', embedding_function=embeddings)

queries = [
    '麻黄汤的完整药物组成',
    '四逆散的组成是什么',
    '桂枝汤服药后的护理要点',
    '当归四逆汤由哪些药物组成',
    '太阳病的提纲条文',
    '小柴胡汤的药物组成',
    '炙甘草汤的组成',
    '白虎汤证的典型症状',
    '水蛭的性味',
    '蛇床子有哪些配伍禁忌',
]

print()
for q in queries:
    results = vs2.similarity_search_with_score(q, k=3)
    print(f'Q: {q}')
    for r, score in results[:2]:
        m = r.metadata
        book = m.get('book', '?')
        clause = m.get('clause_number', '')
        formula = m.get('formula', '')
        clause_str = f' #cl{clause}' if clause else ''
        formula_str = f' [{formula}]' if formula else ''
        print(f'  {score:.3f} | {book}{clause_str}{formula_str} | {r.page_content[:80]}')
    print()

print()
print("=" * 60)
print("STEP 4: Plain LLM comparison (for questions with good retrieval)")
print("=" * 60)

llm = create_llm('alibaba', 'qwen-turbo', 0.3)

llm_test_qs = [
    '水蛭的性味是什么？出自哪本经典？',
    '蛇床子有哪些配伍禁忌？恶什么药？',
    '麻黄汤的完整药物组成和剂量是什么？请引用原文。',
    '四逆散由哪些药物组成？',
    '当归四逆汤的组成是什么？',
]

for q in llm_test_qs:
    resp = llm.invoke(q)
    text = resp.content if hasattr(resp, 'content') else str(resp)
    print(f'Q: {q}')
    print(f'Plain LLM (first 200 chars): {text[:200]}')
    print()

print()
print("=" * 60)
print("FINAL VERDICT")
print("=" * 60)
print("Check: for each question above, does the RAG retrieval provide")
print("MORE ACCURATE information than the plain LLM?")
print("Key signal: wrong properties, fabricated ingredients, missing dosages")
