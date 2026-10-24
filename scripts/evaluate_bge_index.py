"""Frozen fresh full-index development evaluation; preflight is offline by default.

--live uses only the pinned free retrieval provider, never chat/generation/Arena.
Unfiltered rankings, exact metadata filters and optional release scope logic are
reported separately. Outputs are exclusive new files under the ignored index root.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
CORPUS = ROOT / 'data/processed/chunks.json'
CASES = ROOT / 'scripts/fixtures/retrieval_independent_cases.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def prepare(index, cases_path=CASES):
    manifest = json.loads((index / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('status') != 'complete':
        raise ValueError('Index manifest is not complete; wait for the builder')
    chunks = json.loads(CORPUS.read_text(encoding='utf-8'))
    by_id = {row['id']: row for row in chunks}
    fingerprint = manifest.get('fingerprint', {})
    if len(by_id) != len(chunks) or manifest.get('verified_count') != len(chunks):
        raise ValueError('Index/corpus counts are not verified')
    if (fingerprint.get('corpus_sha256') != sha(CORPUS) or
        fingerprint.get('model') != 'BAAI/bge-m3' or fingerprint.get('dimensions') != 1024 or
        fingerprint.get('preprocessing') != 'unaltered-content-v1'):
        raise ValueError('Index fingerprint does not match the frozen evidence/model')
    fixture = json.loads(cases_path.read_text(encoding='utf-8'))
    cases = fixture['cases']
    if fixture.get('version') != 2 or len({case['id'] for case in cases}) != len(cases):
        raise ValueError('Case IDs must be unique')
    for case in cases:
        expected = case['expected_chunk_ids']
        if case.get('expected_match') not in ('any', 'all', 'none'):
            raise ValueError('Missing explicit match semantics')
        if (not expected) != case['negative_control'] or (case['expected_match'] == 'none') != case['negative_control']:
            raise ValueError('Invalid positive/control labels')
        if len(set(expected)) != len(expected):
            raise ValueError('Duplicate expected source')
        anchors = set()
        for evidence in case['evidence']:
            cid, quote = evidence['chunk_id'], evidence['quote']
            if cid not in expected or cid not in by_id or not quote or quote not in by_id[cid]['content']:
                raise ValueError('A frozen evidence anchor is invalid')
            anchors.add(cid)
        if anchors != set(expected):
            raise ValueError('Every expected source needs a literal anchor')
    return chunks, cases, manifest


def metrics(cases, rankings):
    rows = []
    for case in cases:
        if case['negative_control']:
            continue
        ids = [r['chunk_id'] for r in rankings[case['id']]]
        labels = set(case['expected_chunk_ids'])
        first = next((i for i, cid in enumerate(ids, 1) if cid in labels), None)
        fraction = len(labels.intersection(ids[:5])) / len(labels)
        coverage = fraction if case['expected_match'] == 'all' else float(bool(first and first <= 5))
        rows.append({'case_id': case['id'], 'first_labelled_rank': first,
                     'hit_at_5': bool(first and first <= 5), 'required_coverage_at_5': coverage,
                     'all_required_at_5': labels <= set(ids[:5]) if case['expected_match'] == 'all' else bool(first and first <= 5),
                     'expected_match': case['expected_match']})
    n = len(rows)
    return {'positive_cases': n, 'hit_at_1_count': sum(r['first_labelled_rank'] == 1 for r in rows),
            'hit_at_5_count': sum(r['hit_at_5'] for r in rows),
            'hit_at_5': sum(r['hit_at_5'] for r in rows) / n if n else None,
            'mrr_at_20': sum(1/r['first_labelled_rank'] if r['first_labelled_rank'] else 0 for r in rows) / n if n else None,
            'macro_required_coverage_at_5': sum(r['required_coverage_at_5'] for r in rows)/n if n else None,
            'per_case': rows,
            'interpretation': 'Frozen manual development labels, incomplete relevance judgments; not clinical or general corpus accuracy.'}


def metadata_matches(meta, where):
    if '$and' in where:
        return all(metadata_matches(meta, item) for item in where['$and'])
    return all(meta.get(key) == value for key, value in where.items())


def evaluate(index, cases_path, *, live=False, include_runtime_scope=False):
    chunks, cases, manifest = prepare(index, cases_path)
    before = {'corpus': sha(CORPUS), 'cases': sha(cases_path), 'manifest': sha(index / 'manifest.json')}
    if not live:
        return {'status': 'preflight_ready', 'mode': 'offline-preflight', 'index': str(index),
                'corpus_count': len(chunks), 'cases': len(cases), 'fingerprint': before,
                'interpretation': 'No API calls, index writes or evaluation results.'}
    import os
    import chromadb
    from chromadb.config import Settings
    from dotenv import dotenv_values
    from siliconflow_retrieval import FreeRetrievalClient
    from demo_mode import DemoVectorStore
    key = os.getenv('SILICONFLOW_API_KEY')
    if not key:
        key = dotenv_values(ROOT / '.env').get('SILICONFLOW_API_KEY')
    if not key:
        raise ValueError('Free retrieval key is not configured locally')
    client = FreeRetrievalClient(key)
    db = chromadb.PersistentClient(path=str(index / 'chroma'), settings=Settings(anonymized_telemetry=False))
    collection = db.get_collection('tcm_bge_m3', embedding_function=None)
    records = collection.get(include=['documents', 'metadatas'])
    original = {row['id']: row for row in chunks}
    if (len(records['ids']) != len(chunks) or len(set(records['ids'])) != len(chunks) or
        set(records['ids']) != set(original)):
        raise ValueError('Actual collection ID/count mismatch')
    for cid, content, meta in zip(records['ids'], records['documents'], records['metadatas']):
        if content != original[cid]['content'] or meta != original[cid]['metadata']:
            raise ValueError('Actual collection provenance mismatch')
    del records
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid4().hex[:8]
    out = ROOT / 'vectorstore/retrieval-validation' / ('full-' + run_id)
    out.mkdir(parents=True, exist_ok=False)
    with (out / 'cases_snapshot.json').open('xb') as handle:
        handle.write(cases_path.read_bytes())
    lexical_store = DemoVectorStore(CORPUS)
    lexical, dense, reranked, filtered = {}, {}, {}, {}
    query_vectors, rerank_cache = {}, {}

    def search(question, where=None):
        if question not in query_vectors:
            query_vectors[question] = client.embed([question])[0]
        kwargs = {'query_embeddings': [query_vectors[question]], 'n_results': 20,
                  'include': ['documents', 'metadatas', 'distances']}
        if where:
            kwargs['where'] = where
        raw = collection.query(**kwargs)
        return [{'chunk_id': cid, 'content': text, 'metadata': meta, 'distance': distance}
                for cid, text, meta, distance in zip(raw['ids'][0], raw['documents'][0], raw['metadatas'][0], raw['distances'][0])]

    def rerank(question, candidates):
        cache_key = (question, tuple(r['chunk_id'] for r in candidates))
        if cache_key not in rerank_cache:
            rerank_cache[cache_key] = client.rerank(question, [r['content'] for r in candidates], top_n=len(candidates)) if candidates else []
        return [{**candidates[r['index']], 'rerank_score': r['relevance_score']} for r in rerank_cache[cache_key]]

    report = {'version': 1, 'mode': 'full-corpus-free-retrieval-development', 'status': 'running',
              'index': str(index), 'run_id': run_id, 'corpus_count': len(chunks),
              'fingerprint': {**before, 'runner_sha256': sha(Path(__file__)), 'case_snapshot_sha256': sha(out / 'cases_snapshot.json')},
              'limitations': 'Fresh source-grounded manual labels; not blinded or clinical validation. Controls are reported separately; nearest neighbours do not establish support or abstention.'}
    current_case = None
    try:
        for case in cases:
            cid, question = case['id'], case['question']
            current_case = cid
            lexical[cid] = [{'chunk_id': doc.id, 'distance': score} for doc, score in lexical_store.similarity_search_with_score(question, k=20)]
            dense[cid] = search(question)  # Explicitly NO automatic book/clause filter.
            reranked[cid] = rerank(question, dense[cid])
            if case.get('metadata_filter'):
                candidates = search(question, case['metadata_filter'])
                filtered[cid] = {'filter': case['metadata_filter'], 'rankings': candidates,
                                 'all_results_match': all(metadata_matches(r['metadata'], case['metadata_filter']) for r in candidates),
                                 'expected_targets_present': set(case['expected_chunk_ids']) <= {r['chunk_id'] for r in candidates}}
        report['lexical'] = {'metrics': metrics(cases, lexical), 'rankings': lexical}
        report['unfiltered_dense'] = {'metrics': metrics(cases, dense), 'rankings': dense}
        report['unfiltered_reranked'] = {'metrics': metrics(cases, reranked), 'rankings': reranked}
        report['metadata_filtered_lane'] = filtered
        report['negative_controls'] = {case['id']: {'question': case['question'], 'lexical': lexical[case['id']],
                                      'dense': dense[case['id']], 'reranked': reranked[case['id']],
                                      'abstention_verified': False} for case in cases if case['negative_control']}
        if include_runtime_scope:
            from unittest.mock import patch
            scope_hash = sha(ROOT / 'src/release_runtime.py')
            import release_runtime
            scoped = {}
            facade = type('CachedRetrieval', (), {})()
            facade.embed = lambda questions: [query_vectors[q] if q in query_vectors else client.embed([q])[0] for q in questions]
            facade.rerank = client.rerank
            with patch.object(release_runtime, 'corpus', return_value=original), patch.object(release_runtime, 'collection', return_value=collection), patch.object(release_runtime, 'retrieval_client', return_value=facade):
                for case in cases:
                    scoped[case['id']] = release_runtime.retrieve(case['question'])
            scoped_metrics = metrics(cases, scoped)
            scoped_metrics['mrr_at_6'] = scoped_metrics.pop('mrr_at_20')
            if scope_hash != sha(ROOT / 'src/release_runtime.py'):
                raise ValueError('Scoped runtime changed during evaluation')
            report['release_scoped_lane'] = {'metrics': scoped_metrics, 'rankings': scoped,
                                             'runtime_sha256': scope_hash,
                                             'note': 'Separate auto clause/book-aware runtime logic; does not replace unfiltered metrics.'}
        dm, rm = report['unfiltered_dense']['metrics'], report['unfiltered_reranked']['metrics']
        checks = {'dense_hit5_at_least_10': dm['hit_at_5_count'] >= 10,
                  'reranked_hit5_at_least_10': rm['hit_at_5_count'] >= 10,
                  'dense_coverage_at_least_075': dm['macro_required_coverage_at_5'] >= .75,
                  'reranked_coverage_at_least_075': rm['macro_required_coverage_at_5'] >= .75,
                  'all_exact_filters_valid': bool(filtered) and all(r['all_results_match'] and r['expected_targets_present'] for r in filtered.values()),
                  'all_required_contexts_in_reranked_top5': all(r['all_required_at_5'] for r in rm['per_case'] if r['expected_match'] == 'all')}
        report['acceptance'] = {'status': 'pass' if all(checks.values()) else 'needs_review', 'checks': checks,
                                'note': 'Development retrieval only; no automatic deployment or clinical acceptance.'}
        report['status'] = 'complete'
    except Exception:
        report['status'] = 'failed'
        report['failure'] = 'Evaluation stopped; no upstream error body or credentials recorded.'
        report['case_in_progress'] = current_case
        report['partial_rankings'] = {'lexical': lexical, 'dense': dense, 'reranked': reranked, 'metadata_filtered': filtered}
    report['preservation'] = {'corpus_unchanged': before['corpus'] == sha(CORPUS),
                              'cases_unchanged': before['cases'] == sha(cases_path),
                              'manifest_unchanged': before['manifest'] == sha(index / 'manifest.json')}
    if not all(report['preservation'].values()):
        report['status'] = 'failed'
        report['failure'] = 'Preserved evidence changed during evaluation; results cannot be accepted.'
    write(out / 'report.json', report)
    return {'status': report['status'], 'report': str(out / 'report.json'),
            'acceptance': report.get('acceptance'), 'preservation': report['preservation']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--cases', type=Path, default=CASES)
    parser.add_argument('--live', action='store_true', help='Use authorized pinned free retrieval APIs')
    parser.add_argument('--include-runtime-scope', action='store_true', help='Also evaluate release book/clause scope separately')
    args = parser.parse_args()
    try:
        result = evaluate(args.index.resolve(), args.cases.resolve(), live=args.live, include_runtime_scope=args.include_runtime_scope)
        print(json.dumps(result, ensure_ascii=True, indent=2))
        sys.exit(1 if result['status'] == 'failed' else 0)
    except Exception:
        print('Evaluation preflight failed; check complete index, frozen cases and local configuration.', file=sys.stderr)
        sys.exit(1)
