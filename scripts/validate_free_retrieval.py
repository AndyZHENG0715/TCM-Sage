"""Isolated corpus retrieval experiment. Defaults to offline preparation.

Never imports the application, generation factory, Arena or ingestion script.
All generated files are confined to a new ignored experiment directory.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import socket
from unittest.mock import patch
from typing import Any
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
CORPUS = ROOT / "data/processed/chunks.json"
CASES = ROOT / "scripts/fixtures/retrieval_cases.json"
OUTPUT_ROOT = ROOT / "vectorstore/retrieval-validation"
MODEL = "BAAI/bge-m3"
RERANKER = "BAAI/bge-reranker-v2-m3"
SEED = 20261024
SAMPLE_SIZE = 256
MAX_CHARS = 2048
TOP_K = 20


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    # Exclusive creation prevents overwriting an existing index or report.
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def load_evidence(corpus_path: Path = CORPUS, cases_path: Path = CASES):
    chunks = json.loads(corpus_path.read_text(encoding="utf-8"))
    case_file = json.loads(cases_path.read_text(encoding="utf-8"))
    cases = case_file["cases"]
    by_id = {chunk["id"]: chunk for chunk in chunks}
    if len(by_id) != len(chunks):
        raise ValueError("Corpus chunk IDs must be unique")
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Case IDs must be unique")
    for case in cases:
        if type(case.get("negative_control")) is not bool:
            raise ValueError("Each case needs an explicit control flag")
        if not isinstance(case["question"], str) or not case["question"].strip():
            raise ValueError("Case question is empty")
        expected = case["expected_chunk_ids"]
        if bool(expected) == case["negative_control"]:
            raise ValueError("Positive cases need labels; controls must be unlabelled")
        if case.get("expected_match") not in ("any", "all", "none"):
            raise ValueError("Each case needs an explicit expected_match policy")
        if (case["expected_match"] == "none") != case["negative_control"]:
            raise ValueError("Control and expected_match policy disagree")
        if len(expected) != len(set(expected)):
            raise ValueError("Duplicate expected chunk IDs")
        for chunk_id in expected:
            if chunk_id not in by_id:
                raise ValueError("Expected chunk is missing from corpus")
            if not 0 < len(by_id[chunk_id]["content"]) <= MAX_CHARS:
                raise ValueError("Expected source exceeds experiment input limit")
        anchored = set()
        for evidence in case.get("evidence", []):
            chunk_id, quote = evidence["chunk_id"], evidence["quote"]
            if chunk_id not in expected or not quote or quote not in by_id[chunk_id]["content"]:
                raise ValueError("Evidence quote must be verbatim in a labelled source")
            anchored.add(chunk_id)
        if set(expected) != anchored:
            raise ValueError("Every expected source needs a quoted anchor")
    return chunks, cases


def choose_sample(chunks, cases, sample_size=SAMPLE_SIZE):
    from demo_mode import DemoVectorStore
    required = {cid for case in cases for cid in case["expected_chunk_ids"]}
    eligible = [c for c in chunks if c["content"].strip() and len(c["content"]) <= MAX_CHARS]
    eligible_ids = {c["id"] for c in eligible}
    if not required <= eligible_ids or len(required) > sample_size:
        raise ValueError("Expected sources do not fit the sample")
    # Hard lexical candidates and adjacent corpus passages accompany known positives.
    full_lexical = DemoVectorStore(CORPUS)
    selected = set(required)
    for case in cases:
        for document in full_lexical.similarity_search(case["question"], k=3):
            if document.id in eligible_ids:
                selected.add(document.id)
    for index, chunk in enumerate(chunks):
        if chunk["id"] in required:
            for nearby in chunks[max(0, index - 1):index + 2]:
                if nearby["id"] in eligible_ids:
                    selected.add(nearby["id"])
    if len(selected) > sample_size:
        raise ValueError("Hard candidates exceed sample size")
    remainder = [c["id"] for c in eligible if c["id"] not in selected]
    random.Random(SEED).shuffle(remainder)
    selected.update(remainder[:sample_size - len(selected)])
    if len(selected) != sample_size:
        raise ValueError("Insufficient eligible corpus records")
    return [c for c in chunks if c["id"] in selected]


def summarize(cases, rankings):
    positives = [case for case in cases if not case["negative_control"]]
    hits, reciprocals, recalls, coverage = [], [], [], []
    for case in positives:
        expected = set(case["expected_chunk_ids"])
        ids = [item["chunk_id"] for item in rankings[case["id"]]]
        rank = next((i for i, cid in enumerate(ids, 1) if cid in expected), None)
        hits.append(bool(rank and rank <= 5))
        reciprocals.append(1 / rank if rank else 0.0)
        labelled_recall = len(set(ids[:5]) & expected) / len(expected)
        # Alternative acceptable passages are any-match; multi-source tasks require all.
        recalls.append(labelled_recall if case["expected_match"] == "all" else float(bool(rank and rank <= 5)))
        coverage.append({"case_id": case["id"], "expected_match": case["expected_match"],
                         "first_labelled_rank": rank,
                         "labelled_recall_at_5": labelled_recall,
                         "all_labels_at_5": expected <= set(ids[:5]),
                         "all_labels_at_20": expected <= set(ids)})
    return {
        "positive_cases": len(positives),
        "hit_at_5_count": sum(hits),
        "hit_at_1_count": sum(rr == 1 for rr in reciprocals),
        "hit_at_5": sum(hits) / len(hits) if hits else None,
        "mrr_at_20": sum(reciprocals) / len(reciprocals) if reciprocals else None,
        "macro_required_coverage_at_5": sum(recalls) / len(recalls) if recalls else None,
        "source_coverage": coverage,
        "negative_controls": sum(c["negative_control"] for c in cases),
        "interpretation": "Incomplete manual labels on an enriched sample, not corpus accuracy. Controls are unscored; neighbours do not establish evidence or abstention.",
    }


def lexical_rankings(sample_path, cases):
    from demo_mode import DemoVectorStore
    store = DemoVectorStore(sample_path)
    return {
        case["id"]: [
            {"chunk_id": doc.id, "distance": score}
            for doc, score in store.similarity_search_with_score(case["question"], k=TOP_K)
        ] for case in cases
    }


def normalize(vector):
    if not vector or any(type(v) not in (int, float) or not math.isfinite(v) for v in vector):
        raise ValueError("Vector must contain finite numeric values")
    norm = math.sqrt(sum(v * v for v in vector))
    if not norm or not math.isfinite(norm):
        raise ValueError("Vector norm must be finite and nonzero")
    return [v / norm for v in vector]


def dense_rankings(sample, document_vectors, cases, query_vectors):
    if len(document_vectors) != len(sample) or len(query_vectors) != len(cases):
        raise ValueError("Vector count does not match evidence")
    vectors = [normalize(v) for v in document_vectors]
    dimensions = len(vectors[0])
    if any(len(v) != dimensions for v in vectors + query_vectors):
        raise ValueError("Vector dimensions do not match")
    result = {}
    for case, raw_query in zip(cases, query_vectors):
        query = normalize(raw_query)
        scores = [sum(a * b for a, b in zip(query, vector)) for vector in vectors]
        order = sorted(range(len(sample)), key=lambda i: (-scores[i], i))[:TOP_K]
        result[case["id"]] = [{"chunk_id": sample[i]["id"], "cosine": scores[i]} for i in order]
    return result


def semantic_gate(dense_metrics, reranked_metrics):
    checks = {
        "dense_hit_at_5_at_least_10_of_12": dense_metrics["hit_at_5_count"] >= 10,
        "reranked_hit_at_5_at_least_10_of_12": reranked_metrics["hit_at_5_count"] >= 10,
        "dense_required_coverage_at_least_075": dense_metrics["macro_required_coverage_at_5"] >= .75,
        "reranked_required_coverage_at_least_075": reranked_metrics["macro_required_coverage_at_5"] >= .75,
        "reranking_does_not_reduce_hits": reranked_metrics["hit_at_5_count"] >= dense_metrics["hit_at_5_count"],
        "reranking_does_not_reduce_mrr": reranked_metrics["mrr_at_20"] + 1e-12 >= dense_metrics["mrr_at_20"],
    }
    for name, metrics in [("dense", dense_metrics), ("reranked", reranked_metrics)]:
        coverage = metrics["source_coverage"]
        clause = next((c for c in coverage if c["case_id"] == "explicit-clause-13"), None)
        checks[name + "_exact_clause_in_top_5"] = bool(clause and clause["first_labelled_rank"] and clause["first_labelled_rank"] <= 5)
        checks[name + "_all_required_sources_in_top_5"] = all(c["all_labels_at_5"] for c in coverage if c["expected_match"] == "all")
    return {"status": "pass" if all(checks.values()) else "needs_review", "checks": checks,
            "interpretation": "Development sample gate only; never triggers a corpus build or production change."}


def batches(texts):
    batch, chars = [], 0
    for text in texts:
        if not text.strip() or len(text) > MAX_CHARS:
            raise ValueError("Text exceeds validated input limit")
        if batch and (len(batch) == 16 or chars + len(text) > 16384):
            yield batch
            batch, chars = [], 0
        batch.append(text)
        chars += len(text)
    if batch:
        yield batch


def live_compare(sample, cases, run_dir, fingerprint):
    # Explicit live mode alone reads local key configuration. Never loads other providers.
    import os
    from dotenv import dotenv_values
    from retrieval_validation_provider import SiliconFlowFreeProvider
    key = os.environ.get("SILICONFLOW_API_KEY")
    if not key:
        key = dotenv_values(ROOT / ".env").get("SILICONFLOW_API_KEY")
    if not key:
        raise ValueError("SILICONFLOW_API_KEY is absent; configure it locally, never in chat")
    provider = SiliconFlowFreeProvider(key)
    doc_batches = list(batches([c["content"] for c in sample]))
    query_batches = list(batches([c["question"] for c in cases]))
    attempts = len(doc_batches) + len(query_batches) + len(cases)
    if attempts > provider.MAX_REQUESTS:
        raise ValueError("Planned requests exceed the fixed experiment allowance")
    # No truncation: overlong rerank candidate sets stop before any inference calls.
    by_id = {c["id"]: c for c in sample}
    longest = sorted((len(c["content"]) for c in sample), reverse=True)[:TOP_K]
    if max(len(c["question"]) for c in cases) + sum(longest) > 16384:
        raise ValueError("Possible rerank set exceeds input allowance; curate shorter sample")
    document_vectors = [v for batch in doc_batches for v in provider.embed(batch)]
    query_vectors = [v for batch in query_batches for v in provider.embed(batch)]
    write_json(run_dir / "semantic_index.json", {
        "version": 1, "model": MODEL, "preprocessing": "unaltered-content-v1",
        "fingerprint": fingerprint, "dimensions": 1024,
        "chunk_ids": [c["id"] for c in sample], "vectors": document_vectors,
    })
    dense = dense_rankings(sample, document_vectors, cases, query_vectors)
    reranked = {}
    for case in cases:
        candidates = dense[case["id"]]
        documents = [by_id[item["chunk_id"]]["content"] for item in candidates]
        reranked[case["id"]] = [
            {"chunk_id": candidates[item["index"]]["chunk_id"], "relevance_score": item["relevance_score"]}
            for item in provider.rerank(case["question"], documents, top_n=len(documents))
        ]
    dense_metrics, reranked_metrics = summarize(cases, dense), summarize(cases, reranked)
    return {
        "status": "completed", "embedding_model": MODEL, "reranking_model": RERANKER,
        "planned_requests": attempts, "dense": {"metrics": dense_metrics, "rankings": dense},
        "reranked": {"metrics": reranked_metrics, "rankings": reranked},
        "acceptance": semantic_gate(dense_metrics, reranked_metrics),
    }


@contextmanager
def forbid_network():
    """Block Python socket connections and DNS throughout offline preparation."""
    def denied(*args, **kwargs):
        raise RuntimeError("Network access is disabled for offline validation")
    with patch.object(socket.socket, "connect", denied), patch.object(socket.socket, "connect_ex", denied), \
         patch.object(socket, "create_connection", denied), patch.object(socket, "getaddrinfo", denied):
        yield


def run(live=False, acknowledge_free_api=False):
    if not live:
        with forbid_network():
            return _run(live, acknowledge_free_api)
    return _run(live, acknowledge_free_api)


def _run(live=False, acknowledge_free_api=False):
    if live and not acknowledge_free_api:
        raise ValueError("Live mode requires --acknowledge-free-api after confirming current free-model prices")
    # Preserve all corpus evidence and pre-existing presentation/config files.
    preserved = [*sorted((ROOT / "data/source").glob("*.txt")),
                 *sorted((ROOT / "data/graph").rglob("*")), CORPUS,
                 ROOT / "data/processed/ingest_checkpoint.json"]
    preserved += [ROOT / rel for rel in [".gitignore", "README.md", "src/embeddings.py", "src/main.py",
                  "src/ui_backend.py", "src/demo_mode.py", ".editorconfig", ".gitattributes"]]
    before = {str(p.relative_to(ROOT)): digest(p) for p in preserved if p.is_file()}
    chunks, cases = load_evidence()
    sample = choose_sample(chunks, cases)
    fingerprint = {"corpus_sha256": digest(CORPUS), "cases_sha256": digest(CASES),
                   "sample_ids": [c["id"] for c in sample], "seed": SEED,
                   "sample_size": SAMPLE_SIZE, "max_characters_per_chunk": MAX_CHARS,
                   "top_k": TOP_K, "preprocessing": "unaltered-content-v1",
                   "sampling": "expected+full-corpus-lexical-top3+adjacent+seeded-random-v1",
                   "scorer_sha256": digest(ROOT / "src/demo_mode.py"),
                   "runner_sha256": digest(Path(__file__))}
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    run_dir = OUTPUT_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "sample_chunks.json", sample)
    write_json(run_dir / "cases_snapshot.json", json.loads(CASES.read_text(encoding="utf-8")))
    fingerprint["sample_sha256"] = digest(run_dir / "sample_chunks.json")
    write_json(run_dir / "manifest.json", fingerprint)
    rankings = lexical_rankings(run_dir / "sample_chunks.json", cases)
    report = {"version": 1, "run_id": run_id, "mode": "free-api-sample" if live else "offline-lexical",
              "sample_size": len(sample), "corpus_size": len(chunks), "case_count": len(cases),
              "fingerprint": fingerprint,
              "lexical": {"metrics": summarize(cases, rankings), "rankings": rankings},
              "semantic": {"status": "not_run", "reason": "Offline preparation makes zero API calls"}}
    try:
        if live:
            report["semantic"] = live_compare(sample, cases, run_dir, fingerprint)
    except Exception as error:
        # Only explicit adapter/value failures carry sanitized messages; other errors are generic.
        from retrieval_validation_provider import ProviderError
        report["semantic"] = {"status": "failed", "reason": str(error) if isinstance(error, (ProviderError, ValueError)) else "Experiment failed"}
    after = {rel: digest(ROOT / rel) for rel in before}
    report["preservation"] = {"files_checked": len(before), "unchanged": before == after,
                              "changed_files": [rel for rel in before if before[rel] != after[rel]]}
    write_json(run_dir / "report.json", report)
    if not report["preservation"]["unchanged"]:
        raise RuntimeError("Evidence preservation check failed")
    print(json.dumps({"report": str(run_dir / "report.json"), "mode": report["mode"],
                      "sample_size": len(sample), "case_count": len(cases),
                      "lexical_metrics": {k: v for k, v in report["lexical"]["metrics"].items() if k != "source_coverage"},
                      "semantic_status": report["semantic"]["status"],
                      "preservation": report["preservation"]}, ensure_ascii=True, indent=2))
    return 1 if report["semantic"]["status"] == "failed" else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Call only the pinned free SiliconFlow models")
    parser.add_argument("--acknowledge-free-api", action="store_true", help="Confirm current free-model pricing before live calls")
    args = parser.parse_args()
    try:
        sys.exit(run(args.live, args.acknowledge_free_api))
    except (ValueError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
