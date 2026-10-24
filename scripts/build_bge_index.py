"""Preflight by default; opt-in resumable full index in a separate versioned path.

Never writes corpus files or the legacy ingest checkpoint. No generation calls.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from siliconflow_retrieval import (
    DIMENSIONS, EMBEDDING_MODEL, PREPROCESSING, FreeRetrievalClient, TOKENS_PER_MINUTE, RetrievalError,
    batches, text_tokens, validate_vector,
)

CORPUS = ROOT / "data/processed/chunks.json"
INDEX_ROOT = ROOT / "vectorstore/bge-m3"
SAMPLE_RUN = ROOT / "vectorstore/retrieval-validation/20261024T014413Z-ecba9244"
COLLECTION = "tcm_bge_m3"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def content_sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")
    os.replace(temp, path)


def evidence(corpus=CORPUS):
    chunks = json.loads(corpus.read_text(encoding="utf-8"))
    if len(chunks) != 12204:
        raise ValueError("Expected exactly 12204 unchanged corpus chunks")
    ids = [c["id"] for c in chunks]
    if len(set(ids)) != len(ids) or not all(isinstance(i, str) and i for i in ids):
        raise ValueError("Corpus IDs must be unique nonempty strings")
    for chunk in chunks:
        text_tokens(chunk["content"])
        if not isinstance(chunk["metadata"], dict) or not chunk["metadata"]:
            raise ValueError("Corpus metadata is missing")
        if any(type(v) not in (str, int, float, bool) for v in chunk["metadata"].values()):
            raise ValueError("Corpus metadata is not Chroma scalar-compatible")
    fingerprint = {
        "corpus_sha256": sha(corpus), "model": EMBEDDING_MODEL,
        "dimensions": DIMENSIONS, "preprocessing": PREPROCESSING,
        "count": len(chunks), "collection": COLLECTION, "distance": "cosine",
        "input_limit_utf8_bytes": 8000, "schema_version": 1,
    }
    return chunks, fingerprint


def cached_vectors(chunks, fingerprint, run=SAMPLE_RUN):
    """Only exact corpus/model/preprocessing and ID/content-matching samples."""
    index_path = run / "semantic_index.json"
    if not index_path.is_file():
        return {}
    index = json.loads(index_path.read_text(encoding="utf-8"))
    sample_path = run / "sample_chunks.json"
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    prior = index.get("fingerprint", {})
    for key, value in (("model", EMBEDDING_MODEL), ("dimensions", DIMENSIONS),
                       ("preprocessing", PREPROCESSING)):
        if index.get(key) != value:
            raise ValueError("Sample embedding configuration mismatch")
    if prior.get("corpus_sha256") != fingerprint["corpus_sha256"] or prior.get("sample_sha256") != sha(sample_path):
        raise ValueError("Sample corpus or content fingerprint mismatch")
    ids, vectors = index["chunk_ids"], index["vectors"]
    if len(ids) != len(vectors) or len(set(ids)) != len(ids):
        raise ValueError("Sample vector ID/count mismatch")
    by_id = {c["id"]: c for c in chunks}
    sample_by_id = {c["id"]: c for c in sample}
    if set(ids) != set(sample_by_id):
        raise ValueError("Sample source ID mismatch")
    cache = {}
    for chunk_id, vector in zip(ids, vectors):
        if chunk_id not in by_id or sample_by_id[chunk_id] != by_id[chunk_id]:
            raise ValueError("Sample source content or metadata differs")
        cache[chunk_id] = validate_vector(vector)
    return cache


def preflight(chunks, fingerprint, cache):
    pending = [c for c in chunks if c["id"] not in cache]
    estimates = [text_tokens(c["content"]) for c in pending]
    return {
        "status": "preflight_ready", "fingerprint": fingerprint,
        "max_characters": max(len(c["content"]) for c in chunks),
        "max_utf8_bytes": max(len(c["content"].encode("utf-8")) for c in chunks),
        "reuse_sample_vectors": len(cache), "new_vectors": len(pending),
        "planned_embedding_batches": len(list(batches([c["content"] for c in pending]))),
        "conservative_reserved_tokens": sum(estimates),
        "throttle_minutes_budget": math.ceil(sum(estimates) / TOKENS_PER_MINUTE),
        "requests_per_minute_ceiling": 60, "tokens_per_minute_ceiling": TOKENS_PER_MINUTE,
        "note": "Byte-based reservation is conservative; no truncation or model calls in preflight.",
    }


def index_directory(resume=None):
    INDEX_ROOT.mkdir(parents=True, exist_ok=True)
    if resume:
        target = Path(resume).resolve()
        if target.parent != INDEX_ROOT.resolve() or not target.is_dir():
            raise ValueError("Resume must name a direct existing versioned BGE index directory")
    else:
        name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
        target = INDEX_ROOT / name
        target.mkdir(exist_ok=False)
    return target


def stored_ids(collection, chunks):
    """Reconcile actual durable records, never trust checkpoint completion IDs."""
    known = {c["id"]: c for c in chunks}
    seen = set()
    total = collection.count()
    for offset in range(0, total, 128):
        page = collection.get(limit=128, offset=offset, include=["documents", "metadatas", "embeddings"])
        for chunk_id, text, metadata, vector in zip(page["ids"], page["documents"], page["metadatas"], page["embeddings"]):
            if chunk_id not in known or chunk_id in seen:
                raise ValueError("Unexpected or duplicate persisted source ID")
            source = known[chunk_id]
            if text != source["content"] or metadata != source["metadata"]:
                raise ValueError("Persisted source provenance mismatch")
            validate_vector(list(vector))
            seen.add(chunk_id)
    if len(seen) != total:
        raise ValueError("Persisted record count mismatch")
    return seen


def portable_export(target, collection, chunks):
    """Cross-platform float32 export; Linux must rebuild HNSW from these vectors.

    No pickle, model call, platform-specific HNSW reuse, or token reprocessing.
    """
    import numpy as np
    ids = [c["id"] for c in chunks]
    positions = {cid: i for i, cid in enumerate(ids)}
    vectors = np.empty((len(ids), DIMENSIONS), dtype=np.float32)
    for offset in range(0, len(ids), 128):
        page = collection.get(limit=128, offset=offset, include=["embeddings"])
        for cid, raw in zip(page["ids"], page["embeddings"]):
            vector = np.asarray(validate_vector(list(raw)), dtype=np.float32)
            if not np.isfinite(vector).all() or not np.any(vector):
                raise ValueError("Portable vector cannot be represented safely in float32")
            vectors[positions[cid]] = vector
    vectors_path = target / "portable_vectors.npz"
    temp = target / "portable_vectors.npz.tmp"
    with temp.open("wb") as handle:
        np.savez_compressed(handle, ids=np.asarray(ids), vectors=vectors)
    os.replace(temp, vectors_path)
    sources_path = target / "portable_sources.json"
    atomic_json(sources_path, chunks)
    # Read back without pickle, validate alignment, shape and finite/nonzero rows.
    with np.load(vectors_path, allow_pickle=False) as stored:
        if stored["ids"].tolist() != ids or stored["vectors"].shape != (len(ids), DIMENSIONS):
            raise ValueError("Portable export source/vector alignment mismatch")
        if not np.isfinite(stored["vectors"]).all() or not np.all(np.any(stored["vectors"] != 0, axis=1)):
            raise ValueError("Portable export contains invalid vectors")
    result = {"format": "numpy-npz-no-pickle", "dtype": "float32", "count": len(ids),
              "dimensions": DIMENSIONS, "model": EMBEDDING_MODEL,
              "preprocessing": PREPROCESSING, "distance": "cosine",
              "vectors_file": vectors_path.name, "vectors_sha256": sha(vectors_path),
              "sources_file": sources_path.name, "sources_sha256": sha(sources_path),
              "linux_note": "Rebuild Chroma/HNSW offline using these exact vectors and source IDs."}
    atomic_json(target / "portable_manifest.json", result)
    return result


def build(chunks, fingerprint, cache, client, resume=None):
    import chromadb
    from chromadb.config import Settings
    target = index_directory(resume)
    manifest_path, checkpoint_path = target / "manifest.json", target / "checkpoint.json"
    if resume:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("fingerprint") != fingerprint or manifest.get("index_directory") != str(target):
            raise ValueError("Resume index fingerprint/path mismatch")
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("fingerprint") != fingerprint:
            raise ValueError("Resume checkpoint fingerprint mismatch")
    else:
        manifest = {"fingerprint": fingerprint, "index_directory": str(target), "status": "building",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "source_content_hashes": {c["id"]: content_sha(c["content"]) for c in chunks}}
        atomic_json(manifest_path, manifest)
        atomic_json(checkpoint_path, {"fingerprint": fingerprint, "inserted_count": 0})
    print(json.dumps({"status": "building", "index_directory": str(target)}), flush=True)
    db = chromadb.PersistentClient(path=str(target / "chroma"), settings=Settings(anonymized_telemetry=False))
    expected_metadata = {"hnsw:space": "cosine", "embedding_model": EMBEDDING_MODEL,
                         "dimensions": DIMENSIONS, "preprocessing": PREPROCESSING,
                         "corpus_sha256": fingerprint["corpus_sha256"]}
    collection = db.get_or_create_collection(
        COLLECTION, embedding_function=None, metadata=expected_metadata,
    )
    if collection.metadata != expected_metadata:
        raise ValueError("Existing collection configuration mismatch")
    inserted = stored_ids(collection, chunks)
    pending = [c for c in chunks if c["id"] not in inserted]
    # Resume reuses already persisted records even if checkpoint lagged after a crash.
    for batch_texts in batches([c["content"] for c in pending]):
        group = pending[:len(batch_texts)]
        pending = pending[len(group):]
        missing = [c for c in group if c["id"] not in cache]
        fresh = client.embed([c["content"] for c in missing]) if missing else []
        mapping = {c["id"]: v for c, v in zip(missing, fresh)}
        vectors = [cache[c["id"]] if c["id"] in cache else mapping[c["id"]] for c in group]
        collection.add(ids=[c["id"] for c in group], documents=[c["content"] for c in group],
                       metadatas=[c["metadata"] for c in group], embeddings=vectors)
        inserted.update(c["id"] for c in group)
        atomic_json(checkpoint_path, {"fingerprint": fingerprint, "inserted_count": len(inserted),
                                    "last_successful_ids": [c["id"] for c in group]})
        print(json.dumps({"inserted": len(inserted), "total": len(chunks), "requests": client.requests}), flush=True)
    verified = stored_ids(collection, chunks)
    if len(verified) != len(chunks) or sha(CORPUS) != fingerprint["corpus_sha256"]:
        raise ValueError("Final count or corpus preservation verification failed")
    portable = portable_export(target, collection, chunks)
    atomic_json(checkpoint_path, {"fingerprint": fingerprint, "inserted_count": len(verified)})
    manifest.update({"status": "complete", "verified_count": len(verified),
                     "completed_at": datetime.now(timezone.utc).isoformat(),
                     "requests_this_process": client.requests, "reused_sample_vectors": len(cache),
                     "portable_export": portable})
    atomic_json(manifest_path, manifest)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--acknowledge-free-api", action="store_true")
    parser.add_argument("--resume")
    args = parser.parse_args()
    if args.build and not args.acknowledge_free_api:
        raise ValueError("Build requires explicit acknowledgement of fixed free-model calls")
    chunks, fingerprint = evidence()
    cache = cached_vectors(chunks, fingerprint)
    print(json.dumps(preflight(chunks, fingerprint, cache), ensure_ascii=True, indent=2), flush=True)
    if not args.build:
        return
    # Only explicit build mode reads a local key, and it never outputs it.
    from dotenv import dotenv_values
    key = os.environ.get("SILICONFLOW_API_KEY") or dotenv_values(ROOT / ".env").get("SILICONFLOW_API_KEY")
    if not key:
        raise ValueError("SILICONFLOW_API_KEY is missing")
    result = build(chunks, fingerprint, cache, FreeRetrievalClient(key), args.resume)
    print(json.dumps({"status": "complete", "index_directory": str(result)}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Neither provider bodies nor exception details may contain credentials.
        print(json.dumps({"status": "failed", "error_type": type(error).__name__,
                          "reason": str(error) if isinstance(error, RetrievalError) else
                          "Build stopped safely; inspect local state and rerun preflight"}), file=sys.stderr)
        sys.exit(1)
