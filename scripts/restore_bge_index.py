"""Rebuild native Chroma/HNSW offline from verified portable BGE vectors.

Input folder needs manifest.json, portable_manifest.json, portable_vectors.npz,
and portable_sources.json. This tool never loads credentials or a model client.
By default destination must not exist; --resume reconciles actual saved records.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import socket
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
COLLECTION = "tcm_bge_m3"
MODEL = "BAAI/bge-m3"
DIMENSIONS = 1024
PREPROCESSING = "unaltered-content-v1"
VECTOR_RTOL = 1e-5
VECTOR_ATOL = 1e-7


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temp, path)


@contextmanager
def offline():
    def denied(*args, **kwargs):
        raise RuntimeError("Network is disabled during portable index restore")
    with patch.object(socket.socket, "connect", denied), patch.object(socket.socket, "connect_ex", denied), \
         patch.object(socket, "create_connection", denied), patch.object(socket, "getaddrinfo", denied):
        yield


def load_portable(source):
    import numpy as np
    source = Path(source).resolve()
    portable = json.loads((source / "portable_manifest.json").read_text(encoding="utf-8"))
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    expected = {"model": MODEL, "dimensions": DIMENSIONS, "preprocessing": PREPROCESSING,
                "distance": "cosine", "count": 12204, "dtype": "float32",
                "format": "numpy-npz-no-pickle", "vectors_file": "portable_vectors.npz",
                "sources_file": "portable_sources.json"}
    if any(portable.get(key) != value for key, value in expected.items()):
        raise ValueError("Portable model/configuration does not match the pinned BGE index")
    fingerprint = manifest.get("fingerprint", {})
    if manifest.get("status") != "complete" or manifest.get("verified_count") != 12204:
        raise ValueError("Source index is not complete and verified")
    for key in ("model", "dimensions", "preprocessing", "distance", "count"):
        if fingerprint.get(key) != expected[key]:
            raise ValueError("Source and portable manifest configurations disagree")
    if fingerprint.get("collection") != COLLECTION or manifest.get("portable_export") != portable:
        raise ValueError("Source collection or portable provenance mismatch")
    corpus_hash = fingerprint.get("corpus_sha256")
    if not isinstance(corpus_hash, str) or len(corpus_hash) != 64 or any(c not in "0123456789abcdef" for c in corpus_hash):
        raise ValueError("Corpus fingerprint is invalid")
    vectors_path, sources_path = source / "portable_vectors.npz", source / "portable_sources.json"
    for path, key in ((vectors_path, "vectors_sha256"), (sources_path, "sources_sha256")):
        if sha(path) != portable.get(key):
            raise ValueError("Portable file hash mismatch")
    sources = json.loads(sources_path.read_text(encoding="utf-8"))
    ids = [c["id"] for c in sources]
    if len(ids) != 12204 or len(set(ids)) != len(ids) or not all(isinstance(cid, str) and cid for cid in ids):
        raise ValueError("Portable source IDs/count mismatch")
    original_hashes = manifest.get("source_content_hashes", {})
    if set(original_hashes) != set(ids):
        raise ValueError("Original source provenance is incomplete")
    for source_record in sources:
        text, metadata = source_record["content"], source_record["metadata"]
        if not isinstance(text, str) or not text.strip() or not isinstance(metadata, dict) or not metadata:
            raise ValueError("Portable source record is invalid")
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != original_hashes[source_record["id"]]:
            raise ValueError("Portable text differs from original source provenance")
        for value in metadata.values():
            if type(value) not in (str, int, float, bool) or (type(value) is float and not math.isfinite(value)):
                raise ValueError("Portable metadata is not scalar-compatible")
    with np.load(vectors_path, allow_pickle=False) as archive:
        if set(archive.files) != {"ids", "vectors"}:
            raise ValueError("Unexpected portable archive fields")
        vectors = archive["vectors"]
        if archive["ids"].tolist() != ids or vectors.dtype != np.dtype("float32") or vectors.shape != (12204, DIMENSIONS):
            raise ValueError("Portable vectors are misaligned or have wrong dimensions/dtype")
        if not np.isfinite(vectors).all() or not np.all(np.any(vectors != 0, axis=1)):
            raise ValueError("Portable vectors must be finite and nonzero")
    return manifest, portable, sources, vectors


def reconcile(collection, sources, expected_vectors):
    import numpy as np
    known = {c["id"]: c for c in sources}
    positions = {c["id"]: i for i, c in enumerate(sources)}
    seen = set()
    count = collection.count()
    for offset in range(0, count, 128):
        page = collection.get(limit=128, offset=offset, include=["documents", "metadatas", "embeddings"])
        for cid, text, metadata, vector in zip(page["ids"], page["documents"], page["metadatas"], page["embeddings"]):
            if cid not in known or cid in seen or text != known[cid]["content"] or metadata != known[cid]["metadata"]:
                raise ValueError("Persisted record/source provenance mismatch")
            array = np.asarray(vector)
            if array.shape != (DIMENSIONS,) or not np.isfinite(array).all() or not np.any(array):
                raise ValueError("Persisted vector validation failed")
            # Chroma cosine collections normalize again on native insertion;
            # a float32 roundtrip can alter the last bits. Bound that drift.
            if not np.allclose(array.astype(np.float32), expected_vectors[positions[cid]],
                               rtol=VECTOR_RTOL, atol=VECTOR_ATOL):
                raise ValueError("Persisted vector differs from the verified portable export")
            seen.add(cid)
    if len(seen) != count:
        raise ValueError("Persisted source count mismatch")
    return seen


def restore(source, destination, resume=False):
    import chromadb
    from chromadb.config import Settings
    # Verify every artifact before creating/opening a destination.
    manifest, portable, sources, vectors = load_portable(source)
    destination = Path(destination).resolve()
    if destination == Path(source).resolve() or destination in Path(source).resolve().parents:
        raise ValueError("Restore destination must be separate from source artifacts")
    marker = destination / "manifest.json"
    provenance = sha(Path(source) / "portable_manifest.json")
    if destination.exists():
        if not resume:
            raise FileExistsError("Destination exists; use explicit --resume for the same artifacts")
        previous = json.loads(marker.read_text(encoding="utf-8"))
        if previous.get("fingerprint") != manifest["fingerprint"] or previous.get("portable_manifest_sha256") != provenance:
            raise ValueError("Resume destination provenance mismatch")
    else:
        if resume:
            raise ValueError("Resume destination does not exist")
        destination.mkdir(parents=True, exist_ok=False)
    restored = {"fingerprint": manifest["fingerprint"], "status": "building",
                "index_directory": str(destination), "portable_manifest_sha256": provenance,
                "source_content_hashes": manifest["source_content_hashes"], "portable_export": portable}
    save(marker, restored)
    db = chromadb.PersistentClient(path=str(destination / "chroma"), settings=Settings(anonymized_telemetry=False))
    metadata = {"hnsw:space": "cosine", "embedding_model": MODEL, "dimensions": DIMENSIONS,
                "preprocessing": PREPROCESSING, "corpus_sha256": manifest["fingerprint"]["corpus_sha256"]}
    collection = db.get_or_create_collection(COLLECTION, embedding_function=None, metadata=metadata)
    if collection.metadata != metadata:
        raise ValueError("Restore collection configuration mismatch")
    existing = reconcile(collection, sources, vectors)
    pending = [i for i, c in enumerate(sources) if c["id"] not in existing]
    for start in range(0, len(pending), 128):
        indices = pending[start:start + 128]
        collection.add(ids=[sources[i]["id"] for i in indices],
                       documents=[sources[i]["content"] for i in indices],
                       metadatas=[sources[i]["metadata"] for i in indices], embeddings=vectors[indices].tolist())
    if len(reconcile(collection, sources, vectors)) != 12204:
        raise ValueError("Restored full index count mismatch")
    restored.update({"status": "complete", "verified_count": 12204,
                     "restored_at": datetime.now(timezone.utc).isoformat(),
                     "restore_mode": "offline-native-hnsw-rebuild",
                     "float32_readback_tolerance": {"rtol": VECTOR_RTOL, "atol": VECTOR_ATOL}})
    save(marker, restored)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    with offline():
        target = restore(args.source, args.destination, args.resume)
    print(json.dumps({"status": "complete", "verified_count": 12204, "index_directory": str(target)}))


if __name__ == "__main__":
    main()
