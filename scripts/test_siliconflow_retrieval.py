"""Offline runtime/build tests: no dotenv loading, provider calls or app imports."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import siliconflow_retrieval as runtime
import build_bge_index as builder
import restore_bge_index as restore


def vec(value=1.):
    return [value] * 1024


class FakeTime:
    def __init__(self):
        self.now = 0.
        self.sleeps = []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class RuntimeTests(unittest.TestCase):
    def client(self, replies):
        calls, timer = [], FakeTime()

        def transport(endpoint, payload, key):
            calls.append((endpoint, copy.deepcopy(payload)))
            response = replies.pop(0)
            if isinstance(response, Exception):
                raise response
            return response

        return runtime.FreeRetrievalClient("dummy-secret", transport, timer.clock, timer.sleep), calls, timer

    def test_langchain_order_fixed_model_and_unaltered_text(self):
        client, calls, _ = self.client([(200, {"model": runtime.EMBEDDING_MODEL, "data": [
            {"index": 1, "embedding": vec(2)}, {"index": 0, "embedding": vec(1)}]}, {})])
        embeddings = runtime.SiliconFlowEmbeddings(client)
        self.assertEqual([v[0] for v in embeddings.embed_documents([" 原文 ", "second"])], [1, 2])
        self.assertEqual(calls[0][1]["input"], [" 原文 ", "second"])
        self.assertEqual(calls[0][1]["model"], "BAAI/bge-m3")

    def test_zero_nan_dimensions_and_duplicate_rejected(self):
        for vector in ([0.] * 1024, [float("nan")] * 1024, [1.], [10 ** 400] * 1024):
            with self.assertRaises(runtime.RetrievalError):
                runtime.validate_vector(vector)
        client, _, _ = self.client([(200, {"model": runtime.EMBEDDING_MODEL, "data": [
            {"index": 0, "embedding": vec()}, {"index": 0, "embedding": vec()}]}, {})])
        with self.assertRaises(runtime.RetrievalError):
            client.embed(["a", "b"])

    def test_persisted_chroma_numpy_scalars_validate(self):
        import numpy as np
        values = list(np.ones(1024, dtype=np.float32))
        self.assertEqual(runtime.validate_vector(values), [1.] * 1024)

    def test_429_retry_after_bounded_and_no_auth_retry(self):
        client, calls, timer = self.client([(429, None, {"Retry-After": "3"})] * 3)
        with self.assertRaises(runtime.RetrievalError):
            client.embed(["a"])
        self.assertEqual(len(calls), 3)
        self.assertEqual(timer.sleeps, [3., 3.])
        for status in (401, 403, 500, 302):
            client, calls, _ = self.client([(status, {"secret": "dummy-secret"}, {})])
            with self.assertRaises(runtime.RetrievalError) as error:
                client.embed(["a"])
            self.assertNotIn("dummy-secret", str(error.exception))
            self.assertEqual(len(calls), 1)
        client, calls, _ = self.client([(429, None, {"Retry-After": "9999"})])
        with self.assertRaises(runtime.RetrievalError):
            client.embed(["a"])
        self.assertEqual(len(calls), 1)

    def test_transport_exception_sanitized(self):
        client, _, _ = self.client([ValueError("dummy-secret user-content")])
        with self.assertRaises(runtime.RetrievalError) as error:
            client.embed(["a"])
        self.assertNotIn("dummy-secret", str(error.exception))
        self.assertTrue(error.exception.__suppress_context__)

    def test_throttle_rpm_and_actual_token_feedback(self):
        client, _, timer = self.client([])
        for _ in range(61):
            client._reserve(1)
        self.assertGreaterEqual(timer.now, 60.)
        client, _, timer = self.client([(200, {"usage": {"total_tokens": 400001}}, {})])
        client._post("embeddings", {}, 100)
        client._reserve(1)
        self.assertGreaterEqual(timer.now, 60.)

    def test_actual_usage_reconciles_downward_and_missing_keeps_reservation(self):
        client, _, _ = self.client([(200, {"usage": {"total_tokens": 25}}, {}), (200, {}, {})])
        client._post("embeddings", {}, 100)
        self.assertEqual(client._events[-1][1], 25)
        client._post("embeddings", {}, 100)
        self.assertEqual(client._events[-1][1], 100)

    def test_rerank_20_result_mapping_and_no_model_selection(self):
        client, calls, _ = self.client([(200, {"results": [
            {"index": 19, "relevance_score": .8}, {"index": 0, "relevance_score": .9}]}, {})])
        docs = [str(i) for i in range(20)]
        result = client.rerank("q", docs, 2)
        self.assertEqual([row["text"] for row in result], ["0", "19"])
        self.assertEqual(calls[0][1]["model"], "BAAI/bge-reranker-v2-m3")
        with self.assertRaises(TypeError):
            runtime.FreeRetrievalClient("dummy", model="Pro/BAAI/bge-m3")
        with self.assertRaises(ValueError):
            client.rerank("q", docs + ["too many"], 1)

    def test_batch_limits_reject_no_truncation_and_allow_beyond_sample_chars(self):
        texts = ["a" * 2500] * 20
        grouped = list(runtime.batches(texts))
        self.assertEqual([t for batch in grouped for t in batch], texts)
        self.assertTrue(all(sum(runtime.text_tokens(t) for t in b) <= 16000 for b in grouped))
        with self.assertRaises(ValueError):
            list(runtime.batches(["x" * 8001]))

    def test_default_embeddings_shared(self):
        with patch.object(runtime, "_default", None):
            first = runtime.get_embeddings("dummy")
            self.assertIs(first, runtime.get_embeddings("dummy"))


class BuildTests(unittest.TestCase):
    def test_full_corpus_preflight_and_sample_reuse(self):
        chunks, fingerprint = builder.evidence()
        cache = builder.cached_vectors(chunks, fingerprint)
        result = builder.preflight(chunks, fingerprint, cache)
        self.assertEqual(result["new_vectors"] + result["reuse_sample_vectors"], 12204)
        self.assertEqual(result["max_utf8_bytes"], 2719)
        # Fresh checkouts omit the ignored sample cache; preflight stays useful.
        self.assertIn(len(cache), (0, 256))

    def test_cache_content_tampering_rejected(self):
        chunks, fingerprint = builder.evidence()
        cache = builder.cached_vectors(chunks, fingerprint)
        if not cache:
            self.skipTest("Ignored sample vectors are not present in this checkout")
        altered = copy.deepcopy(chunks)
        next(c for c in altered if c["id"] in cache)["content"] += "changed"
        with self.assertRaises(ValueError):
            builder.cached_vectors(altered, fingerprint)

    def test_actual_collection_provenance_verified(self):
        chunks = [{"id": "a", "content": "original", "metadata": {"book": "test"}}]

        class Collection:
            def count(self):
                return 1

            def get(self, **kwargs):
                return {"ids": ["a"], "documents": [self.text], "metadatas": [{"book": "test"}], "embeddings": [vec()]}

        collection = Collection()
        collection.text = "original"
        self.assertEqual(builder.stored_ids(collection, chunks), {"a"})
        collection.text = "modified"
        with self.assertRaises(ValueError):
            builder.stored_ids(collection, chunks)

    def test_versioned_path_confined_and_no_existing_overwrite(self):
        builder.INDEX_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=builder.INDEX_ROOT) as temp:
            target = Path(temp)
            self.assertEqual(builder.index_directory(str(target)), target.resolve())
            with self.assertRaises(ValueError):
                builder.index_directory(str(builder.INDEX_ROOT))

    def test_crash_after_insert_resume_uses_actual_ids_and_checkpoint_lags(self):
        import chromadb
        chunks = [{"id": "a", "content": "original", "metadata": {"book": "test"}}]
        fingerprint = {"corpus_sha256": builder.sha(builder.CORPUS), "test": True}

        class Collection:
            metadata = None

            def __init__(self):
                self.records = {}
                self.add_calls = 0

            def count(self):
                return len(self.records)

            def get(self, **kwargs):
                return {"ids": list(self.records),
                        "documents": [r[0] for r in self.records.values()],
                        "metadatas": [r[1] for r in self.records.values()],
                        "embeddings": [r[2] for r in self.records.values()]}

            def add(self, ids, documents, metadatas, embeddings):
                self.add_calls += 1
                for i, cid in enumerate(ids):
                    self.records[cid] = (documents[i], metadatas[i], embeddings[i])
                # Simulate successful durable write followed by process failure.
                raise RuntimeError("simulated crash")

        class Database:
            def get_or_create_collection(self, name, embedding_function, metadata):
                collection.metadata = metadata
                return collection

        class Client:
            requests = 0

            def embed(self, texts):
                raise AssertionError("Sample vector must be reused")

        collection = Collection()
        builder.INDEX_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=builder.INDEX_ROOT) as directory:
            with patch.object(builder, "INDEX_ROOT", Path(directory)), \
                 patch.object(chromadb, "PersistentClient", return_value=Database()):
                with self.assertRaisesRegex(RuntimeError, "simulated crash"):
                    builder.build(chunks, fingerprint, {"a": vec()}, Client())
                target = next(Path(directory).iterdir())
                checkpoint = json.loads((target / "checkpoint.json").read_text(encoding="utf-8"))
                self.assertEqual(checkpoint["inserted_count"], 0)
                resumed = builder.build(chunks, fingerprint, {"a": vec()}, Client(), str(target))
                manifest = json.loads((resumed / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["status"], "complete")
                self.assertEqual(manifest["verified_count"], 1)
                self.assertEqual(collection.add_calls, 1)
                self.assertEqual(manifest["portable_export"]["count"], 1)
                self.assertTrue((target / "portable_vectors.npz").is_file())


class PortableRestoreTests(unittest.TestCase):
    SOURCE = ROOT / "vectorstore/bge-m3/20261024T015651Z-223a0f34"

    def portable_source(self):
        if not (self.SOURCE / "portable_manifest.json").is_file():
            packaged = ROOT / "deploy/bge-m3"
            if not (packaged / "portable_manifest.json").is_file():
                self.skipTest("Portable artifact verification requires a built or packaged index")
            self.SOURCE = packaged
        return self.SOURCE

    def test_real_portable_export_hashes_dimensions_and_alignment(self):
        with restore.offline():
            manifest, portable, sources, vectors = restore.load_portable(self.portable_source())
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(vectors.shape, (12204, 1024))
        self.assertEqual(portable["model"], "BAAI/bge-m3")
        self.assertEqual(len(sources), 12204)

    def test_hash_mismatch_rejected_before_destination_creation(self):
        source = self.portable_source()
        original = restore.sha

        def changed(path):
            return "0" * 64 if Path(path).name == "portable_vectors.npz" else original(path)

        with patch.object(restore, "sha", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                restore.load_portable(source)

    def test_offline_guard_denies_network(self):
        import socket
        with restore.offline():
            with self.assertRaisesRegex(RuntimeError, "disabled"):
                socket.create_connection(("example.invalid", 443))

    def test_restore_allows_float32_normalization_roundoff_not_corruption(self):
        import numpy as np
        sources = [{"id": "a", "content": "original", "metadata": {"book": "test"}}]
        expected = np.ones((1, 1024), dtype=np.float32) / 32

        class Collection:
            delta = 1e-8

            def count(self):
                return 1

            def get(self, **kwargs):
                return {"ids": ["a"], "documents": ["original"], "metadatas": [{"book": "test"}],
                        "embeddings": [expected[0] + self.delta]}

        collection = Collection()
        self.assertEqual(restore.reconcile(collection, sources, expected), {"a"})
        collection.delta = .1
        with self.assertRaisesRegex(ValueError, "differs"):
            restore.reconcile(collection, sources, expected)


if __name__ == "__main__":
    unittest.main()
