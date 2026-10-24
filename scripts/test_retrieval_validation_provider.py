"""Offline contract tests; no credentials, application imports, or inference."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from retrieval_validation_provider import ProviderError, SiliconFlowFreeProvider


class ProviderTests(unittest.TestCase):
    def provider(self, body=None, status=200, error=None):
        calls = []

        def transport(url, headers, payload, timeout):
            calls.append((url, payload, timeout))
            if error:
                raise error
            return status, body

        return SiliconFlowFreeProvider("test-secret", transport), calls

    def test_embedding_input_order_and_fixed_model(self):
        provider, calls = self.provider({"model": "BAAI/bge-m3", "data": [
            {"index": 1, "embedding": [2.0] * 1024},
            {"index": 0, "embedding": [1.0] * 1024},
        ]})
        self.assertEqual([row[0] for row in provider.embed(["a", "b"])], [1, 2])
        self.assertEqual(calls[0][0], "https://api.siliconflow.cn/v1/embeddings")
        self.assertEqual(calls[0][1]["model"], "BAAI/bge-m3")
        self.assertEqual(calls[0][2], 30)

    def test_bad_embedding_contracts(self):
        valid = {"index": 0, "embedding": [0.0] * 1024}
        for data in ([], [valid, valid], [{"index": 1, "embedding": [0.] * 1024}],
                     [{"index": 0, "embedding": [0.]}],
                     [{"index": 0, "embedding": [float("nan")] * 1024}],
                     [{"index": True, "embedding": [0.] * 1024}]):
            with self.subTest(data_length=len(data)):
                provider, _ = self.provider({"model": "BAAI/bge-m3", "data": data})
                with self.assertRaises(ProviderError):
                    provider.embed(["a"])

    def test_duplicate_embedding_indexes(self):
        item = {"index": 0, "embedding": [0.] * 1024}
        provider, _ = self.provider({"model": "BAAI/bge-m3", "data": [item, item]})
        with self.assertRaises(ProviderError):
            provider.embed(["a", "b"])

    def test_zero_norm_embedding_rejected(self):
        provider, _ = self.provider({"model": "BAAI/bge-m3", "data": [
            {"index": 0, "embedding": [0.] * 1024}
        ]})
        with self.assertRaisesRegex(ProviderError, "Zero-norm"):
            provider.embed(["a"])

    def test_twenty_rerank_candidates_separate_from_embedding_batch(self):
        provider, calls = self.provider({"results": [
            {"index": i, "relevance_score": i / 20} for i in range(20)
        ]})
        results = provider.rerank("q", ["document"] * 20, 20)
        self.assertEqual([item["index"] for item in results], list(reversed(range(20))))
        self.assertEqual(len(calls), 1)
        with self.assertRaises(ValueError):
            provider.rerank("q", ["document"] * 21, 20)
        with self.assertRaises(ValueError):
            provider.embed(["text"] * 20)
        self.assertEqual(len(calls), 1)

    def test_rerank_orders_scores_and_preserves_indexes(self):
        provider, calls = self.provider({"results": [
            {"index": 2, "relevance_score": .1}, {"index": 0, "relevance_score": .9}
        ]})
        self.assertEqual([x["index"] for x in provider.rerank("q", ["a", "b", "c"], 2)], [0, 2])
        self.assertEqual(calls[0][1]["model"], "BAAI/bge-reranker-v2-m3")
        self.assertFalse(calls[0][1]["return_documents"])

    def test_bad_rerank_contracts(self):
        for results in ([], [{"index": -1, "relevance_score": .2}],
                        [{"index": 0, "relevance_score": float("inf")}],
                        [{"index": 0, "relevance_score": 10 ** 400}],
                        [{"index": 0, "relevance_score": "secret"}]):
            provider, _ = self.provider({"results": results})
            with self.assertRaises(ProviderError):
                provider.rerank("q", ["a"], 1)
        provider, _ = self.provider({"results": [{"index": 0, "relevance_score": .1}] * 2})
        with self.assertRaises(ProviderError):
            provider.rerank("q", ["a", "b"], 2)

    def test_auth_rate_limit_and_exception_redaction_no_retry(self):
        for status in (401, 403, 429, 500, 302):
            provider, calls = self.provider({"message": "test-secret private input"}, status)
            with self.assertRaises(ProviderError) as caught:
                provider.embed(["private input"])
            self.assertNotIn("test-secret", str(caught.exception))
            self.assertNotIn("private input", str(caught.exception))
            self.assertEqual(len(calls), 1)
        provider, calls = self.provider(error=RuntimeError("test-secret private input"))
        with self.assertRaises(ProviderError) as caught:
            provider.embed(["private input"])
        self.assertNotIn("test-secret", str(caught.exception))
        self.assertTrue(caught.exception.__suppress_context__)
        self.assertEqual(len(calls), 1)

    def test_input_and_lifetime_limits_before_network(self):
        provider, calls = self.provider(status=429)
        for texts in ([], [""], ["x"] * 17, ["x" * 2049], ["x" * 2048] * 9):
            with self.assertRaises(ValueError):
                provider.embed(texts)
        self.assertEqual(calls, [])
        for _ in range(provider.MAX_REQUESTS):
            with self.assertRaises(ProviderError):
                provider.embed(["a"])
        with self.assertRaisesRegex(ProviderError, "limit reached"):
            provider.embed(["a"])
        self.assertEqual(len(calls), provider.MAX_REQUESTS)

    def test_no_model_or_endpoint_selection(self):
        with self.assertRaises(TypeError):
            SiliconFlowFreeProvider("test-secret", model="Pro/BAAI/bge-m3")
        with self.assertRaises(TypeError):
            SiliconFlowFreeProvider("test-secret", base_url="https://other.invalid")
        provider, _ = self.provider({"model": "Pro/BAAI/bge-m3", "data": []})
        with self.assertRaisesRegex(ProviderError, "Unexpected embedding model"):
            provider.embed(["a"])


if __name__ == "__main__":
    unittest.main()
