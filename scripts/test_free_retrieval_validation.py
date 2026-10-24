"""Offline experiment integrity and interpretation tests; no provider calls."""

import json
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_free_retrieval as experiment


class ExperimentTests(unittest.TestCase):
    def test_multi_source_coverage_is_not_an_any_hit(self):
        cases = [
            {"id": "cross", "negative_control": False, "expected_match": "all", "expected_chunk_ids": ["a", "b"]},
            {"id": "alternate", "negative_control": False, "expected_match": "any", "expected_chunk_ids": ["a", "b"]},
            {"id": "negative", "negative_control": True, "expected_match": "none", "expected_chunk_ids": []},
        ]
        rankings = {c["id"]: [{"chunk_id": "a"}] for c in cases}
        metrics = experiment.summarize(cases, rankings)
        self.assertEqual(metrics["hit_at_5"], 1)
        self.assertEqual(metrics["macro_required_coverage_at_5"], .75)
        self.assertFalse(metrics["source_coverage"][0]["all_labels_at_5"])
        self.assertEqual(metrics["positive_cases"], 2)

    def test_dense_ranking_uses_cosine_and_source_ids(self):
        sample = [{"id": "wrong"}, {"id": "right"}]
        cases = [{"id": "q"}]
        rankings = experiment.dense_rankings(sample, [[0., 2.], [5., 0.]], cases, [[2., 0.]])
        self.assertEqual(rankings["q"][0]["chunk_id"], "right")
        self.assertEqual(rankings["q"][0]["cosine"], 1.)
        with self.assertRaises(ValueError):
            experiment.dense_rankings(sample, [[0., 0.], [5., 0.]], cases, [[2., 0.]])
        with self.assertRaises(ValueError):
            experiment.dense_rankings(sample, [[0., 2.], [5., 0.]], cases, [[2.]])

    def test_gate_rejects_missing_exact_clause_and_reranking_regressions(self):
        good = {"hit_at_5_count": 11, "mrr_at_20": .8, "macro_required_coverage_at_5": .9,
                "source_coverage": [{"case_id": "explicit-clause-13", "first_labelled_rank": 2,
                                     "expected_match": "any", "all_labels_at_5": True}]}
        self.assertEqual(experiment.semantic_gate(good, good)["status"], "pass")
        missing = {**good, "source_coverage": []}
        self.assertEqual(experiment.semantic_gate(missing, missing)["status"], "needs_review")
        worse = {**good, "mrr_at_20": .7}
        self.assertFalse(experiment.semantic_gate(good, worse)["checks"]["reranking_does_not_reduce_mrr"])

    def test_batch_budgets_preserve_text_and_order(self):
        texts = ["x" * 2000 for _ in range(20)]
        batches = list(experiment.batches(texts))
        self.assertEqual([t for batch in batches for t in batch], texts)
        self.assertTrue(all(len(batch) <= 16 and sum(map(len, batch)) <= 16384 for batch in batches))
        with self.assertRaises(ValueError):
            list(experiment.batches(["x" * 2049]))

    def test_network_guard_blocks_connect_and_dns(self):
        with experiment.forbid_network():
            with self.assertRaisesRegex(RuntimeError, "disabled"):
                socket.create_connection(("example.invalid", 443))
            with self.assertRaisesRegex(RuntimeError, "disabled"):
                socket.getaddrinfo("example.invalid", 443)

    def test_live_mode_needs_ack_before_data_or_credentials(self):
        with patch.object(experiment, "load_evidence", side_effect=AssertionError("must not load")):
            with self.assertRaisesRegex(ValueError, "acknowledge-free-api"):
                experiment.run(live=True)

    def test_output_never_overwrites_previous_artifact(self):
        experiment.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=experiment.OUTPUT_ROOT) as directory:
            self.assertTrue(Path(directory).resolve().is_relative_to(experiment.ROOT.resolve()))
            target = Path(directory) / "artifact.json"
            experiment.write_json(target, {"kept": True})
            with self.assertRaises(FileExistsError):
                experiment.write_json(target, {"kept": False})
            self.assertEqual(json.loads(target.read_text(encoding="utf-8")), {"kept": True})

    def test_fixture_quotes_match_immutable_corpus(self):
        chunks, cases = experiment.load_evidence()
        self.assertEqual(len(chunks), 12204)
        self.assertEqual(sum(not c["negative_control"] for c in cases), 12)
        self.assertEqual(sum(c["negative_control"] for c in cases), 3)


if __name__ == "__main__":
    unittest.main()
