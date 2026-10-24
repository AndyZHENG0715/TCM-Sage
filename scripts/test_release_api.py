"""Offline ASGI release checks; dotenv and all provider/runtime work are mocked."""

import importlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

# Patch BEFORE import: no key configuration or .env file may be loaded by tests.
with patch.dict(os.environ, {}, clear=True), patch('dotenv.load_dotenv', return_value=False) as blocked_dotenv:
    api = importlib.import_module('release_api')
    blocked_dotenv.assert_called_once()

from fastapi.testclient import TestClient
from release_limits import GateError


class FakeLease:
    def __init__(self):
        self.released = False
        self.physical_releases = 0

    def release(self):
        if not self.released:
            self.released = True
            self.physical_releases += 1


class ReleaseApiTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {'LIVE_ENABLED': '1'}, clear=True)
        self.env.start()
        self.lease = FakeLease()
        self.gate = Mock()
        self.gate.acquire.side_effect = self.acquire
        self.gate_patch = patch.object(api, 'gate', return_value=self.gate)
        self.gate_patch.start()
        model = api.runtime.CHAT_MODEL
        self.runtime_patch = patch.object(api, 'runtime')
        self.runtime = self.runtime_patch.start()
        self.runtime.CHAT_MODEL = model
        self.runtime.run_query.return_value = iter(['answer [1]', {'type': 'metadata', 'citations': []}])
        self.client = TestClient(api.app, raise_server_exceptions=False)

    def tearDown(self):
        self.client.close()
        self.runtime_patch.stop()
        self.gate_patch.stop()
        self.env.stop()

    def acquire(self, authorization, client_ip):
        if os.environ.get('LIVE_ENABLED') != '1':
            raise GateError(503, 'Live access is disabled')
        if authorization != 'Bearer synthetic-access-code':
            raise GateError(401, 'Valid live access code required')
        return self.lease

    def post(self, body=None, **kwargs):
        return self.client.post('/query', json=body or {'question': 'historical text question'},
                                headers={'Authorization': 'Bearer synthetic-access-code'}, **kwargs)

    def test_missing_and_invalid_code_never_start_runtime(self):
        for headers in ({}, {'Authorization': 'Bearer wrong'}):
            response = self.client.post('/query', json={'question': 'historical text question'}, headers=headers)
            self.assertEqual(response.status_code, 401)
        self.runtime.run_query.assert_not_called()
        self.assertFalse(self.lease.released)

    def test_live_off_never_starts_runtime(self):
        os.environ['LIVE_ENABLED'] = '0'
        response = self.post()
        self.assertEqual(response.status_code, 503)
        self.runtime.run_query.assert_not_called()

    def test_body_and_question_bounds_precede_admission(self):
        for body, status in [({'question': 'x' * 1001}, 422),
                             ({'question': '   '}, 400),
                             ({'question': ''}, 422),
                             ({'question': 'x', 'chat_history': [{}] * 7}, 422)]:
            with self.subTest(status=status):
                self.assertEqual(self.post(body).status_code, status)
        response = self.client.post('/query', content=b'x' * 16385,
                                    headers={'Content-Type': 'application/json'})
        self.assertEqual(response.status_code, 413)
        self.gate.acquire.assert_not_called()
        self.runtime.run_query.assert_not_called()

    def test_provider_and_retrieval_override_rejected(self):
        for settings in ({'provider': 'paid-provider'}, {'model': 'paid-model'}, {'retrieval_k': 999}):
            self.assertEqual(self.post({'question': 'history', 'settings': settings}).status_code, 400)
        self.assertEqual(self.post({'question': 'history', 'provider': 'paid-provider'}).status_code, 422)
        self.gate.acquire.assert_not_called()
        self.runtime.run_query.assert_not_called()

    def test_sse_contract_and_lease_release(self):
        self.runtime.run_query.return_value = iter(['first\nsecond', {'type': 'metadata', 'citations': [], 'severity': 'informational'}])
        response = self.post()
        self.assertEqual(response.status_code, 200)
        self.assertIn('text/event-stream', response.headers['content-type'])
        self.assertEqual(response.headers['cache-control'], 'no-store')
        self.assertIn('data: first\\nsecond\n\n', response.text)
        self.assertIn('event: metadata\ndata: ', response.text)
        self.assertIn('"citations": []', response.text)
        self.assertTrue(self.lease.released)
        self.assertEqual(self.lease.physical_releases, 1)

    def test_provider_error_redacted_and_releases_lease(self):
        def broken(question):
            yield 'partial'
            raise RuntimeError('sensitive-provider-payload fake-secret')
        self.runtime.run_query.side_effect = broken
        response = self.post()
        self.assertEqual(response.status_code, 200)
        self.assertIn('event: error', response.text)
        self.assertNotIn('sensitive-provider-payload', response.text)
        self.assertNotIn('fake-secret', response.text)
        self.assertTrue(self.lease.released)
        self.assertEqual(self.lease.physical_releases, 1)

    def test_runtime_creation_error_releases_lease(self):
        self.runtime.run_query.side_effect = RuntimeError('secret-value')
        response = self.post()
        self.assertIn('event: error', response.text)
        self.assertNotIn('secret-value', response.text)
        self.assertTrue(self.lease.released)

    def test_public_read_routes_use_read_gate(self):
        self.runtime.source_context.return_value = {'chunk_id': 'original-id', 'context_scope': 'cited_chunk'}
        self.runtime.subgraph.return_value = {'nodes': [], 'edges': [], 'cited_ids': []}
        graph = Mock()
        graph.search_by_name.return_value = ['node1']
        graph.graph.nodes = {'node1': {'name': 'name', 'type': 'Herb'}}
        self.runtime.graph.return_value = graph
        for route in ['/config', '/source/original-id/context', '/graph/subgraph?entity=name', '/graph/search?q=name']:
            with self.subTest(route=route):
                self.assertEqual(self.client.get(route).status_code, 200)
        self.assertEqual(self.gate.check_read.call_count, 4)
        self.gate.acquire.assert_not_called()
        self.runtime.run_query.assert_not_called()

    def test_denied_read_does_not_load_source_graph_or_book(self):
        self.gate.check_read.side_effect = GateError(429, 'Request rate limit reached', 60)
        for route in ['/config', '/source/id/context', '/graph/subgraph?entity=name', '/graph/search?q=name', '/books/unknown']:
            response = self.client.get(route)
            self.assertEqual(response.status_code, 429)
            self.assertEqual(response.headers['retry-after'], '60')
        self.runtime.source_context.assert_not_called()
        self.runtime.subgraph.assert_not_called()
        self.runtime.graph.assert_not_called()

    def test_arena_queries_writes_models_and_statistics_disabled(self):
        for path in ['/arena/query', '/arena/vote', '/arena/stats', '/arena/models']:
            for method in ('GET', 'POST'):
                with self.subTest(path=path, method=method):
                    self.assertEqual(self.client.request(method, path, json={}).status_code, 403)
        self.gate.acquire.assert_not_called()
        self.runtime.run_query.assert_not_called()

    def test_forwarding_header_cannot_choose_quota_identity(self):
        self.client.post('/query', json={'question': 'history'},
                         headers={'Authorization': 'Bearer synthetic-access-code', 'X-Forwarded-For': '1.2.3.4'})
        self.assertEqual(self.gate.acquire.call_args.args[1], 'testclient')

    def test_graph_bounds_and_missing_source(self):
        self.runtime.source_context.return_value = None
        self.assertEqual(self.client.get('/source/missing/context').status_code, 404)
        for path in ['/graph/subgraph?entity=name&hops=3', '/graph/search?q=name&limit=21', '/graph/search?q=']:
            self.assertEqual(self.client.get(path).status_code, 400)
        self.runtime.graph.assert_not_called()
        self.runtime.subgraph.assert_not_called()

    def test_unexpected_public_error_is_sanitized(self):
        self.runtime.source_context.side_effect = RuntimeError('private-source-path fake-secret')
        response = self.client.get('/source/id/context')
        self.assertEqual(response.status_code, 503)
        self.assertNotIn('private-source-path', response.text)
        self.assertNotIn('fake-secret', response.text)


if __name__ == '__main__':
    unittest.main()
