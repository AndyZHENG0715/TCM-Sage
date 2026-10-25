"""Offline public-runtime security checks. All network/provider work is mocked."""

import io
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import release_runtime as runtime


def event(value):
    return b"data: " + json.dumps(value).encode() + b"\n\n"


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        runtime.retrieval_client.cache_clear()

    def tearDown(self):
        runtime.retrieval_client.cache_clear()
        self.environment.stop()

    def stream(self, wire):
        return patch.object(runtime, "_chat_request", return_value=io.BytesIO(wire))

    def row(self, cid="original-id", book="伤寒论", content="原始文本。", source="篇章"):
        return {"id": cid, "content": content,
                "metadata": {"book": book, "source": source, "chunk_index": 7}}

    def retrieval_row(self, source):
        return {"chunk_id": source['id'], "content": source['content'],
                "metadata": source['metadata'], "distance": 0.2}

    def fingerprint_fixture(self):
        records = [self.row(content='第一行\n第二行。')]
        lf = (json.dumps(records, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
        crlf = lf.replace(b'\n', b'\r\n')
        canonical = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
        return records, lf, crlf, {'corpus_sha256': hashlib.sha256(crlf).hexdigest(),
                                  'corpus_canonical_sha256': hashlib.sha256(canonical).hexdigest(),
                                  'corpus_canonical_format': 'json-sort-keys-utf8-v1'}

    def test_corpus_fingerprint_accepts_json_lineending_only_difference(self):
        records, lf, crlf, fingerprint = self.fingerprint_fixture()
        self.assertNotEqual(hashlib.sha256(lf).hexdigest(), fingerprint['corpus_sha256'])
        path = Mock()
        path.read_bytes.return_value = lf
        with patch.object(runtime, 'CHUNKS_PATH', path):
            runtime.verify_corpus_fingerprint(fingerprint)
        self.assertEqual(json.loads(lf), json.loads(crlf))
        self.assertEqual(records[0]['content'], '第一行\n第二行。')

    def test_corpus_fingerprint_raw_match_backward_compatible(self):
        _, _, crlf, fingerprint = self.fingerprint_fixture()
        path = Mock()
        path.read_bytes.return_value = crlf
        with patch.object(runtime, 'CHUNKS_PATH', path):
            runtime.verify_corpus_fingerprint({'corpus_sha256': fingerprint['corpus_sha256']})

    def test_corpus_fingerprint_changed_text_or_metadata_fails(self):
        for mutation in ('text', 'metadata', 'embedded-lineending', 'id'):
            with self.subTest(mutation=mutation):
                records, _, _, fingerprint = self.fingerprint_fixture()
                if mutation == 'text':
                    records[0]['content'] += 'changed'
                elif mutation == 'metadata':
                    records[0]['metadata']['source'] = 'different chapter'
                elif mutation == 'embedded-lineending':
                    records[0]['content'] = records[0]['content'].replace('\n', '\r\n')
                else:
                    records[0]['id'] = 'different-id'
                path = Mock()
                path.read_bytes.return_value = json.dumps(records, ensure_ascii=False).encode('utf-8')
                with patch.object(runtime, 'CHUNKS_PATH', path):
                    with self.assertRaises(runtime.RuntimeFailure) as error:
                        runtime.verify_corpus_fingerprint(fingerprint)
                self.assertEqual(str(error.exception), 'The research index does not match the source corpus.')

    def test_corpus_fingerprint_missing_or_wrong_canonical_fails(self):
        _, lf, _, fingerprint = self.fingerprint_fixture()
        for invalid in (
            {'corpus_sha256': fingerprint['corpus_sha256']},
            {**fingerprint, 'corpus_canonical_sha256': ''},
            {**fingerprint, 'corpus_canonical_sha256': '0' * 64},
            {**fingerprint, 'corpus_canonical_format': 'unknown-v2'},
        ):
            with self.subTest(fingerprint=invalid):
                path = Mock()
                path.read_bytes.return_value = lf
                with patch.object(runtime, 'CHUNKS_PATH', path):
                    with self.assertRaises(runtime.RuntimeFailure):
                        runtime.verify_corpus_fingerprint(invalid)

    def test_chat_is_fixed_free_openrouter_and_uses_only_its_key(self):
        os.environ.update(OPENROUTER_API_KEY="fake-chat-secret", SILICONFLOW_API_KEY="fake-retrieval-secret",
                          LLM_PROVIDER="alibaba", LLM_MODEL="paid-model-ignored")
        opener = Mock()
        opener.open.return_value = io.BytesIO(b"{}")
        with patch.object(runtime.urllib.request, "build_opener", return_value=opener):
            runtime._chat_request([{"role": "user", "content": "historical source"}], 9999, True).close()
        request = opener.open.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(request.full_url, "https://openrouter.ai/api/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer fake-chat-secret")
        self.assertNotIn("fake-retrieval-secret", str(request.headers))
        self.assertEqual(payload['model'], "nvidia/nemotron-3-ultra-550b-a55b:free")
        self.assertEqual(payload['max_tokens'], 1536)
        self.assertEqual(payload['provider']['max_price'], {'prompt': 0, 'completion': 0})
        self.assertNotIn('models', payload)
        self.assertFalse(payload['reasoning']['enabled'])

    def test_retrieval_uses_only_siliconflow_key(self):
        os.environ.update(SILICONFLOW_API_KEY="fake-retrieval-secret", OPENROUTER_API_KEY="fake-chat-secret")
        factory = Mock(return_value=object())
        with patch.dict(sys.modules, {'siliconflow_retrieval': SimpleNamespace(FreeRetrievalClient=factory)}):
            runtime.retrieval_client()
        factory.assert_called_once_with("fake-retrieval-secret")

    def test_missing_provider_key_does_not_fall_back(self):
        os.environ['SILICONFLOW_API_KEY'] = 'fake-retrieval-secret'
        with patch.object(runtime.urllib.request, "build_opener") as opener:
            with self.assertRaises(runtime.RuntimeFailure):
                runtime._chat_request([], 32)
        opener.assert_not_called()
        os.environ.clear()
        os.environ['OPENROUTER_API_KEY'] = 'fake-chat-secret'
        with self.assertRaises(runtime.RuntimeFailure):
            runtime.retrieval_client()

    def test_chat_transport_failure_redacts_exception_and_has_no_retry(self):
        os.environ['OPENROUTER_API_KEY'] = 'fake-chat-secret'
        opener = Mock()
        opener.open.side_effect = OSError('sensitive-response fake-chat-secret')
        with patch.object(runtime.urllib.request, "build_opener", return_value=opener):
            with self.assertRaises(runtime.RuntimeFailure) as error:
                runtime._chat_request([], 32)
        self.assertNotIn('sensitive-response', str(error.exception))
        self.assertNotIn('fake-chat-secret', str(error.exception))
        self.assertEqual(opener.open.call_count, 1)

    def test_redirect_handler_denies_redirect(self):
        self.assertIsNone(runtime._NoRedirect().redirect_request(None, None, 302, None, None, None))

    def test_chat_input_limit_precedes_transport(self):
        os.environ['OPENROUTER_API_KEY'] = 'fake-chat-secret'
        with patch.object(runtime.urllib.request, "build_opener") as opener:
            with self.assertRaises(runtime.RuntimeFailure):
                runtime._chat_request([{'content': 'x' * 12001}], 32)
        opener.assert_not_called()

    def test_stream_requires_done_and_returns_content(self):
        wire = event({'choices': [{'delta': {'content': 'answer [1]'}}]}) + b'data: [DONE]\n\n'
        with self.stream(wire):
            self.assertEqual(list(runtime.chat_stream([])), ['answer [1]'])
        with self.stream(event({'choices': [{'delta': {'content': 'partial'}}]})):
            with self.assertRaises(runtime.RuntimeFailure):
                list(runtime.chat_stream([]))

    def test_empty_done_stream_is_not_success(self):
        for wire in (b'data: [DONE]\n\n', event({'choices': [{'delta': {}, 'finish_reason': 'stop'}]}) + b'data: [DONE]\n\n'):
            with self.subTest(wire=wire), self.stream(wire):
                with self.assertRaises(runtime.RuntimeFailure):
                    list(runtime.chat_stream([]))

    def test_truncated_finish_reason_is_not_success(self):
        for reason in ('length', 'content_filter'):
            with self.subTest(reason=reason):
                wire = event({'choices': [{'delta': {'content': 'partial'}, 'finish_reason': reason}]}) + b'data: [DONE]\n\n'
                with self.stream(wire):
                    with self.assertRaises(runtime.RuntimeFailure):
                        list(runtime.chat_stream([]))

    def test_stream_provider_error_redacted(self):
        wire = event({'error': {'message': 'secret-echo should-not-escape'}})
        with self.stream(wire):
            with self.assertRaises(runtime.RuntimeFailure) as error:
                list(runtime.chat_stream([]))
        self.assertNotIn('secret-echo', str(error.exception))

    def test_stream_malformed_oversize_and_output_limits(self):
        wires = [b'data: INVALID-SECRET\n', b'data: ' + b'x' * 65537,
                 event({'choices': [{'delta': {'content': 'x' * 10001}}]})]
        for wire in wires:
            with self.subTest(length=len(wire)), self.stream(wire):
                with self.assertRaises(runtime.RuntimeFailure) as error:
                    list(runtime.chat_stream([]))
                self.assertNotIn('INVALID-SECRET', str(error.exception))

    def test_out_of_scope_does_not_retrieve_or_generate(self):
        with patch.object(runtime, 'chat', return_value='out_of_scope') as classifier, \
             patch.object(runtime, 'retrieve') as retrieve, patch.object(runtime, 'chat_stream') as generate:
            result = list(runtime.run_query('modern trial DOI?'))
        retrieve.assert_not_called()
        generate.assert_not_called()
        self.assertEqual(classifier.call_count, 1)
        self.assertEqual(result[-1]['citations'], [])
        self.assertEqual(result[-1]['verification']['status'], 'UNAVAILABLE')

    def test_unusable_classifier_fails_before_retrieval(self):
        for classification in ('', 'informational or prescriptive', 'probably informational', 'SECRET-ECHO'):
            with self.subTest(classification=classification), \
                 patch.object(runtime, 'chat', return_value=classification), \
                 patch.object(runtime, 'retrieve') as retrieve, \
                 patch.object(runtime, 'chat_stream') as stream:
                with self.assertRaises(runtime.RuntimeFailure) as error:
                    list(runtime.run_query('historical question'))
                self.assertNotIn('SECRET-ECHO', str(error.exception))
                retrieve.assert_not_called()
                stream.assert_not_called()

    def test_nonstream_truncated_empty_or_malformed_response_rejected(self):
        choices = [
            {'finish_reason': 'length', 'message': {'content': 'partial'}},
            {'finish_reason': 'content_filter', 'message': {'content': 'partial'}},
            {'finish_reason': 'stop', 'message': {'content': ''}},
            {'finish_reason': 'stop', 'message': {'content': '   '}},
            {'finish_reason': 'stop', 'message': {'content': None}},
            {'finish_reason': 'stop', 'message': {'content': {'invalid': 'shape'}}},
        ]
        for choice in choices:
            with self.subTest(choice=choice), patch.object(runtime, '_chat_request', return_value=io.BytesIO(json.dumps({'choices': [choice]}).encode())):
                with self.assertRaises(runtime.RuntimeFailure):
                    runtime.chat([])

    def test_verifier_failure_is_unavailable_and_history_not_forwarded(self):
        row = self.row()
        with patch.object(runtime, 'corpus', return_value={row['id']: row}), \
             patch.object(runtime, 'retrieve', return_value=[self.retrieval_row(row)]), \
             patch.object(runtime, 'chat', side_effect=['informational', runtime.RuntimeFailure('unavailable')]), \
             patch.object(runtime, 'chat_stream', return_value=iter(['historical answer [1]'])) as stream:
            result = list(runtime.run_query('historical question', [{'role': 'system', 'content': 'UNTRUSTED-HISTORY'}]))
        self.assertEqual(result[-1]['verification']['status'], 'UNAVAILABLE')
        self.assertNotIn('UNTRUSTED-HISTORY', str(stream.call_args))
        self.assertEqual(result[-1]['citations'][0]['content'], row['content'])

    def test_citation_out_of_range_overrides_supported_verifier(self):
        row = self.row()
        with patch.object(runtime, 'corpus', return_value={row['id']: row}), \
             patch.object(runtime, 'retrieve', return_value=[self.retrieval_row(row)]), \
             patch.object(runtime, 'chat', side_effect=['informational', 'SUPPORTED']), \
             patch.object(runtime, 'chat_stream', return_value=iter(['unsupported marker [2]'])):
            meta = list(runtime.run_query('historical question'))[-1]
        self.assertEqual(meta['verification']['status'], 'UNSUPPORTED')
        self.assertEqual(meta['citation_bounds']['out_of_range'], [2])

    def test_absent_clause_reference_overrides_supported_verifier(self):
        source = self.row(content='13．太阳病，头痛、发热、汗出、恶风。用前第一方。')
        source['metadata']['clause_number'] = 13
        for answer in ('The formula is listed in Clause 1 [1].', 'CLAUSE 1 supplies the ingredients [1].',
                       '方药见第1条 [1]。', '方药见第 1 条 [1]。'):
            with self.subTest(answer=answer), \
                 patch.object(runtime, 'corpus', return_value={source['id']: source}), \
                 patch.object(runtime, 'retrieve', return_value=[self.retrieval_row(source)]), \
                 patch.object(runtime, 'chat', side_effect=['informational', 'SUPPORTED']), \
                 patch.object(runtime, 'chat_stream', return_value=iter([answer])):
                meta = list(runtime.run_query('Explain historical clause13'))[-1]
                self.assertEqual(meta['verification']['status'], 'UNSUPPORTED')
                self.assertIn('1', meta['verification']['explanation'])
                self.assertTrue(meta['citation_bounds']['is_valid'])

    def test_supplied_clause13_and_formula_number_distinction(self):
        source = self.row(content='13．原始条文。用前第一方。')
        source['metadata']['clause_number'] = 13
        for answer in ('Clause 13 contains this wording [1].', '第13条有这段文字 [1]。',
                       'The source says first formula; no clause cross-reference is supplied [1].'):
            with self.subTest(answer=answer), \
                 patch.object(runtime, 'corpus', return_value={source['id']: source}), \
                 patch.object(runtime, 'retrieve', return_value=[self.retrieval_row(source)]), \
                 patch.object(runtime, 'chat', side_effect=['informational', 'SUPPORTED']), \
                 patch.object(runtime, 'chat_stream', return_value=iter([answer])):
                meta = list(runtime.run_query('Explain historical clause13'))[-1]
                self.assertEqual(meta['verification']['status'], 'SUPPORTED')

    def test_clause_guard_ignores_source_excluded_from_context(self):
        admitted = self.row('admitted-13', content='13．源文。')
        admitted['metadata']['clause_number'] = 13
        excluded = self.row('excluded-1', content='x' * (runtime.MAX_CONTEXT_CHARS + 1))
        excluded['metadata']['clause_number'] = 1
        sources = {row['id']: row for row in (admitted, excluded)}
        with patch.object(runtime, 'corpus', return_value=sources), \
             patch.object(runtime, 'retrieve', return_value=[self.retrieval_row(admitted), self.retrieval_row(excluded)]), \
             patch.object(runtime, 'chat', side_effect=['informational', 'SUPPORTED']), \
             patch.object(runtime, 'chat_stream', return_value=iter(['Clause 1 supplies it [1].'])):
            meta = list(runtime.run_query('Historical sources'))[-1]
        self.assertEqual(meta['verification']['status'], 'UNSUPPORTED')
        self.assertEqual([c['chunk_id'] for c in meta['citations']], ['admitted-13'])

    def test_mutated_retrieved_content_rejected_before_generation(self):
        row = self.row()
        retrieved = self.retrieval_row(row)
        retrieved['content'] = 'ALTERED'
        with patch.object(runtime, 'corpus', return_value={row['id']: row}), \
             patch.object(runtime, 'retrieve', return_value=[retrieved]), \
             patch.object(runtime, 'chat', return_value='informational'), \
             patch.object(runtime, 'chat_stream') as generate:
            with self.assertRaises(runtime.RuntimeFailure):
                list(runtime.run_query('historical question'))
        generate.assert_not_called()

    def test_source_context_preserves_cited_text_and_scope(self):
        row = self.row(content='原文\n第二行。')
        with patch.object(runtime, 'corpus', return_value={row['id']: row}):
            context = runtime.source_context(row['id'])
            self.assertIsNone(runtime.source_context('unknown'))
        self.assertEqual(context['book'], row['metadata']['book'])
        self.assertEqual(context['full_chapter_text'], row['content'])
        self.assertEqual(context['paragraph_text'], row['content'])
        self.assertEqual(context['context_scope'], 'cited_chunk')
        self.assertEqual(context['highlight_end'], len(row['content']))

    def test_shanghan_clause_filter_qualified_by_book(self):
        client, db = Mock(), Mock()
        client.embed.return_value = [[0.1]]
        client.rerank.return_value = []
        db.query.return_value = {'ids': [[]], 'documents': [[]], 'metadatas': [[]], 'distances': [[]]}
        with patch.object(runtime, 'retrieval_client', return_value=client), patch.object(runtime, 'collection', return_value=db):
            runtime.retrieve('伤寒论第八十二条')
        self.assertEqual(db.query.call_args.kwargs['where'], {'$and': [{'book': '伤寒论'}, {'clause_number': 82}]})

    def test_english_clause_reference_parsed_case_insensitively(self):
        self.assertEqual(runtime._clause_number('Explain Shanghan clause 13'), 13)
        self.assertEqual(runtime._clause_number('SHANGHAN CLAUSE 82'), 82)
        self.assertEqual(runtime._clause_number('伤寒论第八十二条'), 82)

    def test_multibook_retrieval_retains_both_books(self):
        rows = [self.row('sh', '伤寒论'), self.row('qj', '备急千金要方'), self.row('other', '其他书')]
        def query(**kwargs):
            where = kwargs.get('where', {})
            subset = [r for r in rows if not where or r['metadata']['book'] == where.get('book')]
            return {'ids': [[r['id'] for r in subset]], 'documents': [[r['content'] for r in subset]],
                    'metadatas': [[r['metadata'] for r in subset]], 'distances': [[0.1 + i for i, _ in enumerate(subset)]]}
        client, db = Mock(), Mock()
        client.embed.return_value = [[0.1]]
        db.query.side_effect = query
        client.rerank.side_effect = lambda q, docs, top_n: [{'index': i, 'relevance_score': 1.0} for i in reversed(range(len(docs)))]
        with patch.object(runtime, 'retrieval_client', return_value=client), patch.object(runtime, 'collection', return_value=db):
            result = runtime.retrieve('比较伤寒论与千金的文字')
        self.assertEqual({r['metadata']['book'] for r in result[:2]}, {'伤寒论', '备急千金要方'})
        self.assertEqual(client.embed.call_count, 1)


if __name__ == '__main__':
    unittest.main()
