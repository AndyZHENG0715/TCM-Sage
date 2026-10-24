"""Pinned free SiliconFlow CN retrieval, unaltered text and cosine-compatible BGE.

No dotenv loading, model discovery, paid fallback, or secret/error-body logging.
API contracts: https://api-docs.siliconflow.cn/docs/api/embeddings-post and
https://api-docs.siliconflow.cn/docs/api/rerank-post (checked 2026-10-24).
The local byte estimate deliberately over-reserves tokens; valid actual usage
replaces that reservation. Limits apply per shared client/process.
"""

from collections import deque
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import math
from numbers import Real
import os
import threading
import time
import urllib.error
import urllib.request

from langchain_core.embeddings import Embeddings

EMBEDDING_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
DIMENSIONS = 1024
PREPROCESSING = "unaltered-content-v1"
MAX_INPUT_BYTES = 8000  # Below official 8192 tokens, reserving special tokens.
MAX_BATCH_ESTIMATE = 16000
REQUESTS_PER_MINUTE = 60
TOKENS_PER_MINUTE = 400000


class RetrievalError(RuntimeError):
    """Safe to display; contains no input, secret, or upstream response body."""


def text_tokens(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Retrieval text must be a nonempty string")
    size = len(text.encode("utf-8"))
    if size > MAX_INPUT_BYTES:
        raise ValueError("Retrieval input exceeds conservative 8000-byte safety limit")
    return size + 32


def validate_vector(vector):
    if not isinstance(vector, list) or len(vector) != DIMENSIONS:
        raise RetrievalError("Embedding dimensions mismatch")
    try:
        if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) for v in vector):
            raise RetrievalError("Embedding contains nonfinite values")
        norm = math.hypot(*vector)
        if not norm or not math.isfinite(norm):
            raise RetrievalError("Embedding norm is invalid")
        return [float(v) for v in vector]
    except (OverflowError, TypeError, ValueError):
        raise RetrievalError("Embedding values are invalid") from None


def batches(texts):
    batch, estimated = [], 0
    for text in texts:
        cost = text_tokens(text)
        if batch and (len(batch) >= 16 or estimated + cost > MAX_BATCH_ESTIMATE):
            yield batch
            batch, estimated = [], 0
        batch.append(text)
        estimated += cost
    if batch:
        yield batch


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _transport(endpoint, payload, key):
    request = urllib.request.Request(
        "https://api.siliconflow.cn/v1/" + endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.build_opener(_NoRedirect()).open(request, timeout=45) as response:
            body = response.read(2_000_001)
            if len(body) > 2_000_000:
                raise RetrievalError("Provider response exceeds safety limit")
            return response.status, json.loads(body), dict(response.headers)
    except urllib.error.HTTPError as error:
        status, headers = error.code, dict(error.headers)
        error.close()  # Do not read or propagate upstream body.
        return status, None, headers


class FreeRetrievalClient:
    """Fixed models, serial requests, <=60 RPM and 400k tokens/minute.

    transport(endpoint, payload, key)->(status, body, headers) is trusted test
    injection only. A 429 gets at most two retries, each <=60s; no other retry.
    """

    def __init__(self, api_key, transport=None, clock=None, sleep=None):
        if not isinstance(api_key, str) or not api_key or any(c.isspace() for c in api_key):
            raise ValueError("A valid SiliconFlow key is required")
        self._key = api_key
        self._transport = transport or _transport
        self._clock, self._sleep = clock or time.monotonic, sleep or time.sleep
        self._events = deque()
        self._lock = threading.Lock()
        self.requests = 0

    def _reserve(self, tokens):
        while True:
            now = self._clock()
            while self._events and now - self._events[0][0] >= 60:
                self._events.popleft()
            if len(self._events) < REQUESTS_PER_MINUTE and sum(e[1] for e in self._events) + tokens <= TOKENS_PER_MINUTE:
                event = [now, tokens]
                self._events.append(event)
                return event
            self._sleep(max(.01, 60 - (now - self._events[0][0])))

    @staticmethod
    def _retry_delay(headers, attempt):
        raw = next((v for k, v in headers.items() if k.lower() == "retry-after"), None)
        if raw is None:
            return 2 ** (attempt + 1)
        try:
            delay = float(raw)
        except (ValueError, TypeError):
            try:
                date = parsedate_to_datetime(raw)
                delay = (date - datetime.now(timezone.utc)).total_seconds()
            except (ValueError, TypeError, OverflowError):
                raise RetrievalError("Provider rate limited; invalid retry delay") from None
        if not math.isfinite(delay) or delay > 60:
            raise RetrievalError("Provider rate limited; retry delay exceeds safety limit")
        return max(1., delay)

    def _post(self, endpoint, payload, estimate):
        if estimate > TOKENS_PER_MINUTE:
            raise ValueError("Request exceeds local token reservation budget")
        with self._lock:
            for attempt in range(3):
                event = self._reserve(estimate)
                self.requests += 1
                try:
                    status, body, headers = self._transport(endpoint, payload, self._key)
                except Exception:
                    raise RetrievalError("Retrieval provider transport failed") from None
                if status == 429:
                    if attempt == 2:
                        raise RetrievalError("Retrieval rate limit; retry allowance exhausted")
                    self._sleep(self._retry_delay(headers, attempt))
                    continue
                if status in (401, 403):
                    raise RetrievalError("Retrieval provider authentication or access denied")
                if status != 200:
                    raise RetrievalError("Retrieval provider HTTP failure")
                if not isinstance(body, dict):
                    raise RetrievalError("Malformed retrieval response")
                usage = body.get("usage", {})
                meta = body.get("meta", {})
                actual = usage.get("total_tokens") if isinstance(usage, dict) else None
                if actual is None and isinstance(meta, dict):
                    tokens = meta.get("tokens", {})
                    actual = tokens.get("input_tokens") if isinstance(tokens, dict) else None
                if type(actual) is int and actual > 0:
                    # Validated billing usage replaces conservative pre-call
                    # reservation, including downward reconciliation.
                    event[1] = actual
                return body
        raise RetrievalError("Retrieval request failed")

    def embed(self, texts):
        if not isinstance(texts, (list, tuple)) or not 1 <= len(texts) <= 16:
            raise ValueError("Embedding batch must have 1 to 16 texts")
        estimate = sum(text_tokens(t) for t in texts)
        if estimate > MAX_BATCH_ESTIMATE:
            raise ValueError("Embedding batch exceeds token reservation limit")
        body = self._post("embeddings", {
            "model": EMBEDDING_MODEL, "input": list(texts), "encoding_format": "float"
        }, estimate)
        if body.get("model") != EMBEDDING_MODEL:
            raise RetrievalError("Unexpected embedding model")
        data = body.get("data")
        if not isinstance(data, list) or len(data) != len(texts):
            raise RetrievalError("Embedding response count mismatch")
        ordered = [None] * len(texts)
        for item in data:
            if not isinstance(item, dict):
                raise RetrievalError("Malformed embedding result")
            index = item.get("index")
            if type(index) is not int or not 0 <= index < len(texts) or ordered[index] is not None:
                raise RetrievalError("Invalid embedding index")
            ordered[index] = validate_vector(item.get("embedding"))
        return ordered

    def rerank(self, query, documents, top_n=5):
        if not isinstance(documents, (list, tuple)) or len(documents) > 20:
            raise ValueError("Reranking accepts at most 20 documents")
        if not documents:
            return []
        if type(top_n) is not int or not 1 <= top_n <= len(documents):
            raise ValueError("Invalid reranking top_n")
        qcost = text_tokens(query)
        costs = [text_tokens(t) for t in documents]
        if any(qcost + cost > 8000 for cost in costs):
            raise ValueError("Query/document pair exceeds conservative token limit")
        estimate = sum(costs) + qcost * len(documents)
        body = self._post("rerank", {
            "model": RERANK_MODEL, "query": query, "documents": list(documents),
            "top_n": top_n, "return_documents": False, "max_chunks_per_doc": 1,
        }, estimate)
        rows = body.get("results")
        if not isinstance(rows, list) or len(rows) != top_n:
            raise RetrievalError("Reranking response count mismatch")
        result, seen = [], set()
        for row in rows:
            if not isinstance(row, dict):
                raise RetrievalError("Malformed reranking result")
            index, score = row.get("index"), row.get("relevance_score")
            if type(index) is not int or not 0 <= index < len(documents) or index in seen:
                raise RetrievalError("Invalid reranking index")
            try:
                valid = type(score) in (int, float) and math.isfinite(score)
            except OverflowError:
                valid = False
            if not valid:
                raise RetrievalError("Invalid reranking score")
            seen.add(index)
            result.append({"index": index, "relevance_score": float(score), "text": documents[index]})
        return sorted(result, key=lambda r: r["relevance_score"], reverse=True)


class SiliconFlowEmbeddings(Embeddings):
    def __init__(self, client):
        self.client = client

    def embed_documents(self, texts):
        return [vector for batch in batches(texts) for vector in self.client.embed(batch)]

    def embed_query(self, text):
        return self.client.embed([text])[0]


_default = None
_default_lock = threading.Lock()


def get_embeddings(api_key=None):
    """Return shared LangChain embeddings; caller loads environment config."""
    global _default
    with _default_lock:
        key = api_key or os.environ.get("SILICONFLOW_API_KEY")
        if not key:
            raise ValueError("SILICONFLOW_API_KEY is not configured")
        if _default is None or _default.client._key != key:
            _default = SiliconFlowEmbeddings(FreeRetrievalClient(key))
        return _default


def rerank(query, docs, top_n=5):
    return get_embeddings().client.rerank(query, docs, top_n)
