"""Bounded standalone SiliconFlow CN validation; never loads application config.

Contracts checked 2026-10-24:
https://api-docs.siliconflow.cn/docs/api/embeddings-post
https://api-docs.siliconflow.cn/docs/api/rerank-post
Free model names are pinned, not a guarantee that future provider pricing stays free.
Transport injection is for trusted tests: callable(url, headers, payload, timeout)
returns (HTTP status, decoded JSON). No retries, redirects, or model fallback.
"""

import json
import math
import urllib.error
import urllib.request


class ProviderError(RuntimeError):
    """Sanitized failure: no request text, credentials, or response body."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _http_transport(url, headers, payload, timeout):
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
    )
    opener = urllib.request.build_opener(_NoRedirect())
    try:
        with opener.open(request, timeout=timeout) as response:
            raw = response.read(2_000_001)
            if len(raw) > 2_000_000:
                raise ProviderError("Response exceeds validation limit")
            return response.status, json.loads(raw)
    except urllib.error.HTTPError as error:
        # Never read the body, which may echo a key or user input.
        status = error.code
        error.close()
        return status, None


class SiliconFlowFreeProvider:
    """Fixed bge-m3 embedding and bge-reranker-v2-m3 validation adapter.

    Limits are deliberately local and conservative, not provider token limits.
    One instance permits 64 attempted requests (including failed requests).
    Reuse the instance throughout a bounded run; this is not a global spend cap.
    """

    __slots__ = ("_api_key", "_transport", "_requests")
    MAX_REQUESTS = 64
    MAX_TEXTS = 16
    MAX_RERANK_DOCUMENTS = 20
    MAX_TEXT_CHARS = 2048
    MAX_TOTAL_CHARS = 16384
    TIMEOUT_SECONDS = 30
    EMBEDDING_DIMENSIONS = 1024

    def __init__(self, api_key, transport=None):
        if not isinstance(api_key, str) or not api_key.strip() or any(
            char.isspace() for char in api_key
        ):
            raise ValueError("A nonempty API key without whitespace is required")
        if transport is not None and not callable(transport):
            raise ValueError("transport must be callable")
        self._api_key = api_key
        self._transport = transport or _http_transport
        self._requests = 0

    @staticmethod
    def _texts(texts, max_count=16):
        if not isinstance(texts, (list, tuple)) or not 1 <= len(texts) <= max_count:
            raise ValueError(f"Provide 1 to {max_count} text strings")
        for text in texts:
            if not isinstance(text, str) or not text.strip() or len(text) > 2048:
                raise ValueError("Each text must contain 1 to 2048 characters")
        if sum(map(len, texts)) > 16384:
            raise ValueError("Total text exceeds 16384 characters")
        return list(texts)

    def _post(self, endpoint, payload):
        if self._requests >= self.MAX_REQUESTS:
            raise ProviderError("Validation request limit reached")
        self._requests += 1
        try:
            status, body = self._transport(
                "https://api.siliconflow.cn/v1/" + endpoint,
                {"Authorization": "Bearer " + self._api_key,
                 "Content-Type": "application/json"},
                payload,
                self.TIMEOUT_SECONDS,
            )
        except Exception:
            raise ProviderError("Provider transport failed") from None
        if status in (401, 403):
            raise ProviderError("Provider authentication or access denied")
        if status == 429:
            raise ProviderError("Provider rate limited; no retry attempted")
        if status != 200:
            raise ProviderError("Provider HTTP request failed")
        if not isinstance(body, dict):
            raise ProviderError("Malformed provider response")
        return body

    @staticmethod
    def _number(value):
        if type(value) not in (int, float):
            return False
        try:
            return math.isfinite(value)
        except OverflowError:
            return False

    def embed(self, texts):
        """Return 1024-dimensional dense vectors in input order."""
        texts = self._texts(texts)
        body = self._post("embeddings", {
            "model": "BAAI/bge-m3", "input": texts, "encoding_format": "float"
        })
        if body.get("model") != "BAAI/bge-m3":
            raise ProviderError("Unexpected embedding model")
        data = body.get("data")
        if not isinstance(data, list) or len(data) != len(texts):
            raise ProviderError("Embedding response count mismatch")
        ordered = [None] * len(texts)
        for item in data:
            if not isinstance(item, dict):
                raise ProviderError("Malformed embedding item")
            index, vector = item.get("index"), item.get("embedding")
            if type(index) is not int or not 0 <= index < len(texts) or ordered[index] is not None:
                raise ProviderError("Invalid or duplicate embedding index")
            if not isinstance(vector, list) or len(vector) != 1024 or not all(
                self._number(value) for value in vector
            ):
                raise ProviderError("Invalid embedding dimensions or values")
            if not any(value != 0 for value in vector):
                raise ProviderError("Zero-norm embedding is unusable")
            ordered[index] = [float(value) for value in vector]
        return ordered

    def rerank(self, query, documents, top_n):
        """Return [{index, relevance_score}] sorted by descending score."""
        self._texts([query])
        documents = self._texts(documents, self.MAX_RERANK_DOCUMENTS)
        if len(query) + sum(map(len, documents)) > self.MAX_TOTAL_CHARS:
            raise ValueError("Total query and document text exceeds limit")
        if type(top_n) is not int or not 1 <= top_n <= len(documents):
            raise ValueError("top_n must be between 1 and the document count")
        body = self._post("rerank", {
            "model": "BAAI/bge-reranker-v2-m3", "query": query,
            "documents": documents, "top_n": top_n, "return_documents": False,
            "max_chunks_per_doc": 1,
        })
        results = body.get("results")
        if not isinstance(results, list) or len(results) != top_n:
            raise ProviderError("Rerank response count mismatch")
        seen, normalized = set(), []
        for item in results:
            if not isinstance(item, dict):
                raise ProviderError("Malformed rerank item")
            index, score = item.get("index"), item.get("relevance_score")
            if type(index) is not int or not 0 <= index < len(documents) or index in seen:
                raise ProviderError("Invalid or duplicate rerank index")
            if not self._number(score):
                raise ProviderError("Invalid rerank score")
            seen.add(index)
            normalized.append({"index": index, "relevance_score": float(score)})
        return sorted(normalized, key=lambda item: item["relevance_score"], reverse=True)
