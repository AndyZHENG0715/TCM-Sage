from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

INGESTION_PREFIX = "为这段中医古籍文本生成语义表示用于检索："
QUERY_PREFIX = "为这个中医临床问题生成语义表示以检索相关古籍段落："
_BATCH_LIMIT = 10


class DashScopeEmbeddings(Embeddings):
    def __init__(self, model: str = "text-embedding-v4", dimension: int = 1024):
        load_dotenv(override=True)
        self.model = model
        self.dimension = dimension
        self.api_url = os.getenv(
            "DASHSCOPE_EMBEDDING_API_URL",
            "https://dashscope-intl.aliyuncs.com/api/v1",
        )
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY not set in environment")
        self.api_key: str = api_key

    def _embed_prefixed_texts(self, prefixed_texts: list[str]) -> list[list[float]]:
        import dashscope

        dashscope.base_http_api_url = self.api_url

        response = dashscope.TextEmbedding.call(
            model=self.model,
            input=prefixed_texts,
            dimension=self.dimension,
            api_key=self.api_key,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"DashScope embedding error: {response.code} - {response.message}"
            )

        return [item["embedding"] for item in response.output["embeddings"]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), _BATCH_LIMIT):
            batch = texts[i : i + _BATCH_LIMIT]
            prefixed_batch = [f"{INGESTION_PREFIX}{text}" for text in batch]
            all_embeddings.extend(self._embed_prefixed_texts(prefixed_batch))
        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        prefixed_query = f"{QUERY_PREFIX}{text}"
        return self._embed_prefixed_texts([prefixed_query])[0]


def get_embedding_model() -> DashScopeEmbeddings:
    return DashScopeEmbeddings(model="text-embedding-v4", dimension=1024)


def rerank_documents(
    query: str,
    documents: list[str],
    top_n: int = 5,
    model: str = "qwen3-rerank",
) -> list[dict]:
    """Rerank documents using DashScope reranker.

    Args:
        query: The search query.
        documents: List of document texts to rerank.
        top_n: Number of top documents to return.
        model: Reranker model name.

    Returns:
        List of dicts with 'index', 'relevance_score', 'text' sorted by relevance.
    """
    import dashscope

    load_dotenv(override=True)
    api_key = os.getenv("DASHSCOPE_API_KEY")
    api_url = os.getenv(
        "DASHSCOPE_EMBEDDING_API_URL",
        "https://dashscope-intl.aliyuncs.com/api/v1",
    )
    dashscope.base_http_api_url = api_url

    if not documents:
        return []

    resp = dashscope.TextReRank.call(
        model=model,
        query=query,
        documents=documents,
        top_n=min(top_n, len(documents)),
        return_documents=True,
        api_key=api_key,
    )

    if resp.status_code != 200:
        # Fallback: return original order if reranker fails
        return [{"index": i, "relevance_score": 0.0, "text": doc} for i, doc in enumerate(documents[:top_n])]

    results = []
    for r in resp.output.results:
        results.append({
            "index": r.index,
            "relevance_score": r.relevance_score,
            "text": r.document.text if r.document else documents[r.index],
        })
    return results
