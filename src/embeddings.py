from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

INGESTION_PREFIX = "为这个句子生成表示以用于检索中文段落："
QUERY_PREFIX = "为这个句子生成表示以用于检索中文段落："
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
