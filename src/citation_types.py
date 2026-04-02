"""
TCM-Sage Type Definitions

TypedDict schemas for structured data used across the RAG pipeline.
"""

from typing import Literal, Optional, TypedDict, Union


class TextCitation(TypedDict):
    """Citation metadata for vector-retrieved text passages."""

    number: int
    type: Literal["text"]
    source: str
    content: str
    chunk_id: Optional[str]
    score: float
    relevance_percent: float


class GraphCitation(TypedDict):
    """Citation metadata for knowledge graph facts."""

    number: int
    type: Literal["graph"]
    fact: str
    depth: int
    source_ref: Optional[dict]


Citation = Union[TextCitation, GraphCitation]
