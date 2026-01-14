"""
TCM-Sage Type Definitions

TypedDict schemas for structured data used across the RAG pipeline.
"""

from typing import TypedDict, Union, Literal, Optional


class TextCitation(TypedDict):
    """Citation metadata for vector-retrieved text passages."""

    number: int
    type: Literal["text"]
    source: str           # Chapter name (e.g., "素問·陰陽應象大論")
    content: str          # Snippet of the text passage
    chunk_id: Optional[str]
    score: float          # Similarity score (lower = better match for distance)


class GraphCitation(TypedDict):
    """Citation metadata for knowledge graph facts."""

    number: int
    type: Literal["graph"]
    fact: str             # Formatted fact (e.g., "川芎 --TREATS--> 頭痛")
    depth: int            # Traversal depth (1-hop, 2-hop, etc.)
    source_ref: Optional[dict]  # Optional provenance reference


# Union type for all citation types
Citation = Union[TextCitation, GraphCitation]
