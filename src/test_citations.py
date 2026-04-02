"""
TCM-Sage Citation System Tests

Unit tests for format_docs_with_citations() and citation metadata structure.
"""

import sys
from pathlib import Path

from langchain_core.documents import Document

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from main import format_docs_with_citations, verify_citation_bounds


def test_format_docs_with_citations_returns_tuple():
    docs = [
        Document(
            page_content="Test content about yin and yang.",
            metadata={"source": "Chapter 1", "source_type": "vector", "score": 0.85},
        )
    ]

    result = format_docs_with_citations(docs)

    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], str)
    assert isinstance(result[1], list)


def test_citation_numbers_are_sequential():
    docs = [
        Document(
            page_content="First passage",
            metadata={"source": "Chapter 1", "source_type": "vector", "score": 0.9},
        ),
        Document(
            page_content="Second passage",
            metadata={"source": "Chapter 2", "source_type": "vector", "score": 1.8},
        ),
        Document(
            page_content="川芎 --TREATS--> 头痛",
            metadata={"source_type": "graph", "depth": 1},
        ),
    ]

    context, citations = format_docs_with_citations(docs)

    assert [citation["number"] for citation in citations] == [1, 2, 3]
    assert "[1]" in context
    assert "[2]" in context
    assert "[3]" in context


def test_text_citation_structure():
    docs = [
        Document(
            page_content="阴阳者，天地之道也，万物之纲纪，变化之父母。",
            metadata={
                "source": "素问·阴阳应象大论",
                "source_type": "vector",
                "score": 0.451,
                "id": "chunk_42",
            },
        )
    ]

    _, citations = format_docs_with_citations(docs)
    citation = citations[0]

    assert len(citations) == 1
    assert citation["number"] == 1
    assert citation["type"] == "text"
    assert citation["source"] == "素问·阴阳应象大论"
    assert citation["chunk_id"] == "chunk_42"
    assert citation["score"] == 0.451
    assert citation["relevance_percent"] == 95.0
    assert "content" in citation


def test_graph_citation_structure():
    docs = [
        Document(
            page_content="川芎 --TREATS--> 头痛 (Symptom)",
            metadata={"source_type": "graph", "depth": 1},
        )
    ]

    _, citations = format_docs_with_citations(docs)
    citation = citations[0]

    assert len(citations) == 1
    assert citation["number"] == 1
    assert citation["type"] == "graph"
    assert citation["fact"] == "川芎 --TREATS--> 头痛 (Symptom)"
    assert citation["depth"] == 1
    assert "source_ref" in citation


def test_empty_docs_returns_empty():
    context, citations = format_docs_with_citations([])

    assert context == ""
    assert citations == []


def test_mixed_source_types():
    docs = [
        Document(
            page_content="Vector content",
            metadata={"source": "Chapter 1", "source_type": "vector", "score": 0.9},
        ),
        Document(
            page_content="KG Fact: Herb treats symptom",
            metadata={"source_type": "graph", "depth": 2},
        ),
        Document(
            page_content="Another vector content",
            metadata={"source": "Chapter 2", "source_type": "vector", "score": 1.4},
        ),
    ]

    _, citations = format_docs_with_citations(docs)

    assert [citation["type"] for citation in citations] == ["text", "text", "graph"]


def test_relevance_percentages_are_ranked_per_response():
    docs = [
        Document(
            page_content="Best match",
            metadata={"source": "Chapter 1", "source_type": "vector", "score": 0.5},
        ),
        Document(
            page_content="Middle match",
            metadata={"source": "Chapter 2", "source_type": "vector", "score": 1.0},
        ),
        Document(
            page_content="Worst match",
            metadata={"source": "Chapter 3", "source_type": "vector", "score": 2.0},
        ),
    ]

    _, citations = format_docs_with_citations(docs)

    percentages = [citation["relevance_percent"] for citation in citations]
    assert percentages == [95.0, 83.3, 60.0]


def test_verify_citation_bounds_valid():
    answer = "According to the Neijing [1], yin and yang are fundamental [2]."
    result = verify_citation_bounds(answer, max_citation=3)

    assert result["is_valid"] is True
    assert result["out_of_range"] == []
    assert result["found_citations"] == [1, 2]


def test_verify_citation_bounds_out_of_range():
    answer = "Based on source [1] and [5], we can conclude [3]."
    result = verify_citation_bounds(answer, max_citation=3)

    assert result["is_valid"] is False
    assert 5 in result["out_of_range"]
    assert result["found_citations"] == [1, 3, 5]


def test_graph_citation_has_source_ref_field():
    docs = [
        Document(
            page_content="营气 --FLOWS_THROUGH--> 脉 (BodyPart)",
            metadata={
                "source_type": "graph",
                "depth": 1,
                "source_ref": {
                    "book": "黄帝内经灵枢集注",
                    "chapter": "<篇名>营卫生会篇第十八",
                    "char_start": 102514,
                    "char_end": 103007,
                },
            },
        )
    ]

    _, citations = format_docs_with_citations(docs)
    citation = citations[0]

    assert len(citations) == 1
    assert citation["source_ref"] is not None
    assert citation["source_ref"]["book"] == "黄帝内经灵枢集注"


if __name__ == "__main__":
    test_format_docs_with_citations_returns_tuple()
    test_citation_numbers_are_sequential()
    test_text_citation_structure()
    test_graph_citation_structure()
    test_empty_docs_returns_empty()
    test_mixed_source_types()
    test_relevance_percentages_are_ranked_per_response()
    test_verify_citation_bounds_valid()
    test_verify_citation_bounds_out_of_range()
    test_graph_citation_has_source_ref_field()
    print("All citation tests passed.")
