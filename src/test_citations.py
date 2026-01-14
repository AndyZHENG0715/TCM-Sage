"""
TCM-Sage Citation System Tests

Unit tests for format_docs_with_citations() and citation metadata structure.
"""

import sys
from pathlib import Path

# Add src directory to path for imports
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from langchain_core.documents import Document

from main import format_docs_with_citations


def test_format_docs_with_citations_returns_tuple():
    """Test that format_docs_with_citations returns (str, list) tuple."""
    docs = [
        Document(
            page_content="Test content about yin and yang.",
            metadata={"source": "素問·陰陽應象大論", "source_type": "vector", "score": 0.85},
        )
    ]

    result = format_docs_with_citations(docs)

    assert isinstance(result, tuple), "Should return a tuple"
    assert len(result) == 2, "Tuple should have 2 elements"
    assert isinstance(result[0], str), "First element should be string (context)"
    assert isinstance(result[1], list), "Second element should be list (citations)"

    print("✅ test_format_docs_with_citations_returns_tuple passed")


def test_citation_numbers_are_sequential():
    """Test that citation numbers are 1-indexed and sequential."""
    docs = [
        Document(
            page_content="First passage",
            metadata={"source": "Chapter 1", "source_type": "vector", "score": 0.9},
        ),
        Document(
            page_content="Second passage",
            metadata={"source": "Chapter 2", "source_type": "vector", "score": 0.8},
        ),
        Document(
            page_content="川芎 --TREATS--> 頭痛",
            metadata={"source_type": "graph", "depth": 1},
        ),
    ]

    context, citations = format_docs_with_citations(docs)

    # Verify citation numbers
    numbers = [c["number"] for c in citations]
    assert numbers == [1, 2, 3], f"Expected [1, 2, 3] but got {numbers}"

    # Verify context contains citation markers
    assert "[1]" in context, "Context should contain [1]"
    assert "[2]" in context, "Context should contain [2]"
    assert "[3]" in context, "Context should contain [3]"

    print("✅ test_citation_numbers_are_sequential passed")


def test_text_citation_structure():
    """Test TextCitation has all required fields."""
    docs = [
        Document(
            page_content="陰陽者，天地之道也，萬物之綱紀，變化之父母。",
            metadata={
                "source": "素問·陰陽應象大論",
                "source_type": "vector",
                "score": 0.451,
                "id": "chunk_42",
            },
        )
    ]

    _, citations = format_docs_with_citations(docs)

    assert len(citations) == 1
    citation = citations[0]

    # Verify required fields
    assert citation["number"] == 1
    assert citation["type"] == "text"
    assert citation["source"] == "素問·陰陽應象大論"
    assert "content" in citation
    assert citation["chunk_id"] == "chunk_42"
    assert citation["score"] == 0.451

    print("✅ test_text_citation_structure passed")


def test_graph_citation_structure():
    """Test GraphCitation has all required fields."""
    docs = [
        Document(
            page_content="川芎 --TREATS--> 頭痛 (Symptom)",
            metadata={"source_type": "graph", "depth": 1},
        )
    ]

    _, citations = format_docs_with_citations(docs)

    assert len(citations) == 1
    citation = citations[0]

    # Verify required fields
    assert citation["number"] == 1
    assert citation["type"] == "graph"
    assert citation["fact"] == "川芎 --TREATS--> 頭痛 (Symptom)"
    assert citation["depth"] == 1
    assert "source_ref" in citation

    print("✅ test_graph_citation_structure passed")


def test_empty_docs_returns_empty():
    """Test that empty input returns empty results."""
    context, citations = format_docs_with_citations([])

    assert context == "", "Empty docs should return empty context"
    assert citations == [], "Empty docs should return empty citations"

    print("✅ test_empty_docs_returns_empty passed")


def test_mixed_source_types():
    """Test handling of mixed vector and graph documents."""
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
            metadata={"source": "Chapter 2", "source_type": "vector", "score": 0.7},
        ),
    ]

    _, citations = format_docs_with_citations(docs)

    # Vector docs are processed first, then graph docs
    # So ordering should be: text[1], text[2], graph[3]
    types = [c["type"] for c in citations]
    assert types == ["text", "text", "graph"], f"Expected text-first ordering, got {types}"

    print("✅ test_mixed_source_types passed")


if __name__ == "__main__":
    print("Running citation tests...\n")

    test_format_docs_with_citations_returns_tuple()
    test_citation_numbers_are_sequential()
    test_text_citation_structure()
    test_graph_citation_structure()
    test_empty_docs_returns_empty()
    test_mixed_source_types()

    print("\n✅ All citation tests passed!")
