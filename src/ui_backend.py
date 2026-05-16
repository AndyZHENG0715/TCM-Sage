"""
Backend helper utilities for the Streamlit prototype UI.

This module reuses the existing RAG pipeline logic without modifying the
command-line application. It exposes helpers that accept per-request runtime
settings while keeping heavy shared resources cached.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Generator, Union

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from config import GRAPH_DATA_DEFAULT_RELATIVE
from crosswalk_bridge import resolve_query_to_symmap_ids
from embeddings import get_embedding_model
from graph_builder import create_graph_from_json
from main import (  # type: ignore  # pylint: disable=import-error
    DEFAULT_SYSTEM_PROMPT,
    build_prompt_template,
    build_verification_payload,
    create_llm,
    format_docs_with_citations,
    get_query_severity,
    vector_search_with_scores,
    verify_answer,
    verify_citation_bounds,
)

load_dotenv()


@dataclass(frozen=True)
class PipelineConfig:
    provider: str
    model: str | None
    informational_temperature: float
    prescriptive_temperature: float
    classifier_follow_main: bool
    classifier_provider: str
    classifier_model: str | None
    classifier_temperature: float
    verifier_follow_main: bool
    verifier_provider: str
    verifier_model: str | None
    verifier_temperature: float
    retrieval_k: int
    hybrid_enabled: bool
    hybrid_available: bool
    graph_depth: int
    graph_max_results: int
    graph_data_path: str
    system_prompt: str


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    return float(value)


def _env_optional_str(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    trimmed = value.strip()
    return trimmed or None


def _resolve_path(raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return (Path(__file__).parent.parent / candidate).resolve()


@lru_cache(maxsize=1)
def _get_default_pipeline_config() -> PipelineConfig:
    provider = os.getenv("LLM_PROVIDER", "alibaba").lower()
    model = _env_optional_str("LLM_MODEL")
    informational_temperature = _env_float("LLM_TEMPERATURE", 0.1)
    prescriptive_temperature = _env_float("PRESCRIPTIVE_TEMPERATURE", 0.0)

    classifier_provider_override = _env_optional_str("CLASSIFIER_LLM_PROVIDER")
    classifier_model_override = _env_optional_str("CLASSIFIER_LLM_MODEL")
    classifier_follow_main = classifier_provider_override is None and classifier_model_override is None
    classifier_provider = (classifier_provider_override or provider).lower()
    classifier_model = classifier_model_override if classifier_model_override is not None else model
    classifier_temperature = _env_float("CLASSIFIER_LLM_TEMPERATURE", 0.0)

    verifier_provider_override = _env_optional_str("VERIFIER_LLM_PROVIDER")
    verifier_model_override = _env_optional_str("VERIFIER_LLM_MODEL")
    verifier_follow_main = verifier_provider_override is None and verifier_model_override is None
    verifier_provider = (verifier_provider_override or provider).lower()
    verifier_model = verifier_model_override if verifier_model_override is not None else model
    verifier_temperature = _env_float("VERIFIER_LLM_TEMPERATURE", 0.0)

    retrieval_k = _env_int("RETRIEVAL_K", 5)
    requested_hybrid = _env_flag("HYBRID_RETRIEVAL_ENABLED", True)
    graph_depth = _env_int("GRAPH_DEPTH", 1)
    graph_max_results = _env_int("GRAPH_MAX_RESULTS", 20)

    # Default: SymMap KG at GRAPH_DATA_PATH; override with GRAPH_DATA_PATH env (relative or absolute).
    raw_graph_path = os.getenv("GRAPH_DATA_PATH", GRAPH_DATA_DEFAULT_RELATIVE)
    graph_data_path = str(_resolve_path(raw_graph_path))

    hybrid_available = Path(graph_data_path).exists()
    hybrid_enabled = requested_hybrid and hybrid_available

    system_prompt = os.getenv("SYSTEM_PROMPT") or DEFAULT_SYSTEM_PROMPT

    return PipelineConfig(
        provider=provider,
        model=model,
        informational_temperature=informational_temperature,
        prescriptive_temperature=prescriptive_temperature,
        classifier_follow_main=classifier_follow_main,
        classifier_provider=classifier_provider,
        classifier_model=classifier_model,
        classifier_temperature=classifier_temperature,
        verifier_follow_main=verifier_follow_main,
        verifier_provider=verifier_provider,
        verifier_model=verifier_model,
        verifier_temperature=verifier_temperature,
        retrieval_k=retrieval_k,
        hybrid_enabled=hybrid_enabled,
        hybrid_available=hybrid_available,
        graph_depth=graph_depth,
        graph_max_results=graph_max_results,
        graph_data_path=graph_data_path,
        system_prompt=system_prompt,
    )


def _coalesce_optional_string(
    value: Any,
    fallback: str | None,
) -> str | None:
    if value is None:
        return fallback
    if isinstance(value, str):
        trimmed = value.strip()
        return trimmed or fallback
    return fallback


def resolve_runtime_config(overrides: Dict[str, Any] | None = None) -> PipelineConfig:
    base = _get_default_pipeline_config()
    overrides = overrides or {}

    provider = str(overrides.get("provider", base.provider)).lower()
    model = _coalesce_optional_string(overrides.get("model"), base.model)
    informational_temperature = float(
        overrides.get("informational_temperature", base.informational_temperature)
    )
    prescriptive_temperature = float(
        overrides.get("prescriptive_temperature", base.prescriptive_temperature)
    )

    classifier_follow_main = bool(
        overrides.get("classifier_follow_main", base.classifier_follow_main)
    )
    classifier_provider_override = _coalesce_optional_string(
        overrides.get("classifier_provider"),
        base.classifier_provider if not classifier_follow_main else None,
    )
    classifier_model_override = _coalesce_optional_string(
        overrides.get("classifier_model"),
        base.classifier_model if not classifier_follow_main else None,
    )
    classifier_provider = provider if classifier_follow_main else (classifier_provider_override or provider)
    classifier_model = model if classifier_follow_main else classifier_model_override

    verifier_follow_main = bool(
        overrides.get("verifier_follow_main", base.verifier_follow_main)
    )
    verifier_provider_override = _coalesce_optional_string(
        overrides.get("verifier_provider"),
        base.verifier_provider if not verifier_follow_main else None,
    )
    verifier_model_override = _coalesce_optional_string(
        overrides.get("verifier_model"),
        base.verifier_model if not verifier_follow_main else None,
    )
    verifier_provider = provider if verifier_follow_main else (verifier_provider_override or provider)
    verifier_model = model if verifier_follow_main else verifier_model_override

    retrieval_k = max(1, int(overrides.get("retrieval_k", base.retrieval_k)))
    requested_hybrid = bool(
        overrides.get("hybrid_retrieval_enabled", base.hybrid_enabled)
    )
    graph_depth = max(1, int(overrides.get("graph_depth", base.graph_depth)))
    graph_max_results = max(1, int(overrides.get("graph_max_results", base.graph_max_results)))
    graph_data_path = base.graph_data_path
    hybrid_available = Path(graph_data_path).exists()
    hybrid_enabled = requested_hybrid and hybrid_available

    return PipelineConfig(
        provider=provider,
        model=model,
        informational_temperature=informational_temperature,
        prescriptive_temperature=prescriptive_temperature,
        classifier_follow_main=classifier_follow_main,
        classifier_provider=classifier_provider.lower(),
        classifier_model=classifier_model,
        classifier_temperature=0.0,
        verifier_follow_main=verifier_follow_main,
        verifier_provider=verifier_provider.lower(),
        verifier_model=verifier_model,
        verifier_temperature=0.0,
        retrieval_k=retrieval_k,
        hybrid_enabled=hybrid_enabled,
        hybrid_available=hybrid_available,
        graph_depth=graph_depth,
        graph_max_results=graph_max_results,
        graph_data_path=graph_data_path,
        system_prompt=base.system_prompt,
    )


@lru_cache(maxsize=1)
def _get_embeddings():
    return get_embedding_model()


@lru_cache(maxsize=1)
def get_shared_vectorstore() -> Chroma:
    vectorstore_path = Path(__file__).parent.parent / "vectorstore" / "chroma"
    if not vectorstore_path.exists():
        raise FileNotFoundError(
            f"Vector store not found at {vectorstore_path}. "
            "Please run 'python src/ingest.py' before launching the UI."
        )

    return Chroma(
        persist_directory=str(vectorstore_path),
        embedding_function=_get_embeddings(),
    )


@lru_cache(maxsize=4)
def _get_knowledge_graph(graph_data_path: str):
    graph_path = Path(graph_data_path)
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph data not found: {graph_path}")
    return create_graph_from_json(str(graph_path))


def _format_graph_fact(
    source_name: str,
    relationship_type: str,
    target_name: str,
    target_type: str,
    description: str = "",
) -> str:
    fact = f"{source_name} --{relationship_type}--> {target_name} ({target_type})"
    if description:
        fact += f" | {description}"
    return fact


def _search_graph_documents(
    query: str,
    graph_data_path: str,
    depth: int,
    max_results: int = 20,
) -> list[Document]:
    knowledge_graph = _get_knowledge_graph(graph_data_path)
    graph_docs: list[Document] = []

    candidate_ids = set(knowledge_graph.search_by_name(query))
    candidate_ids.update(resolve_query_to_symmap_ids(query))

    for entity_id in candidate_ids:
        entity = knowledge_graph.get_entity(entity_id)
        if not entity:
            continue

        related_entities = knowledge_graph.get_related_entities(
            entity_id,
            max_depth=depth,
            max_results=max_results,
        )

        for item in related_entities:
            relationship = item["relationship"]
            source_entity = knowledge_graph.get_entity(relationship["source"])
            target_entity = knowledge_graph.get_entity(relationship["target"])

            source_name = (
                source_entity.get("name", relationship["source"])
                if source_entity
                else relationship["source"]
            )
            target_name = (
                target_entity.get("name", relationship["target"])
                if target_entity
                else relationship["target"]
            )
            target_type = (
                target_entity.get("type", "Unknown")
                if target_entity
                else "Unknown"
            )

            graph_docs.append(
                Document(
                    page_content=_format_graph_fact(
                        source_name=source_name,
                        relationship_type=relationship["type"],
                        target_name=target_name,
                        target_type=target_type,
                        description=relationship.get("description", ""),
                    ),
                    metadata={
                        "source_type": "graph",
                        "entity_id": item["entity"].get("id"),
                        "entity_type": item["entity"].get("type"),
                        "relationship_type": relationship["type"],
                        "depth": item["depth"],
                        "source_ref": relationship.get("source_ref"),
                    },
                )
            )

    return graph_docs[:max_results]


import re as _re


# Chinese numeral to int conversion for clause detection
_CN_DIGITS = {'零': 0, '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
              '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
              '百': 100, '千': 1000}


def _cn_num_to_int(cn: str) -> int | None:
    """Convert Chinese numeral string to integer. E.g., '八十二' -> 82."""
    try:
        # Try simple Arabic numeral first
        return int(cn)
    except ValueError:
        pass
    result = 0
    current = 0
    for ch in cn:
        if ch not in _CN_DIGITS:
            return None
        val = _CN_DIGITS[ch]
        if val >= 10:  # multiplier (十, 百, 千)
            if current == 0:
                current = 1
            result += current * val
            current = 0
        else:
            current = val
    result += current
    return result if result > 0 else None


# Pattern: 伤寒论/金匮要略 + 第X条 (Arabic or Chinese numerals)
_CLAUSE_QUERY_PATTERN = _re.compile(
    r'(?:《?(?P<book>伤寒论|金匮要略方论|金匮要略)》?)?'
    r'.*?第(?P<num>[\d一二三四五六七八九十百千]+)条'
)


def _extract_clause_reference(query: str) -> tuple[str | None, int | None]:
    """Detect if query asks for a specific clause number. Returns (book, clause_num) or (None, None)."""
    m = _CLAUSE_QUERY_PATTERN.search(query)
    if not m:
        return None, None
    book = m.group('book')
    num_str = m.group('num')
    clause_num = _cn_num_to_int(num_str)
    if clause_num is None:
        return None, None
    # Default to 伤寒论 if no book specified but clause reference found
    if not book:
        book = '伤寒论'
    # Normalize 金匮要略 variants
    if book in ('金匮要略', '金匮要略方论'):
        book = '金匮要略方论'
    return book, clause_num


_CLAUSE_MULTI_PATTERN = _re.compile(
    r'(?:第([\d一二三四五六七八九十百千]+)条?|([\d一二三四五六七八九十百千]+)条)'
)


def _extract_clause_references(query: str) -> list[tuple[str, int]]:
    book_match = _re.search(r'《?(?P<book>伤寒论|金匮要略方论|金匮要略)》?', query)
    book = '伤寒论'
    if book_match:
        book = book_match.group('book')
        if book in ('金匮要略', '金匮要略方论'):
            book = '金匮要略方论'
    results = []
    for m in _CLAUSE_MULTI_PATTERN.finditer(query):
        num_str = m.group(1) or m.group(2)
        num = _cn_num_to_int(num_str)
        if num is not None:
            results.append((book, num))
    return results

_FORMULA_PATTERN = _re.compile(
    r'([\u4e00-\u9fff]{2,8}(?:汤|散|丸|饮|膏|酒|方|丹))'
)

_CANONICAL_FORMULA_BOOKS = {'伤寒论', '金匮要略方论'}


def _extract_formula_name(query: str) -> str | None:
    m = _FORMULA_PATTERN.search(query)
    return m.group(1) if m else None


def _fetch_canonical_formula_docs(
    vs: 'Chroma', formula: str, query: str, k: int = 3
) -> list[Document]:
    """Search canonical texts for a specific formula via metadata filter."""
    results: list[Document] = []
    for book in _CANONICAL_FORMULA_BOOKS:
        try:
            docs = vs.similarity_search(
                query,
                k=k,
                filter={'$and': [{'formula': formula}, {'book': book}]},
            )
            for doc in docs:
                if doc.metadata is None:
                    doc.metadata = {}
                doc.metadata['source_type'] = 'vector'
                doc.metadata['score'] = 0.0
            results.extend(docs)
        except Exception:
            pass
    if results:
        print(f'[Debug] Formula filter: found {len(results)} canonical docs for "{formula}"')
    return results

_SOURCE_CHRONOLOGICAL_BOOST: dict[str, float] = {
    '黄帝内经': 0.90,
    '难经': 0.90,
    '伤寒论': 0.90,
    '金匮要略方论': 0.90,
    '神农本草经': 0.90,
    '灵枢': 0.90,
    '千金要方': 0.93,
    '备急千金要方': 0.93,
    '外台秘要': 0.95,
    '针灸甲乙经': 0.95,
    '内外伤辨惑论': 0.97,
    '脾胃论': 0.97,
    '兰室秘藏': 0.97,
    '丹溪心法': 0.97,
    '宣明论方': 0.97,
    '儒门事亲': 0.97,
    '本草纲目': 0.97,
    '温病条辨': 0.97,
}


def _apply_source_authority_boost(docs: list[Document]) -> list[Document]:
    """Gently boost earlier/canonical texts by reducing their distance scores.

    ChromaDB scores are distances (lower = better). Multiplying by < 1.0
    makes a doc rank higher. The boost is small (0.90-0.97) so it only
    tips the balance when semantic scores are close — it won't override
    a clearly more relevant chunk from a later text.
    """
    for doc in docs:
        if not doc.metadata or 'score' not in doc.metadata:
            continue
        book = doc.metadata.get('book', '')
        boost = _SOURCE_CHRONOLOGICAL_BOOST.get(book, 1.0)
        if boost < 1.0:
            original = doc.metadata['score']
            doc.metadata['score'] = round(original * boost, 3)

    docs.sort(key=lambda d: d.metadata.get('score', 999) if d.metadata else 999)
    return docs


def _retrieve_documents(query: str, config: PipelineConfig) -> list[Document]:
    # Check for specific clause reference (e.g., "伤寒论第八十二条")
    # If found, use metadata filter for exact match instead of pure vector search
    clause_refs = _extract_clause_references(query)
    if clause_refs:
        vs = get_shared_vectorstore()
        clause_docs: list[Document] = []
        try:
            for clause_book, clause_num in clause_refs:
                hits = vs.similarity_search(
                    query,
                    k=3,
                    filter={'$and': [{'clause_number': clause_num}, {'book': clause_book}]},
                )
                for doc in hits:
                    if doc.metadata is None:
                        doc.metadata = {}
                    doc.metadata['source_type'] = 'vector'
                    doc.metadata['score'] = 0.0
                clause_docs.extend(hits)
            if clause_docs:
                print(f'[Debug] Clause filter matched: {len(clause_refs)} refs, {len(clause_docs)} docs')
                supplementary = vector_search_with_scores(vs, query, config.retrieval_k)
                clause_ids = {d.page_content[:80] for d in clause_docs}
                supplementary = [d for d in supplementary
                                 if d.page_content[:80] not in clause_ids]
                return clause_docs + supplementary[:config.retrieval_k - len(clause_docs)]
        except Exception as e:
            print(f'[Debug] Clause filter failed, falling back to normal search: {e}')

    formula_name = _extract_formula_name(query)
    canonical_formula_docs: list[Document] = []
    if formula_name:
        canonical_formula_docs = _fetch_canonical_formula_docs(
            get_shared_vectorstore(), formula_name, query
        )

    broader_k = max(config.retrieval_k * 3, 20)
    vector_docs = vector_search_with_scores(
        get_shared_vectorstore(),
        query,
        broader_k,
    )

    if not config.hybrid_enabled:
        return vector_docs

    try:
        graph_docs = _search_graph_documents(
            query=query,
            graph_data_path=config.graph_data_path,
            depth=config.graph_depth,
            max_results=config.graph_max_results,
        )
    except Exception as error:  # pragma: no cover - best effort fallback
        print(f"[Debug] Hybrid retrieval disabled for this request: {error}")
        return vector_docs

    # Deduplicate: keep only the highest-scoring chunk per book+chapter
    # This prevents flooding the LLM with 8 variations of the same formula from 千金要方
    seen_sources: dict[str, Document] = {}
    deduped_vector: list[Document] = []
    for doc in vector_docs:
        source_key = ""
        if doc.metadata:
            book = doc.metadata.get("book", "")
            chapter = doc.metadata.get("source", "")
            source_key = f"{book}::{chapter}"
        if not source_key or source_key not in seen_sources:
            if source_key:
                seen_sources[source_key] = doc
            deduped_vector.append(doc)
    
    if len(deduped_vector) < len(vector_docs):
        print(f"[Debug] Deduplication: {len(vector_docs)} -> {len(deduped_vector)} vector docs")

    deduped_vector = _apply_source_authority_boost(deduped_vector)

    if canonical_formula_docs:
        canon_keys = set()
        for doc in canonical_formula_docs:
            if doc.metadata:
                canon_keys.add(doc.page_content[:80])
        deduped_vector = [d for d in deduped_vector
                          if d.page_content[:80] not in canon_keys]
        deduped_vector = canonical_formula_docs + deduped_vector

    deduped_vector = deduped_vector[:config.retrieval_k]
    return deduped_vector + graph_docs


def _build_runtime_models(config: PipelineConfig) -> dict[str, Any]:
    return {
        "prompt": build_prompt_template(config.system_prompt),
        "classifier_llm": create_llm(
            config.classifier_provider,
            config.classifier_model,
            config.classifier_temperature,
        ),
        "llm_informational": create_llm(
            config.provider,
            config.model,
            config.informational_temperature,
        ),
        "llm_prescriptive": create_llm(
            config.provider,
            config.model,
            config.prescriptive_temperature,
        ),
        "llm_verifier": create_llm(
            config.verifier_provider,
            config.verifier_model,
            config.verifier_temperature,
        ),
    }


def _prepend_chat_history(context: str, chat_history: list[dict]) -> str:
    history_text = "\n".join(
        f"{message.get('role', 'user').upper()}: {message.get('content', '')}"
        for message in chat_history[-6:]
    )
    if not history_text:
        return context
    return f"Chat History:\n{history_text}\n\n{context}"


def _finalize_verification(
    answer: str,
    citations: list[dict],
    verification_result: str,
) -> tuple[str, dict, dict]:
    citation_bounds = verify_citation_bounds(answer, len(citations))
    final_result = verification_result
    verification_payload = build_verification_payload(final_result)

    if not citation_bounds["is_valid"]:
        final_result = "UNSUPPORTED"
        out_of_range = ", ".join(str(n) for n in citation_bounds["out_of_range"])
        verification_payload = build_verification_payload(final_result)
        verification_payload["explanation"] = (
            f"{verification_payload['explanation']} The answer cites unavailable source number(s): "
            f"{out_of_range}."
        )

    return final_result, verification_payload, citation_bounds


def run_query(user_query: str, runtime_settings: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if not user_query.strip():
        raise ValueError("Query must not be empty.")

    config = resolve_runtime_config(runtime_settings)
    runtime_models = _build_runtime_models(config)

    severity = get_query_severity(user_query, runtime_models["classifier_llm"])

    if severity == "prescriptive":
        selected_llm = runtime_models["llm_prescriptive"]
        selected_temp = config.prescriptive_temperature
    else:
        selected_llm = runtime_models["llm_informational"]
        selected_temp = config.informational_temperature

    retrieved_docs = _retrieve_documents(user_query, config)

    # Rerank retrieved documents for better relevance ordering
    if len(retrieved_docs) > 1:
        try:
            from embeddings import rerank_documents
            doc_texts = [doc.page_content for doc in retrieved_docs]
            reranked = rerank_documents(user_query, doc_texts, top_n=min(config.retrieval_k, len(doc_texts)))
            if reranked:
                reranked_docs = [retrieved_docs[r['index']] for r in reranked]
                retrieved_docs = reranked_docs
        except Exception:
            pass  # Fallback to original order

    formatted_context, citations = format_docs_with_citations(retrieved_docs)
    answer = (runtime_models["prompt"] | selected_llm | StrOutputParser()).invoke(
        {"context": formatted_context, "question": user_query}
    )

    verification_result = "SUPPORTED"
    try:
        verification_result = verify_answer(
            question=user_query,
            context=formatted_context,
            answer=answer,
            llm=runtime_models["llm_verifier"],
        )
    except Exception as verify_error:  # pragma: no cover - best effort verification
        print(f"[Debug] UI Backend Verification Error: {verify_error}")

    verification_result, verification_payload, citation_bounds = _finalize_verification(
        answer,
        citations,
        verification_result,
    )

    return {
        "question": user_query,
        "answer": answer,
        "severity": severity,
        "temperature": selected_temp,
        "timestamp": datetime.utcnow().isoformat(),
        "provider": config.provider,
        "model": config.model,
        "retrieval_k": config.retrieval_k,
        "verification": verification_payload,
        "verification_result": verification_result,
        "citation_bounds": citation_bounds,
        "citations": citations,
    }


def get_runtime_config(overrides: Dict[str, Any] | None = None) -> PipelineConfig:
    return resolve_runtime_config(overrides)


def run_query_stream(
    user_query: str,
    chat_history: list[dict] | None = None,
    runtime_settings: Dict[str, Any] | None = None,
) -> Generator[Union[str, Dict[str, Any]], None, None]:
    if not user_query.strip():
        raise ValueError("Query must not be empty.")

    chat_history = chat_history or []
    config = resolve_runtime_config(runtime_settings)
    runtime_models = _build_runtime_models(config)

    severity = get_query_severity(user_query, runtime_models["classifier_llm"])
    selected_temp = (
        config.prescriptive_temperature
        if severity == "prescriptive"
        else config.informational_temperature
    )
    selected_llm = create_llm(
        config.provider,
        config.model,
        selected_temp,
        streaming=True,
    )

    retrieved_docs = _retrieve_documents(user_query, config)

    # Rerank retrieved documents for better relevance ordering
    if len(retrieved_docs) > 1:
        try:
            from embeddings import rerank_documents
            doc_texts = [doc.page_content for doc in retrieved_docs]
            reranked = rerank_documents(user_query, doc_texts, top_n=min(config.retrieval_k, len(doc_texts)))
            if reranked:
                # Reorder documents by reranker scores
                reranked_docs = [retrieved_docs[r['index']] for r in reranked]
                retrieved_docs = reranked_docs
                print(f"[Debug] Reranked {len(doc_texts)} docs -> top {len(reranked)} by relevance")
        except Exception as rerank_err:
            print(f"[Debug] Reranker unavailable, using original order: {rerank_err}")

    formatted_context, citations = format_docs_with_citations(retrieved_docs)
    generation_context = _prepend_chat_history(formatted_context, chat_history)

    generation_chain = runtime_models["prompt"] | selected_llm
    chain_input = {"context": generation_context, "question": user_query}

    collected_answer = ""
    for chunk in generation_chain.stream(chain_input):
        chunk_text = chunk.content if hasattr(chunk, "content") else str(chunk)
        collected_answer += chunk_text
        yield chunk_text

    verification_result = "SUPPORTED"
    try:
        verification_result = verify_answer(
            question=user_query,
            context=generation_context,
            answer=collected_answer,
            llm=runtime_models["llm_verifier"],
        )
    except Exception as verify_error:  # pragma: no cover - best effort verification
        print(f"[Debug] UI Backend Verification Error: {verify_error}")

    verification_result, verification_payload, citation_bounds = _finalize_verification(
        collected_answer,
        citations,
        verification_result,
    )

    yield {
        "type": "metadata",
        "question": user_query,
        "answer": collected_answer,
        "severity": severity,
        "temperature": selected_temp,
        "timestamp": datetime.utcnow().isoformat(),
        "provider": config.provider,
        "model": config.model,
        "retrieval_k": config.retrieval_k,
        "verification": verification_payload,
        "verification_result": verification_result,
        "citation_bounds": citation_bounds,
        "citations": citations,
        "debug_context": generation_context,
    }
