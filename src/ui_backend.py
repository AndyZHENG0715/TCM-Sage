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
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from config import GRAPH_DATA_DEFAULT_RELATIVE
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
)

load_dotenv()

EMBEDDING_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"


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


def _get_huggingface_cache_root() -> Path:
    hf_home = os.getenv("HF_HOME")
    if hf_home:
        return Path(hf_home)

    xdg_cache_home = os.getenv("XDG_CACHE_HOME")
    if xdg_cache_home:
        return Path(xdg_cache_home) / "huggingface"

    return Path.home() / ".cache" / "huggingface"


def _get_local_embedding_snapshot(model_name: str) -> Path | None:
    repo_cache_dir = _get_huggingface_cache_root() / "hub" / f"models--{model_name.replace('/', '--')}"
    snapshots_dir = repo_cache_dir / "snapshots"
    if not snapshots_dir.exists():
        return None

    try:
        snapshots = sorted(
            (path for path in snapshots_dir.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return None

    return snapshots[0] if snapshots else None


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

    # Default: SymMap KG at GRAPH_DATA_PATH; override with GRAPH_DATA_PATH env (relative or absolute).
    raw_graph_path = os.getenv("GRAPH_DATA_PATH", GRAPH_DATA_DEFAULT_RELATIVE)
    graph_data_path = str(_resolve_path(raw_graph_path))

    if not Path(graph_data_path).exists():
        for rel in (
            "data/graph/entities.json",
            "data/graph/entities_partial.json",
        ):
            candidate = str(_resolve_path(rel))
            if Path(candidate).exists():
                graph_data_path = candidate
                break

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
        graph_data_path=graph_data_path,
        system_prompt=base.system_prompt,
    )


@lru_cache(maxsize=1)
def _get_embeddings() -> HuggingFaceEmbeddings:
    local_snapshot = _get_local_embedding_snapshot(EMBEDDING_MODEL_NAME)
    local_files_only = _env_flag("HF_LOCAL_FILES_ONLY", local_snapshot is not None)
    model_name_or_path = str(local_snapshot) if local_snapshot is not None else EMBEDDING_MODEL_NAME

    return HuggingFaceEmbeddings(
        model_name=model_name_or_path,
        model_kwargs={
            "trust_remote_code": True,
            "local_files_only": local_files_only,
        },
    )


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
) -> list[Document]:
    knowledge_graph = _get_knowledge_graph(graph_data_path)
    graph_docs: list[Document] = []

    for entity_id in knowledge_graph.search_by_name(query):
        entity = knowledge_graph.get_entity(entity_id)
        if not entity:
            continue

        related_entities = knowledge_graph.get_related_entities(
            entity_id,
            max_depth=depth,
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

    return graph_docs


def _retrieve_documents(query: str, config: PipelineConfig) -> list[Document]:
    vector_docs = vector_search_with_scores(
        get_shared_vectorstore(),
        query,
        config.retrieval_k,
    )

    if not config.hybrid_enabled:
        return vector_docs

    try:
        graph_docs = _search_graph_documents(
            query=query,
            graph_data_path=config.graph_data_path,
            depth=config.graph_depth,
        )
    except Exception as error:  # pragma: no cover - best effort fallback
        print(f"[Debug] Hybrid retrieval disabled for this request: {error}")
        return vector_docs

    return vector_docs + graph_docs


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

    return {
        "question": user_query,
        "answer": answer,
        "severity": severity,
        "temperature": selected_temp,
        "timestamp": datetime.utcnow().isoformat(),
        "provider": config.provider,
        "model": config.model,
        "retrieval_k": config.retrieval_k,
        "verification": build_verification_payload(verification_result),
        "verification_result": verification_result,
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
        "verification": build_verification_payload(verification_result),
        "verification_result": verification_result,
        "citations": citations,
        "debug_context": generation_context,
    }
