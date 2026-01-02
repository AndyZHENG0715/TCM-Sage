"""
Backend helper utilities for the Streamlit prototype UI.

This module reuses the existing RAG pipeline logic without modifying the
command-line application. It exposes a `run_query` helper that the UI layer can
call to obtain answers along with severity and temperature metadata.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

# Ensure we can import from the existing CLI module without restructuring.
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from main import (  # type: ignore  # pylint: disable=import-error
    create_llm,
    format_docs,
    get_query_severity,
    verify_answer,
)


@dataclass(frozen=True)
class PipelineConfig:
    provider: str
    model: str | None
    informational_temperature: float
    prescriptive_temperature: float
    classifier_provider: str
    classifier_model: str | None
    classifier_temperature: float
    verifier_provider: str
    verifier_model: str | None
    verifier_temperature: float
    retrieval_k: int
    system_prompt: str


@lru_cache(maxsize=1)
def _initialize_pipeline() -> Dict[str, Any]:
    """
    Lazily build and cache the resources required to answer questions.
    Returns a dictionary to keep the implementation simple for the prototype.
    """

    load_dotenv()

    provider = os.getenv("LLM_PROVIDER", "alibaba").lower()
    model = os.getenv("LLM_MODEL") or None
    informational_temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    prescriptive_temperature = float(os.getenv("PRESCRIPTIVE_TEMPERATURE", "0.0"))

    classifier_provider = os.getenv("CLASSIFIER_LLM_PROVIDER", provider).lower()
    classifier_model = os.getenv("CLASSIFIER_LLM_MODEL") or None
    classifier_temperature = float(os.getenv("CLASSIFIER_LLM_TEMPERATURE", "0.0"))

    verifier_provider = os.getenv("VERIFIER_LLM_PROVIDER", provider).lower()
    verifier_model = os.getenv("VERIFIER_LLM_MODEL") or None
    verifier_temperature = float(os.getenv("VERIFIER_LLM_TEMPERATURE", "0.0"))

    retrieval_k = int(os.getenv("RETRIEVAL_K", "5"))

    # Hybrid retrieval configuration
    hybrid_enabled = os.getenv("HYBRID_RETRIEVAL_ENABLED", "false").lower() == "true"
    graph_data_path = os.getenv("GRAPH_DATA_PATH", "data/graph/entities.json")
    graph_depth = int(os.getenv("GRAPH_DEPTH", "1"))

    system_prompt = os.getenv("SYSTEM_PROMPT")
    if not system_prompt:
        system_prompt = (
            "You are an expert assistant specializing in Classical Chinese Medicine, "
            "specifically the Huangdi Neijing (黃帝内經). Your task is to answer questions "
            "accurately based ONLY on the provided source text. Your answer must be in the "
            "same language as the question. After providing the answer, cite the source "
            'chapter for the information you provide in a "Sources:" section.'
        )

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore_path = Path(__file__).parent.parent / "vectorstore" / "chroma"
    if not vectorstore_path.exists():
        raise FileNotFoundError(
            f"Vector store not found at {vectorstore_path}. "
            "Please run 'python src/ingest.py' before launching the UI."
        )

    vectorstore = Chroma(
        persist_directory=str(vectorstore_path),
        embedding_function=embeddings,
    )

    # Create retriever (standard or hybrid)
    if hybrid_enabled:
        try:
            from retriever import create_hybrid_retriever
            hybrid_retriever = create_hybrid_retriever(
                vectorstore_path=str(vectorstore_path),
                graph_data_path=graph_data_path,
                vector_k=retrieval_k,
                graph_depth=graph_depth,
            )

            # Wrap in RunnableLambda for LangChain pipe compatibility
            retriever = RunnableLambda(lambda query: hybrid_retriever.hybrid_search(query))
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to initialize hybrid retriever: {e}. Falling back to vector.")
            retriever = vectorstore.as_retriever(k=retrieval_k)
    else:
        retriever = vectorstore.as_retriever(k=retrieval_k)

    template = system_prompt + "\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:\n"
    prompt = ChatPromptTemplate.from_template(template)

    classifier_llm = create_llm(
        classifier_provider,
        classifier_model,
        classifier_temperature,
    )

    llm_informational = create_llm(
        provider,
        model,
        informational_temperature,
    )

    llm_prescriptive = create_llm(
        provider,
        model,
        prescriptive_temperature,
    )

    llm_verifier = create_llm(
        verifier_provider,
        verifier_model,
        verifier_temperature,
    )

    return {
        "config": PipelineConfig(
            provider=provider,
            model=model,
            informational_temperature=informational_temperature,
            prescriptive_temperature=prescriptive_temperature,
            classifier_provider=classifier_provider,
            classifier_model=classifier_model,
            classifier_temperature=classifier_temperature,
            verifier_provider=verifier_provider,
            verifier_model=verifier_model,
            verifier_temperature=verifier_temperature,
            retrieval_k=retrieval_k,
            system_prompt=system_prompt,
        ),
        "prompt": prompt,
        "retriever": retriever,
        "classifier_llm": classifier_llm,
        "llm_informational": llm_informational,
        "llm_prescriptive": llm_prescriptive,
        "llm_verifier": llm_verifier,
    }


def run_query(user_query: str) -> Dict[str, Any]:
    """
    Execute a single query through the RAG pipeline.

    Returns a dictionary containing:
        - question
        - answer
        - severity
        - temperature
        - timestamp (ISO format)
    """

    if not user_query.strip():
        raise ValueError("Query must not be empty.")

    pipeline = _initialize_pipeline()
    classifier_llm = pipeline["classifier_llm"]
    prompt = pipeline["prompt"]
    retriever = pipeline["retriever"]
    llm_informational = pipeline["llm_informational"]
    llm_prescriptive = pipeline["llm_prescriptive"]
    llm_verifier = pipeline["llm_verifier"]
    config: PipelineConfig = pipeline["config"]

    severity = get_query_severity(user_query, classifier_llm)

    if severity == "prescriptive":
        selected_llm = llm_prescriptive
        selected_temp = config.prescriptive_temperature
    else:
        selected_llm = llm_informational
        selected_temp = config.informational_temperature

    # Build RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | selected_llm
        | StrOutputParser()
    )

    # Retrieve context for verification
    retrieved_docs = retriever.invoke(user_query)
    formatted_context = format_docs(retrieved_docs)

    answer = rag_chain.invoke(user_query)

    # Self-critique verification step
    verification_result = "SUPPORTED"
    try:
        verification_result = verify_answer(
            question=user_query,
            context=formatted_context,
            answer=answer,
            llm=llm_verifier
        )
    except Exception as verify_error:
        # Log error in background but proceed
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
        "verification_result": verification_result,
    }


def get_runtime_config() -> PipelineConfig:
    """
    Expose the cached pipeline configuration for display in the UI.
    """

    pipeline = _initialize_pipeline()
    return pipeline["config"]
