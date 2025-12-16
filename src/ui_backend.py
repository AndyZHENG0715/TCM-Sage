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
from langchain_core.runnables import RunnablePassthrough
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

    retrieval_k = int(os.getenv("RETRIEVAL_K", "5"))
    system_prompt = os.getenv(
        "SYSTEM_PROMPT",
        "You are an expert assistant specializing in Classical Chinese Medicine, "
        "specifically the Huangdi Neijing (黃帝内經). Your task is to answer questions "
        "accurately based ONLY on the provided source text. Your answer must be in the "
        "same language as the question. After providing the answer, cite the source "
        'chapter for the information you provide in a "Sources:" section.',
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

    retriever = vectorstore.as_retriever(k=retrieval_k)

    template = f"""{system_prompt}

Context:
{{context}}

Question:
{{question}}

Answer:
"""
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

    return {
        "config": PipelineConfig(
            provider=provider,
            model=model,
            informational_temperature=informational_temperature,
            prescriptive_temperature=prescriptive_temperature,
            classifier_provider=classifier_provider,
            classifier_model=classifier_model,
            classifier_temperature=classifier_temperature,
            retrieval_k=retrieval_k,
            system_prompt=system_prompt,
        ),
        "prompt": prompt,
        "retriever": retriever,
        "classifier_llm": classifier_llm,
        "llm_informational": llm_informational,
        "llm_prescriptive": llm_prescriptive,
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
    config: PipelineConfig = pipeline["config"]

    severity = get_query_severity(user_query, classifier_llm)

    if severity == "prescriptive":
        selected_llm = llm_prescriptive
        selected_temp = config.prescriptive_temperature
    else:
        selected_llm = llm_informational
        selected_temp = config.informational_temperature

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | selected_llm
        | StrOutputParser()
    )

    answer = rag_chain.invoke(user_query)

    return {
        "question": user_query,
        "answer": answer,
        "severity": severity,
        "temperature": selected_temp,
        "timestamp": datetime.utcnow().isoformat(),
        "provider": config.provider,
        "model": config.model,
        "retrieval_k": config.retrieval_k,
    }


def get_runtime_config() -> PipelineConfig:
    """
    Expose the cached pipeline configuration for display in the UI.
    """

    pipeline = _initialize_pipeline()
    return pipeline["config"]
