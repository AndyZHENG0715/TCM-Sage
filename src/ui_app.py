"""
Streamlit prototype UI for TCM-Sage.

This interface is intentionally lightweight so it can be deprecated later without
affecting the core CLI workflow.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

import streamlit as st

# Ensure we can import the helper module without restructuring the existing src codebase.
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from ui_backend import get_runtime_config, run_query  # type: ignore  # pylint: disable=import-error

st.set_page_config(
    page_title="TCM-Sage Prototype",
    page_icon="🌿",
    layout="wide",
)

st.title("🌿 TCM-Sage Prototype UI")

# Feedback button in header area for mobile visibility
feedback_url = os.getenv("FEEDBACK_FORM_URL")
if feedback_url:
    header_cols = st.columns([4, 1])
    with header_cols[0]:
        st.caption("Discovery UI for demonstrating query routing and evidence-backed answers.")
    with header_cols[1]:
        st.link_button("📝 Feedback", feedback_url, type="primary", use_container_width=True)
else:
    st.caption("Discovery UI for demonstrating query routing and evidence-backed answers.")

if "history" not in st.session_state:
    st.session_state.history: List[dict] = []

if "query_input" not in st.session_state:
    st.session_state.query_input = ""


def handle_submit(query: str) -> None:
    with st.spinner("Analyzing and generating answer..."):
        result = run_query(query)
    st.session_state.history.insert(0, result)


def set_query(q: str) -> None:
    st.session_state.query_input = q


with st.sidebar:
    st.header("Configuration")
    try:
        config = get_runtime_config()
        st.markdown(
            f"""
- **Main Provider:** `{config.provider}`
- **Model Override:** `{config.model or "default"}`
- **Informational Temp:** `{config.informational_temperature}`
- **Prescriptive Temp:** `{config.prescriptive_temperature}`
- **Classifier Provider:** `{config.classifier_provider}`
- **Classifier Model:** `{config.classifier_model or "default"}`
- **Verifier Provider:** `{config.verifier_provider}`
- **Verifier Model:** `{config.verifier_model or "default"}`
- **Retriever k:** `{config.retrieval_k}`
            """
        )
    except Exception as sidebar_error:  # pylint: disable=broad-except
        st.error(f"Unable to load configuration: {sidebar_error}")

    st.divider()
    st.header("Sample Questions")
    st.caption("Click to populate the search box")
    
    st.button("1. 阴阳是什么？ (Concepts)", on_click=set_query, args=("阴阳是什么？",), use_container_width=True)
    st.button("2. 頭痛如何治療？ (Clinical)", on_click=set_query, args=("頭痛如何治療？",), use_container_width=True)
    st.button("3. Neijing vs COVID-19 (Safety)", on_click=set_query, args=("黄帝内经怎么看待COVID-19",), use_container_width=True)

    st.divider()
    st.markdown(
        "⚠️ This prototype runs on live APIs. Keep queries concise to control latency and cost."
    )


st.subheader("Ask a question about the Huangdi Neijing")
query = st.text_area(
    "Your question",
    placeholder="例如：陰陽是什麼？ or 頭痛應該用什麼方劑？",
    key="query_input",
)

col1, col2 = st.columns([1, 1])
with col1:
    submit_clicked = st.button("Generate Answer", type="primary", use_container_width=True)
with col2:
    if st.button("Clear History", use_container_width=True):
        st.session_state.history.clear()
        st.success("History cleared.")

if submit_clicked:
    if not query.strip():
        st.warning("Please enter a valid question before submitting.")
    else:
        try:
            handle_submit(query.strip())
        except Exception as submit_error:  # pylint: disable=broad-except
            st.error(f"Unable to generate answer: {submit_error}")


if st.session_state.history:
    latest = st.session_state.history[0]
    st.divider()
    st.subheader("Latest Answer")
    st.markdown(f"**Detected Severity:** `{latest['severity']}`")
    st.markdown(f"**Temperature Used:** `{latest['temperature']}`")
    st.markdown(f"**Provider:** `{latest['provider']}`")
    st.markdown(f"**Model:** `{latest['model'] or 'default'}`")
    st.write(latest["answer"])
    if latest.get("verification_result") == "UNSUPPORTED":
        st.warning("⚠️ [Self-Critique Warning]: This answer may contain information not directly supported by the provided citations.")
    else:
        st.success("✅ [Self-Critique Pass]: This answer has been verified against the provided citations.")
else:
    st.info("No queries yet. Ask a question to see the answer here.")


st.divider()
st.subheader("Session History")
if not st.session_state.history:
    st.caption("Ask a question to start building history.")
else:
    for idx, item in enumerate(st.session_state.history, start=1):
        with st.expander(f"{idx}. {item['question']} — {item['severity']} @ {item['timestamp']}"):
            st.markdown(f"**Temperature:** `{item['temperature']}`")
            st.write(item["answer"])
            if item.get("verification_result") == "UNSUPPORTED":
                st.warning("⚠️ [Self-Critique Warning]: This answer may contain information not directly supported by the provided citations.")
            else:
                st.success("✅ [Self-Critique Pass]: This answer has been verified against the provided citations.")
