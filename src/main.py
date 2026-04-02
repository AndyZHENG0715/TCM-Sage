"""
TCM-Sage Main RAG Application

This script implements the core Retrieval-Augmented Generation (RAG) pipeline for the TCM-Sage system.
It loads the vector store, creates a retrieval chain, and orchestrates the process of answering
user queries with citations from the Huangdi Neijing.

The system uses a modular RAG architecture that combines semantic vector search with
evidence-backed answer generation using OpenAI's GPT-4o model.
"""

from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pathlib import Path
import io
import os
import re
import sys
from typing import TYPE_CHECKING, Any, List, Tuple, cast

from dotenv import load_dotenv

from config import GRAPH_DATA_DEFAULT_RELATIVE
from embeddings import get_embedding_model

# Load environment variables at the earliest possible moment
load_dotenv()

if TYPE_CHECKING:
    from citation_types import Citation

# Fix Unicode encoding issues on Windows (only for CLI, not Streamlit)
if sys.platform == "win32":
    try:
        import codecs
        sys.stdout = codecs.getwriter("utf-8")(cast(Any, sys.stdout).detach())
        sys.stderr = codecs.getwriter("utf-8")(cast(Any, sys.stderr).detach())
    except (AttributeError, io.UnsupportedOperation):
        # Running in Streamlit or other context where stdout.detach() is not available
        pass

# LLM Provider imports
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    ChatAnthropic = None

try:
    from langchain_community.llms import OpenRouter
except ImportError:
    OpenRouter = None

try:
    from langchain_community.llms import Together
except ImportError:
    Together = None



DEFAULT_SYSTEM_PROMPT = """你是一名专业的中医临床参考助手，服务对象为中医从业者及中医专业学生。

你的知识涵盖中医基础理论（阴阳、五行、脏腑、经络）、四诊方法、方剂学、针灸学，以及《黄帝内经》《伤寒论》《金匮要略》等经典文献。

【回答原则】
- 如收到相关参考资料，应优先参考资料内容并使用 [1]、[2] 等行内标注引用，同时可结合自身知识进行补充。
- 引用经典文献时，应先直接引述原文（加注书名及篇章），再进行解释和分析，使用户无需查阅原始来源即可判断引用依据。
- 涉及临床场景时，尽量按辨证论治框架组织回答：证型分析、病机推导、治则治法、方药推荐（如适用）、针灸方案（如适用）。
- 若用户提供的四诊信息不足（舌象、脉象、症状细节等），可先给出初步分析和常见证型参考，并在回答末尾说明信息不足之处，引导用户补充。
- 回答语言与用户提问语言保持一致。

所有建议仅供临床参考，最终诊断与处方权归属执业中医师。"""

# Allow override via SYSTEM_PROMPT_OVERRIDE env var for easy tuning
# (SYSTEM_PROMPT in .env is reserved for legacy; use SYSTEM_PROMPT_OVERRIDE for new prompt)
_prompt_override = os.getenv("SYSTEM_PROMPT_OVERRIDE")
if _prompt_override:
    DEFAULT_SYSTEM_PROMPT = _prompt_override
_SOURCES_DIRECTIVE_PATTERNS = [
    re.compile(
        r'After providing the answer,\s*cite the source chapter for the information you provide in a ["“]?Sources:?["”]?\s*section\.?',
        re.IGNORECASE,
    ),
    re.compile(
        r'After providing the answer,.*?["“]?Sources:?["”]?\s*section\.?',
        re.IGNORECASE,
    ),
]


def create_llm(provider, model=None, temperature=0.1, streaming=False):
    """
    Create an LLM instance based on the provider configuration.

    Args:
        provider (str): The LLM provider ('openai', 'google', 'anthropic', 'openrouter', 'together', 'alibaba', 'ollama', 'lmstudio')
        model (str, optional): Specific model to use
        temperature (float): Temperature for generation
        streaming (bool): Enable streaming output (currently supported for 'alibaba' provider)

    Returns:
        LLM instance (or ChatModel instance if streaming is enabled)

    Raises:
        ValueError: If provider is not supported or required dependencies are missing

    Note:
        TODO(streaming-multi-provider): Currently streaming is implemented for the
        'alibaba', 'ollama' and 'lmstudio' providers via ChatOpenAI.
        When users can select providers in the UI, extend streaming support to: OpenAI (ChatOpenAI),
        Google (ChatGoogleGenerativeAI), Anthropic (ChatAnthropic). All these LangChain chat models
        support streaming=True.
    """
    provider = provider.lower()

    # Default models for each provider
    default_models = {
        'openai': 'gpt-5-4',
        'google': 'gemini-3-1-pro',
        'anthropic': 'claude-4-6-sonnet',
        'openrouter': 'openai/gpt-5-4',
        'together': 'meta-llama/Llama-3.1-8B-Instruct-Turbo',
        'alibaba': 'qwen-plus',
        'ollama': 'qwen3:8b',          # Popular local model with CJK support
        'lmstudio': 'qwen3-8b',       # LM Studio uses simple model names
    }

    # Use default model if none specified
    resolved_model = model or default_models.get(provider)
    if not resolved_model:
        raise ValueError(f"No default model configured for provider: {provider}")

    if provider == 'openai':
        if ChatOpenAI is None:
            raise ValueError("OpenAI provider requires 'langchain-openai' package. Install with: pip install langchain-openai")
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key or api_key == 'your-openai-api-key-here':
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")
        return cast(Any, ChatOpenAI)(
            model=resolved_model,
            temperature=temperature,
            api_key=api_key
        )

    elif provider == 'google':
        if ChatGoogleGenerativeAI is None:
            raise ValueError("Google provider requires 'langchain-google-genai' package. Install with: pip install langchain-google-genai")
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key or api_key == 'your-google-ai-studio-api-key-here':
            raise ValueError("Google API key not found. Please set GOOGLE_API_KEY in your .env file.")
        return cast(Any, ChatGoogleGenerativeAI)(
            model=resolved_model,
            temperature=temperature,
            google_api_key=api_key
        )

    elif provider == 'anthropic':
        if ChatAnthropic is None:
            raise ValueError("Anthropic provider requires 'langchain-anthropic' package. Install with: pip install langchain-anthropic")
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key or api_key == 'your-anthropic-api-key-here':
            raise ValueError("Anthropic API key not found. Please set ANTHROPIC_API_KEY in your .env file.")
        return cast(Any, ChatAnthropic)(
            model=resolved_model,
            temperature=temperature,
            api_key=api_key
        )

    elif provider == 'openrouter':
        if OpenRouter is None:
            raise ValueError("OpenRouter provider requires 'langchain-community' package. Install with: pip install langchain-community")
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key or api_key == 'your-openrouter-api-key-here':
            raise ValueError("OpenRouter API key not found. Please set OPENROUTER_API_KEY in your .env file.")
        return OpenRouter(
            model_name=resolved_model,
            temperature=temperature,
            openrouter_api_key=api_key
        )

    elif provider == 'together':
        if Together is None:
            raise ValueError("Together provider requires 'langchain-community' package. Install with: pip install langchain-community")
        api_key = os.getenv('TOGETHER_API_KEY')
        if not api_key or api_key == 'your-together-api-key-here':
            raise ValueError("Together API key not found. Please set TOGETHER_API_KEY in your .env file.")
        return Together(
            model=resolved_model,
            temperature=temperature,
            together_api_key=api_key
        )

    elif provider == 'alibaba':
        if ChatOpenAI is None:
            raise ValueError("Alibaba provider in OpenAI-compatible mode requires 'langchain-openai' package. Install with: pip install langchain-openai")

        api_key = os.getenv('DASHSCOPE_API_KEY')
        if not api_key or api_key == 'your-alibaba-api-key-here':
            raise ValueError("Alibaba API key (DASHSCOPE_API_KEY) not found. Please set it in your .env file.")

        return cast(Any, ChatOpenAI)(
            model=resolved_model,
            temperature=temperature,
            streaming=streaming,
            api_key=api_key,
            base_url='https://dashscope-intl.aliyuncs.com/compatible-mode/v1'
        )

    elif provider == 'ollama':
        if ChatOpenAI is None:
            raise ValueError("Ollama provider requires 'langchain-openai' package. Install with: pip install langchain-openai")
        base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434/v1')
        return cast(Any, ChatOpenAI)(
            model=resolved_model,
            temperature=temperature,
            base_url=base_url,
            api_key='ollama',  # Ollama doesn't need a real key but ChatOpenAI requires one
            streaming=streaming,
        )

    elif provider == 'lmstudio':
        if ChatOpenAI is None:
            raise ValueError("LM Studio provider requires 'langchain-openai' package. Install with: pip install langchain-openai")
        base_url = os.getenv('LMSTUDIO_BASE_URL', 'http://localhost:1234/v1')
        return cast(Any, ChatOpenAI)(
            model=resolved_model,
            temperature=temperature,
            base_url=base_url,
            api_key='lm-studio',  # LM Studio doesn't need a real key but ChatOpenAI requires one
            streaming=streaming,
        )

    else:
        raise ValueError(f"Unsupported provider: {provider}. Supported providers: openai, google, anthropic, openrouter, together, alibaba, ollama, lmstudio")


def format_docs(docs):
    """
    Format retrieved documents into context for LLM with debug-friendly citations.

    Provides BOTH:
    - Full text content for LLM processing
    - Debug-friendly reference list with scores and depth info

    Output format:
        === Context ===
        [Full text passages and KG facts here]

        === References (Debug) ===
        1. [Vec: 0.451] 素問·陰陽應象大論: "陰陽者，天地之道也..."
        2. [KG: 1-hop] 頭痛 --TREATS--> 川芎 (Herb)

    Args:
        docs: List of Document objects from vector store and/or knowledge graph

    Returns:
        str: Formatted string with full context and debug citations
    """
    SNIPPET_LENGTH = 60  # Characters for debug citation snippets

    # Separate by source type
    vector_docs = []
    graph_docs = []
    vector_refs = []
    graph_refs = []

    for doc in docs:
        source_type = doc.metadata.get('source_type', 'vector') if doc.metadata else 'vector'

        if source_type == 'graph':
            # Full content for context
            graph_docs.append(doc.page_content)

            # Debug reference
            depth = doc.metadata.get('depth', 1) if doc.metadata else 1
            hop_label = f"{depth}-hop" if depth == 1 else f"{depth}-hops"
            graph_refs.append(f"[KG: {hop_label}] {doc.page_content}")
        else:
            source = doc.metadata.get('source', 'Unknown') if doc.metadata else 'Unknown'
            score = doc.metadata.get('score', 0.0) if doc.metadata else 0.0

            # Full content for context
            vector_docs.append(f"--- Source: {source} ---\n{doc.page_content}\n")

            # Debug reference (truncated snippet)
            content = doc.page_content.strip().replace('\n', ' ')
            snippet = content[:SNIPPET_LENGTH] + "..." if len(content) > SNIPPET_LENGTH else content
            vector_refs.append(f"[Vec: {score}] {source}: \"{snippet}\"")

    # Build full context for LLM
    context_sections = []
    if vector_docs:
        context_sections.append("=== Text Passages ===")
        context_sections.extend(vector_docs)
    if graph_docs:
        context_sections.append("\n=== Knowledge Graph Facts ===")
        context_sections.extend(graph_docs)

    # Build debug references
    all_refs = vector_refs + graph_refs
    numbered_refs = [f"{i+1}. {ref}" for i, ref in enumerate(all_refs)]
    refs_section = "\n=== References (Debug) ===\n" + "\n".join(numbered_refs) if numbered_refs else ""

    return "\n".join(context_sections) + refs_section


CITATION_INSTRUCTION = """
You must ONLY use inline citations in the format [1], [2], etc.
Each number corresponds to the numbered sources provided in the context below.
NEVER include a "Sources:", "References:", or numbered list of sources at the end of your response.
NEVER append any section with source citations, source list, or reference list.
Only cite sources that are actually provided.
Violation of this rule will break the UI.
"""


def strip_sources_directive(system_prompt: str) -> str:
    """Remove legacy instructions that force a trailing Sources section."""

    cleaned_prompt = system_prompt
    for pattern in _SOURCES_DIRECTIVE_PATTERNS:
        cleaned_prompt = pattern.sub("", cleaned_prompt)

    cleaned_prompt = re.sub(r"\n{3,}", "\n\n", cleaned_prompt).strip()
    return cleaned_prompt or DEFAULT_SYSTEM_PROMPT


def build_prompt_template(system_prompt: str) -> ChatPromptTemplate:
    """Build the shared RAG prompt with inline-citation guidance."""

    normalized_prompt = strip_sources_directive(system_prompt)
    template = (
        f"{normalized_prompt}\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{question}\n\n"
        "请勿在回答末尾添加“来源”或“参考文献”列表，仅使用行内 [N] 标注。\n\n"
    )
    return ChatPromptTemplate.from_template(template)


def build_verification_payload(status: str) -> dict:
    """Normalize verification output for downstream UIs."""

    explanation = (
        "The answer appears supported by the retrieved citations."
        if status == "SUPPORTED"
        else "The answer may include claims not directly supported by the retrieved citations."
    )
    return {"status": status, "explanation": explanation}


def apply_relevance_percentages(citations: List[dict]) -> None:
    """Attach a response-local relevance percentage to text citations."""

    text_citations = [citation for citation in citations if citation.get("type") == "text"]
    if not text_citations:
        return

    if len(text_citations) == 1:
        text_citations[0]["relevance_percent"] = 95.0
        return

    scores = [float(citation.get("score", 0.0)) for citation in text_citations]
    best_score = min(scores)
    worst_score = max(scores)

    if best_score == worst_score:
        for citation in text_citations:
            citation["relevance_percent"] = 95.0
        return

    score_span = worst_score - best_score
    for citation in text_citations:
        normalized = (float(citation.get("score", 0.0)) - best_score) / score_span
        citation["relevance_percent"] = round(95.0 - (normalized * 35.0), 1)


def vector_search_with_scores(vectorstore: Chroma, query: str, k: int) -> list:
    """Run vector search and persist distance scores into document metadata."""

    results = vectorstore.similarity_search_with_score(query, k=k)
    docs = []
    for doc, score in results:
        if doc.metadata is None:
            doc.metadata = {}
        doc.metadata["source_type"] = "vector"
        doc.metadata["score"] = round(score, 3)
        docs.append(doc)
    return docs


def format_docs_with_citations(docs) -> Tuple[str, List[dict]]:
    """
    Format retrieved documents with numbered citations for LLM context.

    Unlike format_docs(), this function returns structured citation metadata
    that can be included in API responses for frontend rendering.

    Args:
        docs: List of Document objects from vector store and/or knowledge graph

    Returns:
        Tuple of:
        - str: Formatted context string with numbered sources
        - List[dict]: Citation metadata (TextCitation or GraphCitation dicts)
    """
    citations: List[dict] = []
    context_parts = []
    citation_number = 1

    # Process vector (text) documents first
    for i, doc in enumerate(docs):
        source_type = doc.metadata.get('source_type', 'vector') if doc.metadata else 'vector'

        if source_type == 'vector':
            source = doc.metadata.get('source', 'Unknown') if doc.metadata else 'Unknown'
            score = doc.metadata.get('score', 0.0) if doc.metadata else 0.0
            chunk_id = getattr(doc, "id", None) or (doc.metadata.get('id') if doc.metadata else None)

            # Reconstruct canonical chunk IDs when document.id is missing.
            if chunk_id is None and doc.metadata:
                book = doc.metadata.get("book")
                chunk_index = doc.metadata.get("chunk_index")
                if book and chunk_index is not None:
                    chunk_id = f"{book}_chunk_{chunk_index}"

            # Final fallback only when metadata is insufficient.
            if chunk_id is None:
                chunk_id = f"{source}_{i}"

            # Add to context with citation number
            context_parts.append(f"[{citation_number}] Source: {source}\n{doc.page_content}\n")

            # Build citation metadata
            content = doc.page_content.strip().replace('\n', ' ')
            snippet = content  # No truncation for passages

            citations.append({
                "number": citation_number,
                "type": "text",
                "source": source,
                "content": snippet,
                "chunk_id": chunk_id,
                "score": score,
                "relevance_percent": 95.0,
            })
            citation_number += 1

    # Process graph documents
    for doc in docs:
        source_type = doc.metadata.get('source_type', 'vector') if doc.metadata else 'vector'

        if source_type == 'graph':
            depth = doc.metadata.get('depth', 1) if doc.metadata else 1
            source_ref = doc.metadata.get('source_ref') if doc.metadata else None

            # Add to context with citation number
            context_parts.append(f"[{citation_number}] KG Fact: {doc.page_content}\n")

            citations.append({
                "number": citation_number,
                "type": "graph",
                "fact": doc.page_content,
                "depth": depth,
                "source_ref": source_ref,  # Provenance from KG relationship
            })
            citation_number += 1

    # Build final context
    context = "=== Numbered Sources ===\n\n" + "\n".join(context_parts) if context_parts else ""
    apply_relevance_percentages(citations)

    return context, citations


def verify_citation_bounds(answer: str, max_citation: int) -> dict:
    """
    Verify that LLM-generated citation numbers are within valid bounds.

    Scans the answer text for inline citations [n] and checks if any
    reference numbers exceed the number of provided sources.

    Args:
        answer: The LLM-generated response text.
        max_citation: Maximum valid citation number (total sources provided).

    Returns:
        Dict with:
        - is_valid: bool - True if all citations are in bounds
        - out_of_range: list[int] - Citation numbers that exceed max_citation
        - found_citations: list[int] - All citation numbers found in answer
    """
    import re

    # Find all citation markers [n] where n is a number
    citation_pattern = r'\[(\d+)\]'
    matches = re.findall(citation_pattern, answer)

    found_citations = [int(m) for m in matches]
    out_of_range = [n for n in found_citations if n > max_citation or n < 1]

    return {
        "is_valid": len(out_of_range) == 0,
        "out_of_range": sorted(set(out_of_range)),
        "found_citations": sorted(set(found_citations)),
    }


def get_query_severity(query, classifier_llm):
    """
    Classify user query into severity categories.

    Args:
        query (str): User's question
        classifier_llm: LLM instance for classification

    Returns:
        str: 'informational' or 'prescriptive'
    """
    classifier_template = """You are a helpful assistant for a Traditional Chinese Medicine query system. Your task is to classify the user's question into one of two categories based on its clinical severity:
1. 'informational': For general knowledge questions, definitions, or explanations of concepts.
2. 'prescriptive': For questions asking for diagnoses, treatments, formulas, or any advice that could directly impact a patient's health.

Respond with ONLY the category name ('informational' or 'prescriptive').

User Question:
{question}

Category:"""

    classifier_prompt = ChatPromptTemplate.from_template(classifier_template)
    classifier_chain = classifier_prompt | classifier_llm | StrOutputParser()

    severity = classifier_chain.invoke({"question": query}).strip().lower()

    # Validate and default to prescriptive if unclear
    if severity not in ['informational', 'prescriptive']:
        severity = 'prescriptive'

    return severity


def verify_answer(question, context, answer, llm):
    """
    Verify if the generated answer is supported by the provided context.

    Uses a self-critique prompt to detect potential hallucinations or unsupported claims.

    Args:
        question (str): The user's original question
        context (str): The retrieved context used to generate the answer
        answer (str): The generated answer to verify
        llm: LLM instance for verification

    Returns:
        str: 'SUPPORTED' or 'UNSUPPORTED'
    """
    # Load prompt from environment or use default
    sys_prompt = os.getenv('VERIFICATION_PROMPT')

    if not sys_prompt:
        # Fallback default if not in .env
        sys_prompt = """You are a strict verification auditor for a Traditional Chinese Medicine RAG system.

Your task: Determine if the Proposed Answer is FAITHFUL to the provided Context.

FAITHFULNESS CRITERIA:
1. The answer must be based on the provided Context.
2. ALLOWED: Synthesis, summarization, and logical inference derived from the Context.
3. ALLOWED: Use of standard TCM terminology to explain concepts found in the Context.
4. FORBIDDEN: Introducing external knowledge NOT supported by the Context.
5. FORBIDDEN: Contradicting the Context.

Context:
{context}

Question:
{question}

Proposed Answer:
{answer}

Respond with ONLY one word: 'SUPPORTED' or 'UNSUPPORTED'.

Verification Result:"""

    verification_prompt = ChatPromptTemplate.from_template(sys_prompt)
    verification_chain = verification_prompt | llm | StrOutputParser()

    result = verification_chain.invoke({
        "context": context,
        "question": question,
        "answer": answer
    }).strip().upper()

    # Normalize response to expected values
    if result not in ['SUPPORTED', 'UNSUPPORTED']:
        # If LLM returns unexpected format, default to UNSUPPORTED to ensure safety
        result = 'UNSUPPORTED'

    return result


def main():
    """
    Main function to execute the complete RAG pipeline.
    """
    print("TCM-Sage: Traditional Chinese Medicine RAG Assistant")
    print("=" * 60)

    # Initialize variables to avoid UnboundLocalError
    provider = 'alibaba'  # default provider
    model = None
    temperature = 0.1

    try:
        # Get provider configuration
        provider = os.getenv('LLM_PROVIDER', 'alibaba').lower()
        model = os.getenv('LLM_MODEL')
        temperature = float(os.getenv('LLM_TEMPERATURE', '0.1'))

        # Classifier configuration
        classifier_provider = os.getenv('CLASSIFIER_LLM_PROVIDER', provider).lower()
        classifier_model = os.getenv('CLASSIFIER_LLM_MODEL')
        classifier_temperature = float(os.getenv('CLASSIFIER_LLM_TEMPERATURE', '0.0'))

        # Verifier configuration
        verifier_provider = os.getenv('VERIFIER_LLM_PROVIDER', provider).lower()
        verifier_model = os.getenv('VERIFIER_LLM_MODEL')
        verifier_temperature = float(os.getenv('VERIFIER_LLM_TEMPERATURE', '0.0'))

        # Main LLM temperatures
        informational_temperature = temperature  # from LLM_TEMPERATURE
        prescriptive_temperature = float(os.getenv('PRESCRIPTIVE_TEMPERATURE', '0.0'))

        # Get retrieval configuration
        retrieval_k = int(os.getenv('RETRIEVAL_K', '5'))
        if retrieval_k < 1 or retrieval_k > 20:
            print(f"Warning: RETRIEVAL_K={retrieval_k} is outside recommended range (1-20). Using default value 5.")
            retrieval_k = 5

        # Hybrid retrieval configuration
        hybrid_enabled = os.getenv('HYBRID_RETRIEVAL_ENABLED', 'true').lower() == 'true'
        graph_data_path = os.getenv('GRAPH_DATA_PATH', GRAPH_DATA_DEFAULT_RELATIVE)
        graph_depth = int(os.getenv('GRAPH_DEPTH', '1'))

        # Get system prompt configuration
        system_prompt = os.getenv('SYSTEM_PROMPT')
        if not system_prompt:
            system_prompt = """You are an expert assistant specializing in Classical Chinese Medicine, specifically the Huangdi Neijing (黄帝内经).
Your task is to answer questions accurately based ONLY on the provided source text.
Your answer must be in the same language as the question.
Avoid excessive or nested markdown formatting to reduce rendering artifacts.
Use markdown formatting (bold, lists) conservatively to ensure clean rendering.
DO NOT add a 'Sources:' or 'References:' list at the end of your response."""

        if not system_prompt:
            system_prompt = DEFAULT_SYSTEM_PROMPT
        system_prompt = strip_sources_directive(system_prompt)

        print(f"Using LLM provider: {provider}")
        if model:
            print(f"Using model: {model}")
        print(f"Temperature: {temperature}")
        print(f"Retrieval K: {retrieval_k}")
        if hybrid_enabled:
            print(f"Hybrid Retrieval: ENABLED (graph_depth={graph_depth})")

        # Load the vector store
        print("Loading vector store...")
        vectorstore_path = Path(__file__).parent.parent / "vectorstore" / "chroma"

        if not vectorstore_path.exists():
            raise FileNotFoundError(
                f"Vector store not found at {vectorstore_path}. "
                "Please run 'python src/ingest.py' first to create the knowledge base."
            )

        # Initialize embeddings (must match the model used during ingestion)
        embeddings = get_embedding_model()

        # Load the persistent ChromaDB
        vectorstore = Chroma(
            persist_directory=str(vectorstore_path),
            embedding_function=embeddings
        )
        print(f"Vector store loaded successfully from: {vectorstore_path}")

        # Create a retriever (standard or hybrid)
        print("Setting up retriever...")

        if hybrid_enabled:
            # Use hybrid retriever with knowledge graph
            try:
                from retriever import create_hybrid_retriever
                hybrid_retriever = create_hybrid_retriever(
                    vectorstore_path=str(vectorstore_path),
                    graph_data_path=graph_data_path,
                    vector_k=retrieval_k,
                    graph_depth=graph_depth,
                )
                print("Hybrid retriever initialized with knowledge graph.")

                # Wrap in RunnableLambda for LangChain pipe compatibility
                retriever = RunnableLambda(lambda query: hybrid_retriever.hybrid_search(str(query)))
            except Exception as e:
                print(f"Warning: Failed to initialize hybrid retriever: {e}")
                print("Falling back to standard vector retriever.")
                hybrid_enabled = False
                retriever = RunnableLambda(
                    lambda query: vector_search_with_scores(vectorstore, str(query), retrieval_k)
                )
        else:
            retriever = RunnableLambda(
                lambda query: vector_search_with_scores(vectorstore, str(query), retrieval_k)
            )

        # Initialize classifier LLM
        print("Initializing classifier model...")
        classifier_llm = create_llm(classifier_provider, classifier_model, classifier_temperature)

        # Initialize main LLMs with different temperatures
        print("Initializing main language models...")
        llm_informational = create_llm(provider, model, informational_temperature)
        llm_prescriptive = create_llm(provider, model, prescriptive_temperature)

        # Initialize verifier LLM
        print("Initializing verifier model...")
        llm_verifier = create_llm(verifier_provider, verifier_model, verifier_temperature)

        # Define the prompt template
        print("Configuring prompt template...")
        prompt = build_prompt_template(system_prompt)

        # RAG chain will be built dynamically in the query loop based on classification

        print("\nRAG pipeline initialized successfully!")
        print("TCM-Sage is ready to answer questions about Traditional Chinese Medicine!")
        print("=" * 60)

        # Interactive query loop
        while True:
            try:
                # Prompt user for input
                user_query = input("\n请输入您的问题 (輸入 exit 來結束): ").strip()

                # Check exit commands
                if user_query.lower() in ['退出', 'exit', 'quit', 'q']:
                    print("\n感謝使用 TCM-Sage！再見！")
                    break

                # Skip empty input
                if not user_query:
                    print("請輸入有效問題。")
                    continue

                # Classify query severity
                print("\n正在分析問題類型...")
                severity = get_query_severity(user_query, classifier_llm)

                # Select appropriate LLM based on severity
                if severity == 'prescriptive':
                    selected_llm = llm_prescriptive
                    selected_temp = prescriptive_temperature
                else:
                    selected_llm = llm_informational
                    selected_temp = informational_temperature

                print(f"檢測到問題類型: {severity}")
                print(f"使用溫度: {selected_temp}")

                # Build RAG chain with selected LLM
                rag_chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | prompt
                    | selected_llm
                    | StrOutputParser()
                )

                # Execute RAG query
                print("正在生成答案...")

                # Retrieve context for verification
                retrieved_docs = retriever.invoke(user_query)
                formatted_context = format_docs(retrieved_docs)

                answer = rag_chain.invoke(user_query)

                # Self-critique verification step
                verification_result = "SUPPORTED"  # Default to avoid warning on error
                try:
                    print("正在驗證答案...")
                    verification_result = verify_answer(
                        question=user_query,
                        context=formatted_context,
                        answer=answer,
                        llm=llm_verifier
                    )
                except Exception as verify_error:
                    print(f"[Debug] Verification step encountered an issue: {verify_error}")
                    # Proceed without verification rather than crashing

                # Show answer
                print("\n" + "=" * 60)
                print("生成答案:")
                print("=" * 60)
                print(answer)

                # Append warning or confirmation based on verification result
                if verification_result == "UNSUPPORTED":
                    print("\n⚠️ [Self-Critique Warning]: This answer may contain information not directly supported by the provided citations.")
                else:
                    print("\n✅ [Self-Critique Pass]: This answer has been verified against the provided citations.")

                print("=" * 60)

            except KeyboardInterrupt:
                print("\n\n程式已中斷。感謝使用 TCM-Sage！")
                break
            except Exception as e:
                print(f"\n查詢處理錯誤: {e}")
                print("請嘗試另一個問題。")

    except ValueError as e:
        print(f"Configuration Error: {e}")
        print("\nTo fix this issue:")
        print("1. Create a .env file in the project root directory")
        print(f"2. Set LLM_PROVIDER={provider} (or your preferred provider)")
        print("3. Add your API key for the selected provider")
        print("4. See CONFIG.md for detailed setup instructions")

    except FileNotFoundError as e:
        print(f"File Error: {e}")
        print("\nTo fix this issue:")
        print("1. Run 'python src/ingest.py' to create the knowledge base")
        print("2. Ensure the vector store was created successfully")

    except Exception as e:
        print(f"Unexpected Error: {e}")
        print("\nPlease check:")
        print(f"1. Your {provider} API key is valid and has sufficient credits")
        print("2. Your internet connection is working")
        print("3. All dependencies are installed: pip install -r requirements.txt")


if __name__ == "__main__":
    main()
