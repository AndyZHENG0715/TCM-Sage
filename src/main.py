"""
TCM-Sage Main RAG Application

This script implements the core Retrieval-Augmented Generation (RAG) pipeline for the TCM-Sage system.
It loads the vector store, creates a retrieval chain, and orchestrates the process of answering
user queries with citations from the Huangdi Neijing.

The system uses a modular RAG architecture that combines semantic vector search with
evidence-backed answer generation using OpenAI's GPT-4o model.
"""

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pathlib import Path
import os
import sys
from dotenv import load_dotenv

# Fix Unicode encoding issues on Windows
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

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

try:
    import dashscope
    from langchain_community.llms import Tongyi
except ImportError:
    dashscope = None
    Tongyi = None


def create_llm(provider, model=None, temperature=0.1):
    """
    Create an LLM instance based on the provider configuration.

    Args:
        provider (str): The LLM provider ('openai', 'google', 'anthropic', 'openrouter', 'together')
        model (str, optional): Specific model to use
        temperature (float): Temperature for generation

    Returns:
        LLM instance

    Raises:
        ValueError: If provider is not supported or required dependencies are missing
    """
    provider = provider.lower()

    # Default models for each provider
    default_models = {
        'openai': 'gpt-4o',
        'google': 'gemini-2.5-pro',
        'anthropic': 'claude-3-5-sonnet-20241022',
        'openrouter': 'openai/gpt-4o',
        'together': 'meta-llama/Llama-3.1-8B-Instruct-Turbo',
        'alibaba': 'qwen3-14b'
    }

    # Use default model if none specified
    if not model:
        model = default_models.get(provider)

    if provider == 'openai':
        if ChatOpenAI is None:
            raise ValueError("OpenAI provider requires 'langchain-openai' package. Install with: pip install langchain-openai")
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key or api_key == 'your-openai-api-key-here':
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")
        return ChatOpenAI(
            model_name=model,
            temperature=temperature,
            api_key=api_key
        )

    elif provider == 'google':
        if ChatGoogleGenerativeAI is None:
            raise ValueError("Google provider requires 'langchain-google-genai' package. Install with: pip install langchain-google-genai")
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key or api_key == 'your-google-ai-studio-api-key-here':
            raise ValueError("Google API key not found. Please set GOOGLE_API_KEY in your .env file.")
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=api_key
        )

    elif provider == 'anthropic':
        if ChatAnthropic is None:
            raise ValueError("Anthropic provider requires 'langchain-anthropic' package. Install with: pip install langchain-anthropic")
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key or api_key == 'your-anthropic-api-key-here':
            raise ValueError("Anthropic API key not found. Please set ANTHROPIC_API_KEY in your .env file.")
        return ChatAnthropic(
            model=model,
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
            model_name=model,
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
            model=model,
            temperature=temperature,
            together_api_key=api_key
        )

    elif provider == 'alibaba':
        if Tongyi is None or dashscope is None:
            raise ValueError("Alibaba provider requires 'dashscope' and 'langchain-community' packages. Install with: pip install dashscope langchain-community")
        api_key = os.getenv('DASHSCOPE_API_KEY')
        if not api_key or api_key == 'your-alibaba-api-key-here':
            raise ValueError("Alibaba API key not found. Please set DASHSCOPE_API_KEY in your .env file.")

        # Set the base URL for Singapore region (international)
        dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'

        # Set the API key globally for DashScope
        dashscope.api_key = api_key

        return Tongyi(
            model_name=model,
            temperature=temperature
        )

    else:
        raise ValueError(f"Unsupported provider: {provider}. Supported providers: openai, google, anthropic, openrouter, together, alibaba")


def format_docs(docs):
    """
    Format retrieved documents into a single string for context.

    Args:
        docs: List of Document objects from the vector store

    Returns:
        str: Formatted string containing all documents with source metadata
    """
    formatted_docs = []
    for doc in docs:
        # Extract source metadata (chapter name)
        source = doc.metadata.get('source', 'Unknown Source') if doc.metadata else 'Unknown Source'
        formatted_docs.append(f"--- Source: {source} ---\n{doc.page_content}\n")

    return "\n".join(formatted_docs)


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
        # Load environment variables
        print("Loading environment configuration...")
        load_dotenv()

        # Get provider configuration
        provider = os.getenv('LLM_PROVIDER', 'alibaba').lower()
        model = os.getenv('LLM_MODEL')
        temperature = float(os.getenv('LLM_TEMPERATURE', '0.1'))

        print(f"Using LLM provider: {provider}")
        if model:
            print(f"Using model: {model}")
        print(f"Temperature: {temperature}")

        # Load the vector store
        print("Loading vector store...")
        vectorstore_path = Path(__file__).parent.parent / "vectorstore" / "chroma"

        if not vectorstore_path.exists():
            raise FileNotFoundError(
                f"Vector store not found at {vectorstore_path}. "
                "Please run 'python src/ingest.py' first to create the knowledge base."
            )

        # Initialize embeddings (must match the model used during ingestion)
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

        # Load the persistent ChromaDB
        vectorstore = Chroma(
            persist_directory=str(vectorstore_path),
            embedding_function=embeddings
        )
        print(f"Vector store loaded successfully from: {vectorstore_path}")

        # Create a retriever
        print("Setting up retriever...")
        retriever = vectorstore.as_retriever(
            k=5  # k=5: Number of most relevant chunks to retrieve. Increase for broader context, decrease for faster responses.
        )

        # Initialize the language model
        print("Initializing language model...")
        llm = create_llm(provider, model, temperature)

        # Define the prompt template
        print("Configuring prompt template...")
        template = """
You are an expert assistant specializing in Classical Chinese Medicine, specifically the Huangdi Neijing (黄帝内经).
Your task is to answer questions accurately based ONLY on the provided source text.
Your answer must be in the same language as the question.
After providing the answer, cite the source chapter for the information you provide in a "Sources:" section.

Context:
{context}

Question:
{question}

Answer:
"""
        prompt = ChatPromptTemplate.from_template(template)

        # Construct the RAG chain using LangChain Expression Language (LCEL)
        print("Building RAG chain...")
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        print("\nRAG pipeline initialized successfully!")
        print("TCM-Sage is ready to answer questions about Traditional Chinese Medicine!")
        print("=" * 60)

        # 交互式查询循环
        while True:
            try:
                # 提示用户输入问题
                user_query = input("\n请输入您的问题 (輸入 exit 來結束): ").strip()

                # 检查退出命令
                if user_query.lower() in ['退出', 'exit', 'quit', 'q']:
                    print("\n感謝使用 TCM-Sage！再見！")
                    break

                # 跳过空输入
                if not user_query:
                    print("請輸入有效問題。")
                    continue

                # 执行RAG查询
                print("\n正在生成答案...")
                answer = rag_chain.invoke(user_query)

                # 显示答案
                print("\n" + "=" * 60)
                print("生成答案:")
                print("=" * 60)
                print(answer)
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
