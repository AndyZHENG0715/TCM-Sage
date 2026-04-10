# TCM-Sage: Configuration Guide

This document provides detailed setup instructions for configuring the TCM-Sage RAG system with different LLM providers.

## LLM Provider Configuration

The system supports multiple LLM providers that can be easily switched using environment variables. Create a `.env` file in the project root directory with the following configuration:

### Environment Variables

```bash
# LLM Provider Configuration
LLM_PROVIDER=alibaba          # Provider: alibaba, openai, google, anthropic, openrouter, together
LLM_MODEL=                    # Optional: Override default model for the provider
LLM_TEMPERATURE=0.1           # Temperature for informational queries (can increase for creativity: 0.3-0.7)
PRESCRIPTIVE_TEMPERATURE=0.0  # Temperature for prescriptive/diagnostic queries (keep at 0.0 for accuracy)
                              # Only increase if you need more creative medical suggestions (not recommended)

# Classifier LLM Configuration (for query routing)
CLASSIFIER_LLM_PROVIDER=alibaba    # Lightweight model provider for fast query classification
CLASSIFIER_LLM_MODEL=qwen3-0.6b    # Small, fast model (recommended: qwen3-0.6b, gemini-2.5-flash)
CLASSIFIER_LLM_TEMPERATURE=0.0     # Keep at 0.0 for consistent classification

# Retrieval Configuration
RETRIEVAL_K=5                 # Number of document chunks to retrieve (3-10 recommended)
                              # Higher values: More comprehensive answers, slower responses
                              # Lower values: Faster responses, potentially less context

# System Prompt Configuration
SYSTEM_PROMPT="You are an expert assistant specializing in Traditional Chinese Medicine, specifically the Huangdi Neijing (黄帝内经). Your task is to answer questions accurately based ONLY on the provided source text. If a direct definition is not present, explain the concept using the information available. Your answer must be in the same language as the question. After providing the answer, cite the source chapter for the information you provide in a \"Sources:\" section."
SYSTEM_PROMPT_OVERRIDE=...       # Overrides the built-in default prompt in src/main.py
VERIFICATION_PROMPT=...          # Optional custom verifier prompt for answer faithfulness checks

# Output Format Configuration (Future UI Support)
OUTPUT_FORMAT=detailed        # detailed, concise, academic
CITATION_STYLE=chapter         # chapter, page, section (display format only, no performance impact)

# Provider-specific API Keys (only set the one you're using)
DASHSCOPE_API_KEY=your-alibaba-dashscope-api-key-here
OPENAI_API_KEY=your-openai-api-key-here
GOOGLE_API_KEY=your-google-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here
OPENROUTER_API_KEY=your-openrouter-api-key-here
TOGETHER_API_KEY=your-together-api-key-here
```

## Hybrid Retrieval (Knowledge Graph)

TCM-Sage supports hybrid retrieval that combines vector search with a knowledge graph for improved TCM terminology resolution.

### Configuration

```bash
# Hybrid Retrieval Configuration
HYBRID_RETRIEVAL_ENABLED=true     # Hybrid vector+graph retrieval (default: true)
GRAPH_DATA_PATH=data/graph/symmap/symmap_entities.json  # SymMap-shaped KG JSON (see data/graph/symmap/README.md)
GRAPH_DEPTH=1                     # Max traversal depth for graph search (1-2 recommended)
```

### How It Works

1. **Vector Search**: Retrieves semantically similar text passages from the Huangdi Neijing
2. **Graph Search**: Traverses the knowledge graph to find related TCM entities (symptoms, herbs, formulas)
3. **Ensemble Context**: Both results are combined as distinct sections in the LLM prompt

### Knowledge Graph Schema

- **Entities**: `Symptom`, `Herb`, `Formula`
- **Relationships**: `TREATS`, `CONTAINS`, `ASSOCIATED_WITH`

Example: Query "頭痛" returns:

- Vector passages mentioning headaches
- Graph facts: "川芎 TREATS 頭痛", "天麻 TREATS 頭痛", "川芎茶調散 TREATS 頭痛"

### Extending the Graph

Edit `data/graph/entities.json` to add new entities and relationships:

## Query Classification and Routing

TCM-Sage now includes an intelligent query classification system that automatically determines the clinical severity of user questions and adjusts the response generation accordingly.

### How It Works

1. **Query Classification**: A lightweight classifier model analyzes each user query to determine if it's:
   - **Informational**: General knowledge questions, definitions, or explanations (e.g., "陰陽是什麼？")
   - **Prescriptive**: Questions asking for diagnoses, treatments, formulas, or medical advice (e.g., "頭痛應該用什麼方劑？")

2. **Dynamic Temperature Adjustment**: Based on the classification:
   - **Informational queries**: Use `LLM_TEMPERATURE` (default 0.1, can be increased for creativity)
   - **Prescriptive queries**: Use `PRESCRIPTIVE_TEMPERATURE` (default 0.0 for maximum accuracy)

### Configuration Options

- **`CLASSIFIER_LLM_PROVIDER`**: Provider for the classification model (recommended: same as main provider)
- **`CLASSIFIER_LLM_MODEL`**: Lightweight model for fast classification (recommended: `qwen3-0.6b`, `gemini-2.5-flash`)
- **`CLASSIFIER_LLM_TEMPERATURE`**: Keep at 0.0 for consistent classification
- **`VERIFIER_LLM_PROVIDER`**: Provider for the verification model (defaults to the main provider)
- **`VERIFIER_LLM_MODEL`**: Model used for the answer-faithfulness check
- **`VERIFIER_LLM_TEMPERATURE`**: Keep at 0.0 for deterministic verification
- **`PRESCRIPTIVE_TEMPERATURE`**: Temperature for medical/prescriptive queries (keep at 0.0 unless necessary)

### Best Practices

- **For informational queries**: You can increase `LLM_TEMPERATURE` to 0.3-0.7 for more creative explanations
- **For prescriptive queries**: Always keep `PRESCRIPTIVE_TEMPERATURE` at 0.0 to ensure medical accuracy
- **Classifier model**: Use a small, fast model to minimize latency and cost

## Arena Configuration

### ARENA_MODELS Parameter

`ARENA_MODELS` overrides the arena tier-to-model mapping as a JSON string.

```bash
ARENA_MODELS='{"flash":"qwen-turbo","plus":"qwen-plus","max":"qwen-max"}'
ARENA_STREAM_TIMEOUT_SECONDS=60  # Optional timeout for arena SSE responses
```

## Prototype UI (Optional)

- The Streamlit demo (`streamlit run src/ui_app.py`) provides a quick way to showcase TCM-Sage to stakeholders without disrupting the CLI workflow.
- It reuses all configuration options documented here, so ensure your `.env` is set up before launching.
- The UI is meant for discovery and may be deprecated later; the CLI remains the primary interface.

## Web Frontend / API Integration

```bash
# FastAPI CORS + Next.js backend proxy target
ALLOWED_ORIGINS=http://localhost:3000,https://your-production-domain.com  # Comma-separated CORS whitelist for the API
BACKEND_URL=http://127.0.0.1:8000                                          # Server-side backend URL for the Next.js proxy route
NEXT_PUBLIC_BACKEND_URL=http://127.0.0.1:8000                              # Client-visible fallback backend URL
FEEDBACK_FORM_URL=https://forms.gle/your-feedback-form                     # Optional feedback link used by the UI
```

## Supported Providers

### 1. Alibaba Cloud Model Studio (Recommended)

**Default Provider** - Cost-effective with 1M free tokens for new users.

- **Provider ID**: `alibaba`
- **Default Model**: `qwen3-14b`
- **API Key**: `DASHSCOPE_API_KEY`
- **Setup**:
  1. Sign up at [Alibaba Cloud Model Studio](https://dashscope.aliyuncs.com/)
  2. Create an API key in the DashScope console
  3. Set `LLM_PROVIDER=alibaba` and your `DASHSCOPE_API_KEY`

### 2. OpenAI

- **Provider ID**: `openai`
- **Default Model**: `gpt-4o`
- **API Key**: `OPENAI_API_KEY`
- **Setup**: Get API key from [OpenAI Platform](https://platform.openai.com/)

### 3. Google AI Studio

- **Provider ID**: `google`
- **Default Model**: `gemini-2.5-pro`
- **API Key**: `GOOGLE_API_KEY`
- **Setup**: Get API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

### 4. Anthropic (Claude)

- **Provider ID**: `anthropic`
- **Default Model**: `claude-3-5-sonnet-20241022`
- **API Key**: `ANTHROPIC_API_KEY`
- **Setup**: Get API key from [Anthropic Console](https://console.anthropic.com/)

### 5. OpenRouter

- **Provider ID**: `openrouter`
- **Default Model**: `openai/gpt-4o`
- **API Key**: `OPENROUTER_API_KEY`
- **Setup**: Get API key from [OpenRouter](https://openrouter.ai/)

### 6. Together AI

- **Provider ID**: `together`
- **Default Model**: `meta-llama/Llama-3.1-8B-Instruct-Turbo`
- **API Key**: `TOGETHER_API_KEY`
- **Setup**: Get API key from [Together AI](https://together.ai/)

### 7. Ollama (Local)

- **Provider ID**: `ollama`
- **Default Model**: `qwen3:8b`
- **API Key**: Not required
- **Base URL**: `OLLAMA_BASE_URL` (default: `http://localhost:11434/v1`)
- **Setup**:
  1. Install Ollama from [ollama.ai](https://ollama.ai/)
  2. Pull a model: `ollama pull qwen3:8b`
  3. Set `LLM_PROVIDER=ollama` in `.env`
  4. Optionally set `LLM_MODEL` to any model you've pulled

### 8. LM Studio (Local)

- **Provider ID**: `lmstudio`
- **Default Model**: `qwen3-8b`
- **API Key**: Not required
- **Base URL**: `LMSTUDIO_BASE_URL` (default: `http://localhost:1234/v1`)
- **Setup**:
  1. Download LM Studio from [lmstudio.ai](https://lmstudio.ai/)
  2. Load a model in the LM Studio UI
  3. Start the local server (LM Studio → Local Server tab)
  4. Set `LLM_PROVIDER=lmstudio` in `.env`
  5. Set `LLM_MODEL` to match the loaded model name

## Model Selection

### Default Models by Provider

Each provider has a recommended default model that balances performance and cost:

- **Alibaba Cloud**: `qwen3-14b` - Economic model with good Chinese language support
- **OpenAI**: `gpt-4o` - High-performance model
- **Google**: `gemini-2.5-pro` - Advanced reasoning capabilities
- **Anthropic**: `claude-3-5-sonnet-20241022` - Strong analytical capabilities
- **OpenRouter**: `openai/gpt-4o` - Access to OpenAI models via OpenRouter
- **Together AI**: `meta-llama/Llama-3.1-8B-Instruct-Turbo` - Open-source model

### Override Default Model

To use a different model, set the `LLM_MODEL` environment variable:

```bash
LLM_PROVIDER=openai
LLM_MODEL=gpt-3.5-turbo  # Override default gpt-4o
```

## Retrieval Configuration

### RETRIEVAL_K Parameter

The `RETRIEVAL_K` parameter controls how many document chunks are retrieved for each query:

- **3-5**: Fast responses, good for simple questions (recommended for most use cases)
- **5-8**: Balanced performance and comprehensiveness (default: 5)
- **8-15**: More comprehensive answers, slower responses (good for complex queries)
- **15+**: Maximum context, slowest responses (use sparingly)

**Performance Impact**: Higher values increase response time and API costs but provide more comprehensive answers.

## System Prompt Configuration

### SYSTEM_PROMPT Parameter

The `SYSTEM_PROMPT` parameter defines how the AI assistant behaves and responds. You can customize:

- **Language**: Modify to support different languages
- **Behavior**: Change how the assistant approaches questions
- **Citation Style**: Adjust how sources are referenced
- **Response Format**: Customize the structure of answers

**Performance Impact**: Longer prompts consume more tokens but provide more precise control over AI behavior.

## Temperature Configuration

The temperature parameter controls the randomness of model responses:

- **0.0**: Most deterministic, factual responses (recommended for TCM)
- **0.1**: Slightly creative but mostly factual (default)
- **0.7**: Balanced creativity and accuracy
- **1.0**: Most creative responses

## Quick Start

1. **Copy the example configuration**:

   ```bash
   cp .env.example .env
   ```

   The `.env.example` file includes all available configuration options with detailed comments.

2. **Edit `.env`** with your preferred provider and API key:

   ```bash
   LLM_PROVIDER=alibaba
   DASHSCOPE_API_KEY=your-actual-api-key-here
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the system**:

   ```bash
   python src/main.py
   ```

## Troubleshooting

### Common Issues

1. **"Configuration Error: API key not found"**
   - Ensure your `.env` file is in the project root
   - Verify the API key variable name matches your provider
   - Check that the API key is valid and has sufficient credits

2. **"Unsupported provider"**
   - Verify `LLM_PROVIDER` is set to one of: `alibaba`, `openai`, `google`, `anthropic`, `openrouter`, `together`
   - Check for typos in the provider name

3. **Import errors**
   - Run `pip install -r requirements.txt` to install all required packages
   - Some providers require specific packages that are automatically installed

4. **API connection errors**
   - Verify your API key is valid
   - Check your internet connection
   - Ensure you have sufficient API credits/quota

### Provider-Specific Notes

- **Alibaba Cloud**: Uses the Singapore region endpoint for optimal performance
- **OpenAI**: Requires a paid account with sufficient credits
- **Google AI Studio**: Free tier available with usage limits
- **Anthropic**: Free tier available with usage limits
- **OpenRouter**: Pay-per-use pricing for various models
- **Together AI**: Competitive pricing for open-source models
- **Ollama**: Ensure Ollama is running (`ollama serve`). Check with `curl http://localhost:11434/v1/models`
- **LM Studio**: Ensure the local server is started and a model is loaded. Check the LM Studio server tab for the port

## Cost Optimization

For cost-effective development and testing:

1. **Use Ollama or LM Studio** for free local inference (requires GPU recommended)
2. **Start with Alibaba Cloud Model Studio** (1M free tokens for cloud)
3. **Use smaller models** when testing (e.g., `gpt-3.5-turbo` instead of `gpt-4o`)
4. **Set lower temperature** to reduce response variability
5. **Monitor usage** through provider dashboards

## Security Notes

- Never commit your `.env` file to version control
- Use environment variables in production deployments
- Rotate API keys regularly
- Monitor API usage for unexpected charges
