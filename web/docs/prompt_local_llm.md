# Prompt: Add Local LLM Support (Ollama + LM Studio)

Send to: **new backend agent**

---

## Context

TCM-Sage is a RAG system for Traditional Chinese Medicine. It supports multiple LLM providers via `create_llm()` in `src/main.py` (lines 74-200). The function uses provider-specific LangChain wrappers.

The user wants local LLM support for:
1. **Ollama** (default: `http://localhost:11434`)
2. **LM Studio** (default: `http://localhost:1234`)

Both expose OpenAI-compatible `/v1/chat/completions` endpoints, so they can reuse `ChatOpenAI` from `langchain-openai` (already in `requirements.txt`) with a custom `base_url`. **No new dependencies needed.**

Branch: `feature/premium-ui` (already checked out).

## What to implement

### 1. Add `ollama` and `lmstudio` providers to `create_llm()` in `src/main.py`

Add two new `elif` blocks before the `else` clause (before line 199):

```python
elif provider == 'ollama':
    if ChatOpenAI is None:
        raise ValueError("Ollama provider requires 'langchain-openai' package. Install with: pip install langchain-openai")
    base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434/v1')
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        base_url=base_url,
        api_key='ollama',  # Ollama doesn't need a real key but ChatOpenAI requires one
        streaming=streaming,
    )

elif provider == 'lmstudio':
    if ChatOpenAI is None:
        raise ValueError("LM Studio provider requires 'langchain-openai' package. Install with: pip install langchain-openai")
    base_url = os.getenv('LMSTUDIO_BASE_URL', 'http://localhost:1234/v1')
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        base_url=base_url,
        api_key='lm-studio',  # LM Studio doesn't need a real key but ChatOpenAI requires one
        streaming=streaming,
    )
```

### 2. Add default models to the `default_models` dict (line 98-106)

```python
default_models = {
    'openai': 'gpt-5-2',
    'google': 'gemini-3-pro',
    'anthropic': 'claude-4-5-sonnet-20241022',
    'openrouter': 'openai/gpt-5-2',
    'together': 'meta-llama/Llama-3.1-8B-Instruct-Turbo',
    'alibaba': 'qwen3-max',
    'ollama': 'qwen3:8b',          # Popular local model with CJK support
    'lmstudio': 'qwen3-8b',       # LM Studio uses simple model names
}
```

### 3. Update the docstring (line 79) and error message (line 200)

- Docstring: Add `'ollama'` and `'lmstudio'` to the provider list
- Error message: Update the supported providers string

### 4. Update the streaming TODO note (lines 90-94)

Both Ollama and LM Studio support streaming natively via ChatOpenAI, so add a note that they are streaming-compatible.

### 5. Update `.env.example` — add local LLM section

Add this near the API keys section:

```bash
# Local LLM Configuration (no API key needed)
# OLLAMA_BASE_URL=http://localhost:11434/v1     # Default Ollama endpoint
# LMSTUDIO_BASE_URL=http://localhost:1234/v1    # Default LM Studio endpoint
```

### 6. Update `docs/CONFIG.md` — add provider docs

Add two new provider sections after the "Together AI" section (after line 158):

```markdown
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
```

Also update the Cost Optimization section to mention local LLMs as the free option:

```markdown
## Cost Optimization

For cost-effective development and testing:

1. **Use Ollama or LM Studio** for free local inference (requires GPU recommended)
2. **Start with Alibaba Cloud Model Studio** (1M free tokens for cloud)
3. **Use smaller models** when testing (e.g., `gpt-3.5-turbo` instead of `gpt-4o`)
4. **Set lower temperature** to reduce response variability
5. **Monitor usage** through provider dashboards
```

### 7. Also update `docs/CONFIG.md` troubleshooting

Add to the Provider-Specific Notes section:

```markdown
- **Ollama**: Ensure Ollama is running (`ollama serve`). Check with `curl http://localhost:11434/v1/models`
- **LM Studio**: Ensure the local server is started and a model is loaded. Check the LM Studio server tab for the port
```

## Key design decisions

- Both providers reuse `ChatOpenAI` — no new dependencies, streaming works out of the box
- `api_key` is set to a dummy string because `ChatOpenAI` validates it's non-empty, but local servers ignore it
- Base URLs are configurable via env vars for non-default ports
- Default models use `qwen3` variants because this is a TCM (Chinese medicine) app and Qwen has excellent CJK support

## Files to modify

- `src/main.py` — add two provider branches + update defaults/docs
- `.env.example` — add local LLM config section
- `docs/CONFIG.md` — add provider documentation

## Verification

After the changes:

1. Start Ollama: `ollama serve` then `ollama pull qwen3:8b`
2. Set `.env`:
   ```
   LLM_PROVIDER=ollama
   LLM_MODEL=qwen3:8b
   CLASSIFIER_LLM_PROVIDER=ollama
   CLASSIFIER_LLM_MODEL=qwen3:8b
   VERIFIER_LLM_PROVIDER=ollama
   VERIFIER_LLM_MODEL=qwen3:8b
   ```
3. Restart uvicorn and test a query via the web UI or CLI
4. Verify streaming works (text should appear incrementally)

## Commit message

```
feat(llm): add Ollama and LM Studio local LLM providers

Both use OpenAI-compatible API via ChatOpenAI with custom base_url.
No new dependencies required. Enables free local inference for
development and as a cost-effective option for end users.
```
