# TCM-Sage Configuration Guide

## LLM Provider Setup

TCM-Sage supports multiple LLM providers to give you flexibility and cost options. You can switch between providers by simply changing a configuration flag.

### Step 1: Create your .env file

Copy the template below to create your `.env` file in the project root:

```bash
# LLM Provider Configuration
# Available providers: openai, google, anthropic, openrouter, together, alibaba
LLM_PROVIDER=alibaba

# API Keys - only fill in the key for your selected provider
OPENAI_API_KEY=your-openai-api-key-here
GOOGLE_API_KEY=your-google-ai-studio-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here
OPENROUTER_API_KEY=your-openrouter-api-key-here
TOGETHER_API_KEY=your-together-api-key-here
DASHSCOPE_API_KEY=your-alibaba-dashscope-api-key-here

# Model Configuration (optional - uses provider defaults if not specified)
# OpenAI: gpt-4o, gpt-4o-mini, gpt-3.5-turbo
# Google: gemini-1.5-pro, gemini-1.5-flash
# Anthropic: claude-3-5-sonnet-20241022, claude-3-haiku-20240307
# OpenRouter: openai/gpt-4o, anthropic/claude-3-5-sonnet, google/gemini-pro
# Together: meta-llama/Llama-3.1-8B-Instruct-Turbo, mistralai/Mixtral-8x7B-Instruct-v0.1
# Alibaba: qwen3-14b, qwen-plus, qwen-max, qwen-turbo
LLM_MODEL=

# Temperature setting (0.0 = deterministic, 1.0 = creative)
LLM_TEMPERATURE=0.1
```

### Step 2: Choose your provider and set API key

#### Alibaba Cloud Model Studio (Recommended - 1M free tokens for new users)
```bash
LLM_PROVIDER=alibaba
DASHSCOPE_API_KEY=your-alibaba-dashscope-api-key-here
```

#### Google AI Studio
```bash
LLM_PROVIDER=google
GOOGLE_API_KEY=your-google-ai-studio-api-key-here
```

#### OpenAI
```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=your-openai-api-key-here
```

#### Anthropic Claude
```bash
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your-anthropic-api-key-here
```

#### OpenRouter (Access to multiple providers)
```bash
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=your-openrouter-api-key-here
```

#### Together AI (Open source models)
```bash
LLM_PROVIDER=together
TOGETHER_API_KEY=your-together-api-key-here
```

### Step 3: Optional - Customize model and temperature

You can specify a particular model and temperature:

```bash
# Use a specific model
LLM_MODEL=qwen3-14b

# Adjust creativity (0.0 = factual, 1.0 = creative)
LLM_TEMPERATURE=0.1
```

## Getting API Keys

### Alibaba Cloud Model Studio (1M Free Tokens - Recommended)
1. Visit [Alibaba Cloud Model Studio](https://dashscope-intl.aliyuncs.com/)
2. Sign up for a new account
3. Go to the API Key management section
4. Create a new API key
5. Copy the key to your `.env` file

**Note**: This provider uses the native DashScope SDK, so it requires the `dashscope` and `langchain-community` packages.

**Available Models:**
- `qwen3-14b` (Recommended - Economic model)
- `qwen-plus` (Balanced performance and cost)
- `qwen-max` (Best performance)
- `qwen-turbo` (Fastest response)

### Google AI Studio (Free)
1. Visit [Google AI Studio](https://aistudio.google.com/)
2. Sign in with your Google account
3. Click "Get API Key" in the left sidebar
4. Create a new API key
5. Copy the key to your `.env` file

### OpenAI
1. Visit [OpenAI Platform](https://platform.openai.com/)
2. Sign up/login and go to API Keys section
3. Create a new secret key
4. Add payment method (required for API usage)

### Anthropic
1. Visit [Anthropic Console](https://console.anthropic.com/)
2. Sign up/login and go to API Keys
3. Create a new key
4. Add payment method

### OpenRouter
1. Visit [OpenRouter](https://openrouter.ai/)
2. Sign up and go to API Keys
3. Create a new key
4. Add credits

### Together AI
1. Visit [Together AI](https://together.ai/)
2. Sign up and go to API Keys
3. Create a new key
4. Add credits

## Troubleshooting

- Make sure your `.env` file is in the project root directory
- Ensure you've installed the required dependencies for your chosen provider
- Check that your API key has sufficient credits/quota
- Verify the provider name is spelled correctly (lowercase)
- For Alibaba Cloud, make sure you have activated the DashScope service

## Provider Comparison

| Provider | Free Tier | Cost | Models | Best For |
|----------|-----------|------|---------|----------|
| Alibaba Cloud | ✅ 1M tokens | Low | Qwen series | Chinese content, cost-effective |
| Google AI Studio | ✅ Limited | Medium | Gemini series | Free usage, good quality |
| OpenAI | ❌ No | High | GPT series | Premium quality |
| Anthropic | ❌ No | High | Claude series | Long context, reasoning |
| OpenRouter | ❌ No | Variable | Multiple providers | Access to many models |
| Together AI | ❌ No | Low | Open source models | Cost-effective, privacy |
