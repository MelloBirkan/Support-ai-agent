# Vocareum GenAI Gateway Integration Guide

This guide explains how to use UDA-Hub with Vocareum's GenAI Gateway instead of direct OpenAI API access.

## Overview

Vocareum's GenAI Gateway provides centralized API key management, access control policies, and usage monitoring for OpenAI models. When using the gateway, UDA-Hub automatically routes all OpenAI API calls through Vocareum's endpoint.

## Setup Instructions

### 1. Configure Vocareum Gateway

1. Access the Vocareum Control Center
2. Navigate to **GenAI** in the sidebar
3. In the **Services** section, add a new service:
   - Select **OpenAI** as the service type
   - Provide your OpenAI API key
   - Set a name and budget for the service
4. Note the gateway endpoint URL provided (e.g., `https://gateway.vocareum.com/v1`)

### 2. Configure Environment Variables

Create or update your `.env` file with the Vocareum gateway configuration:

```bash
# Vocareum Gateway Configuration
OPENAI_API_BASE=https://gateway.vocareum.com/v1
OPENAI_API_KEY=<vocareum-provided-api-key>

# Optional: Model Configuration
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

**Important Notes:**
- `OPENAI_API_BASE`: The gateway endpoint URL from Vocareum (required for gateway mode)
- `OPENAI_API_KEY`: Use the API key provided by Vocareum, not your direct OpenAI key
- Model names should match what's allowed by your Vocareum policies

### 3. Verify Configuration

Check if Vocareum Gateway is being used:

```python
from agentic.config import is_vocareum_gateway

if is_vocareum_gateway():
    print("✅ Using Vocareum Gateway")
else:
    print("Using direct OpenAI API")
```

## How It Works

### Centralized Configuration

UDA-Hub uses a centralized configuration module (`agentic/config.py`) that:

1. Reads environment variables (`OPENAI_API_KEY`, `OPENAI_API_BASE`)
2. Automatically detects if Vocareum Gateway is configured (when `OPENAI_API_BASE` is set)
3. Creates LLM and embedding clients with appropriate base URLs

### Automatic Detection

The system automatically uses Vocareum Gateway when `OPENAI_API_BASE` is set:

```python
# Direct OpenAI API (default)
OPENAI_API_KEY=sk-...

# Vocareum Gateway (when OPENAI_API_BASE is set)
OPENAI_API_BASE=https://gateway.vocareum.com/v1
OPENAI_API_KEY=<vocareum-key>
```

### Components Using Gateway

All components automatically use the gateway when configured:

- **LLM Models** (`ChatOpenAI`): Used by all agents (Classifier, Resolver, Tool Agent, Escalation)
- **Embeddings** (`OpenAIEmbeddings`): Used by RAG system for knowledge base retrieval

## Usage Examples

### Example 1: Basic Setup

```python
# .env file
OPENAI_API_BASE=https://gateway.vocareum.com/v1
OPENAI_API_KEY=vk-vocareum-key-here
OPENAI_MODEL=gpt-4o-mini

# Python code (same as before)
from agentic.workflow import orchestrator
from langchain_core.messages import HumanMessage

state = {
    "messages": [HumanMessage(content="I can't login")],
    "ticket_metadata": {
        "ticket_id": "TKT-001",
        "account_id": "cultpass",
        "user_email": "user@example.com"
    }
}

# Works automatically with Vocareum Gateway
result = orchestrator.invoke(state, {"configurable": {"thread_id": "TKT-001"}})
```

### Example 2: Custom Model Configuration

```python
from agentic.config import create_llm, create_embeddings

# Use Vocareum Gateway with custom model
llm = create_llm(model="gpt-4", temperature=0.7)
embeddings = create_embeddings(model="text-embedding-ada-002")
```

### Example 3: Programmatic Detection

```python
from agentic.config import get_openai_config, is_vocareum_gateway

config = get_openai_config()
if is_vocareum_gateway():
    print(f"Gateway endpoint: {config['base_url']}")
    print(f"Using Vocareum key: {config['api_key'][:10]}...")
```

## Vocareum-Specific Features

### Access Policies

Vocareum allows you to:
- **Restrict models**: Control which OpenAI models are available
- **Set budgets**: Limit API usage per service or user
- **Monitor usage**: Track API calls and costs through Vocareum dashboard

### User Keys

In Vocareum environments (like AI Notebook), user-specific keys are automatically provided:
- `OPENAI_API_KEY`: Provided as environment variable
- `OPENAI_API_BASE`: Provided as environment variable

### Model Restrictions

Your Vocareum policies may restrict certain models. If you encounter errors:

1. Check available models in Vocareum Control Center → GenAI → Policies
2. Update `OPENAI_MODEL` in `.env` to match allowed models
3. Default fallback: `gpt-4o-mini` (usually available)

## Troubleshooting

### Issue: "Invalid API key"

**Solution**: Ensure you're using the Vocareum-provided API key, not your direct OpenAI key.

### Issue: "Model not found"

**Solution**: Check which models are allowed in your Vocareum policies and update `OPENAI_MODEL` accordingly.

### Issue: "Connection refused"

**Solution**: Verify `OPENAI_API_BASE` is correctly set to the Vocareum gateway endpoint.

### Issue: "Rate limit exceeded"

**Solution**: Check your Vocareum budget settings. Adjust limits in Control Center.

## Switching Between Modes

### Direct OpenAI API

```bash
# .env
OPENAI_API_KEY=sk-your-direct-openai-key
# Don't set OPENAI_API_BASE
```

### Vocareum Gateway

```bash
# .env
OPENAI_API_BASE=https://gateway.vocareum.com/v1
OPENAI_API_KEY=vk-vocareum-key
```

**Note**: Simply adding or removing `OPENAI_API_BASE` switches between modes. No code changes needed.

## Benefits of Vocareum Gateway

1. **Centralized Management**: Single place to manage API keys
2. **Usage Monitoring**: Track API calls and costs
3. **Access Control**: Restrict models and users via policies
4. **Budget Management**: Set spending limits per service/user
5. **Security**: Keys managed by Vocareum, not exposed in code

## Technical Details

### Implementation

The integration is implemented in `agentic/config.py`:

- `create_llm()`: Creates `ChatOpenAI` instances with gateway support
- `create_embeddings()`: Creates `OpenAIEmbeddings` instances with gateway support
- `get_openai_config()`: Reads and validates configuration
- `is_vocareum_gateway()`: Checks if gateway is configured

### API Compatibility

Vocareum Gateway is compatible with OpenAI's API format, so no code changes are required in agents or tools. The gateway acts as a transparent proxy.

## Additional Resources

- [Vocareum GenAI Gateway Documentation](https://help.vocareum.com/en/articles/9251522-genai-gateway)
- [Vocareum REST API Documentation](https://help.vocareum.com/en/articles/3658972-vocareum-rest-api)
- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)

## Support

For issues with:
- **Vocareum Gateway**: Contact Vocareum support or check documentation
- **UDA-Hub Integration**: Check this documentation or project README



