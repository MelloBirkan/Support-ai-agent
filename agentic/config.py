"""
OpenAI Configuration Module

This module provides centralized configuration for OpenAI API clients with support
for direct OpenAI API and Vocareum GenAI Gateway integration.

Vocareum Gateway:
When using Vocareum's GenAI Gateway, set OPENAI_API_BASE environment variable
to the gateway endpoint provided by Vocareum. The gateway will handle API key
management and model access policies.

Example for Vocareum:
    OPENAI_API_BASE=https://gateway.vocareum.com/v1
    OPENAI_API_KEY=<vocareum-provided-key>
"""

import os
from typing import Optional
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_openai_config() -> dict:
    """
    Get OpenAI configuration from environment variables.
    
    Returns:
        Dictionary with configuration:
        - api_key: OpenAI API key (or Vocareum key)
        - base_url: API base URL (for Vocareum gateway or None for direct OpenAI)
        - model: Model name (default: gpt-4o-mini)
        - embedding_model: Embedding model name (default: text-embedding-3-small)
    
    Environment Variables:
        OPENAI_API_KEY: Required. API key for OpenAI or Vocareum gateway
        OPENAI_API_BASE: Optional. Base URL for API (e.g., Vocareum gateway endpoint)
        OPENAI_MODEL: Optional. Model name (default: gpt-4o-mini)
        OPENAI_EMBEDDING_MODEL: Optional. Embedding model (default: text-embedding-3-small)
    """
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE")  # For Vocareum gateway
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is required. "
            "Please set it in your .env file or environment."
        )
    
    config = {
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
        "embedding_model": embedding_model
    }
    
    return config


def create_llm(
    model: Optional[str] = None,
    temperature: float = 0,
    **kwargs
) -> ChatOpenAI:
    """
    Create a ChatOpenAI instance with proper configuration.
    
    Supports both direct OpenAI API and Vocareum GenAI Gateway.
    When OPENAI_API_BASE is set, uses the gateway endpoint.
    
    Args:
        model: Model name (defaults to OPENAI_MODEL or gpt-4o-mini)
        temperature: Temperature setting (default: 0)
        **kwargs: Additional arguments to pass to ChatOpenAI
    
    Returns:
        Configured ChatOpenAI instance
        
    Example:
        >>> llm = create_llm()
        >>> llm = create_llm(model="gpt-4", temperature=0.7)
        
    Example with Vocareum:
        Set in .env:
            OPENAI_API_BASE=https://gateway.vocareum.com/v1
            OPENAI_API_KEY=<vocareum-key>
        >>> llm = create_llm()  # Uses Vocareum gateway automatically
    """
    config = get_openai_config()
    
    # Use provided model or default from config
    model_name = model or config["model"]
    
    # Build kwargs for ChatOpenAI
    llm_kwargs = {
        "model": model_name,
        "temperature": temperature,
        "api_key": config["api_key"],
        **kwargs
    }
    
    # Add base_url if using gateway (Vocareum)
    if config["base_url"]:
        llm_kwargs["base_url"] = config["base_url"]
    
    return ChatOpenAI(**llm_kwargs)


def create_embeddings(
    model: Optional[str] = None,
    **kwargs
) -> OpenAIEmbeddings:
    """
    Create an OpenAIEmbeddings instance with proper configuration.
    
    Supports both direct OpenAI API and Vocareum GenAI Gateway.
    When OPENAI_API_BASE is set, uses the gateway endpoint.
    
    Args:
        model: Embedding model name (defaults to OPENAI_EMBEDDING_MODEL or text-embedding-3-small)
        **kwargs: Additional arguments to pass to OpenAIEmbeddings
    
    Returns:
        Configured OpenAIEmbeddings instance
        
    Example:
        >>> embeddings = create_embeddings()
        >>> embeddings = create_embeddings(model="text-embedding-ada-002")
        
    Example with Vocareum:
        Set in .env:
            OPENAI_API_BASE=https://gateway.vocareum.com/v1
            OPENAI_API_KEY=<vocareum-key>
        >>> embeddings = create_embeddings()  # Uses Vocareum gateway automatically
    """
    config = get_openai_config()
    
    # Use provided model or default from config
    model_name = model or config["embedding_model"]
    
    # Build kwargs for OpenAIEmbeddings
    embeddings_kwargs = {
        "model": model_name,
        "api_key": config["api_key"],
        **kwargs
    }
    
    # Add base_url if using gateway (Vocareum)
    if config["base_url"]:
        embeddings_kwargs["base_url"] = config["base_url"]
    
    return OpenAIEmbeddings(**embeddings_kwargs)


def is_vocareum_gateway() -> bool:
    """
    Check if Vocareum Gateway is being used.
    
    Returns:
        True if OPENAI_API_BASE is set (indicating gateway usage)
    """
    return bool(os.getenv("OPENAI_API_BASE"))


__all__ = [
    "create_llm",
    "create_embeddings",
    "get_openai_config",
    "is_vocareum_gateway"
]



