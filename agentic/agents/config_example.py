"""
Example usage of OpenAI configuration module with Vocareum Gateway support.

This file demonstrates how to use the centralized configuration for both
direct OpenAI API and Vocareum GenAI Gateway.
"""

from agentic.config import (
    create_llm,
    create_embeddings,
    get_openai_config,
    is_vocareum_gateway
)

# Example 1: Create LLM with default configuration
def example_create_llm():
    """Create an LLM instance using centralized configuration."""
    llm = create_llm()
    print(f"Created LLM with model: {llm.model_name}")
    
    if is_vocareum_gateway():
        print("✅ Using Vocareum Gateway")
    else:
        print("✅ Using direct OpenAI API")
    
    return llm


# Example 2: Create LLM with custom settings
def example_custom_llm():
    """Create an LLM with custom model and temperature."""
    llm = create_llm(model="gpt-4", temperature=0.7)
    print(f"Created custom LLM: {llm.model_name}, temperature: {llm.temperature}")
    return llm


# Example 3: Create embeddings
def example_create_embeddings():
    """Create embeddings instance using centralized configuration."""
    embeddings = create_embeddings()
    print(f"Created embeddings with model: {embeddings.model}")
    
    if is_vocareum_gateway():
        print("✅ Embeddings using Vocareum Gateway")
    
    return embeddings


# Example 4: Check configuration
def example_check_config():
    """Check current configuration."""
    config = get_openai_config()
    
    print("Current Configuration:")
    print(f"  Model: {config['model']}")
    print(f"  Embedding Model: {config['embedding_model']}")
    print(f"  Using Gateway: {is_vocareum_gateway()}")
    
    if config['base_url']:
        print(f"  Gateway URL: {config['base_url']}")
    else:
        print("  API Base: Direct OpenAI API")
    
    # Mask API key for security
    api_key_preview = config['api_key'][:10] + "..." if len(config['api_key']) > 10 else "***"
    print(f"  API Key: {api_key_preview}")
    
    return config


if __name__ == "__main__":
    print("=" * 60)
    print("OpenAI Configuration Examples")
    print("=" * 60)
    print()
    
    print("1. Checking configuration...")
    example_check_config()
    print()
    
    print("2. Creating default LLM...")
    llm = example_create_llm()
    print()
    
    print("3. Creating custom LLM...")
    custom_llm = example_custom_llm()
    print()
    
    print("4. Creating embeddings...")
    embeddings = example_create_embeddings()
    print()
    
    print("=" * 60)
    print("✅ Configuration examples completed!")
    print("=" * 60)

