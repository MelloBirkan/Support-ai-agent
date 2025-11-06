"""
Pytest Configuration and Shared Fixtures

Provides common test fixtures and configuration for all test modules.

IMPORTANT: This conftest patches ChatOpenAI and OpenAIEmbeddings to prevent
live API calls during tests. All tests run hermetically without network access
to OpenAI services.
"""

import pytest
import os
from unittest.mock import Mock, MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.embeddings import Embeddings
import numpy as np


class MockEmbeddings(Embeddings):
    """Mock embeddings that return deterministic vectors."""
    
    def embed_documents(self, texts):
        """Return mock embedding vectors for documents."""
        return [self._get_embedding(text) for text in texts]
    
    def embed_query(self, text):
        """Return mock embedding vector for query."""
        return self._get_embedding(text)
    
    def _get_embedding(self, text):
        """Generate deterministic embedding based on text hash."""
        # Use text length and hash to generate semi-realistic vector
        np.random.seed(hash(text) % (2**32))
        return np.random.randn(1536).tolist()  # OpenAI ada-002 dimension


@pytest.fixture(autouse=True)
def mock_openai_api(monkeypatch):
    """
    Auto-applied fixture that mocks all OpenAI API calls.
    
    This prevents live API calls and ensures tests run hermetically.
    Applied to ALL tests automatically via autouse=True.
    """
    # Set a fake API key to prevent environment errors
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-key-for-testing")
    
    # Mock ChatOpenAI
    with patch("langchain_openai.ChatOpenAI") as mock_chat:
        # Create a mock that returns structured responses
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = AIMessage(
            content="This is a mock response from the classifier agent.",
            additional_kwargs={}
        )
        mock_chat.return_value = mock_instance
        
        # Mock OpenAIEmbeddings
        with patch("langchain_openai.OpenAIEmbeddings") as mock_embeddings:
            mock_embeddings.return_value = MockEmbeddings()
            
            yield {
                "chat": mock_chat,
                "embeddings": mock_embeddings
            }


@pytest.fixture
def mock_llm():
    """Mock LLM for testing without API calls."""
    llm = Mock()
    llm.invoke = Mock(return_value=AIMessage(content="Mock response"))
    return llm


@pytest.fixture
def mock_chat_openai():
    """Mock ChatOpenAI for explicit use in tests."""
    mock = MagicMock()
    mock.invoke.return_value = AIMessage(
        content="Mocked LLM response",
        additional_kwargs={}
    )
    return mock


@pytest.fixture
def mock_embeddings():
    """Mock OpenAIEmbeddings for explicit use in tests."""
    return MockEmbeddings()


@pytest.fixture
def mock_rag_system():
    """Mock RAG retrieval system."""
    retriever = Mock()
    retriever.invoke = Mock(return_value=[
        {
            "page_content": "Mock knowledge base content",
            "metadata": {"title": "Mock Article", "similarity_score": 0.85}
        }
    ])
    return retriever


@pytest.fixture
def mock_vector_store():
    """Mock vector store with sample documents."""
    mock = MagicMock()
    mock.similarity_search_with_score.return_value = [
        (
            Mock(page_content="Login troubleshooting guide", metadata={"article_id": "article_1"}),
            0.92
        ),
        (
            Mock(page_content="Password reset instructions", metadata={"article_id": "article_2"}),
            0.85
        ),
        (
            Mock(page_content="Account security best practices", metadata={"article_id": "article_3"}),
            0.78
        )
    ]
    return mock


@pytest.fixture
def mock_database():
    """Mock database responses."""
    db = {
        "users": {
            "test@example.com": {
                "user_id": 1,
                "email": "test@example.com",
                "name": "Test User"
            }
        },
        "subscriptions": {
            1: {
                "tier": "premium",
                "status": "active",
                "credits_remaining": 10
            }
        },
        "reservations": {
            1: [
                {
                    "reservation_id": 101,
                    "experience": "Yoga Class",
                    "date": "2025-11-10",
                    "status": "confirmed"
                }
            ]
        }
    }
    return db


@pytest.fixture
def mock_memory_store():
    """Mock CustomerMemoryStore."""
    mock = MagicMock()
    mock.get_user_ticket_history.return_value = []
    mock.get_user_preferences.return_value = {}
    mock.find_similar_issues.return_value = []
    mock.get_recurring_issues.return_value = {}
    mock.store_ticket_resolution.return_value = None
    mock.store_conversation_summary.return_value = None
    return mock


@pytest.fixture
def sample_state():
    """Sample workflow state for testing."""
    return {
        "messages": [HumanMessage(content="I need help with my account")],
        "ticket_metadata": {
            "ticket_id": "TEST-001",
            "account_id": "cultpass",
            "user_id": "user_123"
        },
        "routing_history": []
    }


@pytest.fixture
def sample_classified_state():
    """Sample state after classification."""
    return {
        "messages": [HumanMessage(content="I need help with my account")],
        "classification": {
            "issue_type": "account",
            "confidence": 0.95,
            "urgency": "medium",
            "complexity": "simple",
            "tags": ["account", "help"]
        },
        "ticket_metadata": {
            "ticket_id": "TEST-001",
            "account_id": "cultpass",
            "user_id": "user_123"
        },
        "routing_history": ["classifier"]
    }


@pytest.fixture
def classification_response():
    """Mock classification result."""
    return {
        "issue_type": "technical",
        "urgency": "high",
        "complexity": "moderate",
        "tags": ["login", "password"],
        "confidence": 0.92
    }


@pytest.fixture
def resolution_response():
    """Mock resolution result."""
    return {
        "resolved": True,
        "confidence": 0.85,
        "answer": "Try resetting your password using the 'Forgot Password' link on the login screen.",
        "articles_used": ["article_1", "article_2"],
        "escalation_reason": None
    }


@pytest.fixture
def tool_results_response():
    """Mock tool execution result."""
    return {
        "tool_name": "user_lookup_tool",
        "success": True,
        "data": {
            "user_id": "user_123",
            "email": "test@example.com",
            "full_name": "Test User",
            "is_blocked": False,
            "subscription": {
                "tier": "premium",
                "status": "active"
            }
        }
    }


@pytest.fixture
def escalation_response():
    """Mock escalation result."""
    return {
        "summary": "User cannot login despite password reset",
        "attempted_steps": [
            "Classified as technical issue (high urgency)",
            "Attempted resolution with confidence 0.65",
            "Checked user account status"
        ],
        "priority": "P2",
        "recommended_action": "Manually verify email and reset authentication tokens",
        "context": {
            "classification": {"issue_type": "technical", "urgency": "high"},
            "user_id": "user_123"
        }
    }
