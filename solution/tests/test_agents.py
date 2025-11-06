"""
Unit Tests for UDA-Hub Agents

Tests individual agent functionality including classification, resolution,
tool execution, and escalation logic.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage
from agentic.agents import (
    UDAHubState,
    create_classifier_agent,
    create_resolver_agent,
    create_tool_agent,
    create_escalation_agent,
)


class TestClassifierAgent:
    """Test suite for the Classifier Agent."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM for testing."""
        llm = Mock()
        llm.invoke = Mock()
        return llm
    
    def test_classify_login_issue_high_confidence(self):
        """Test classification of clear login issue with high confidence."""
        state = {
            "messages": [HumanMessage(content="I can't login to my account")],
            "ticket_metadata": {"ticket_id": "TEST-001"}
        }
        
        # Expected classification
        expected = {
            "issue_type": "login",
            "confidence": "high",
            "urgency": "medium",
            "category": "technical"
        }
        
        # Mock the classifier response
        with patch('agentic.agents.classifier.ChatOpenAI') as mock_chat:
            mock_instance = Mock()
            mock_instance.invoke.return_value = AIMessage(
                content="classification: login, confidence: high"
            )
            mock_chat.return_value = mock_instance
            
            classifier = create_classifier_agent()
            result = classifier.invoke(state)
            
            assert result["classification"] is not None
            assert "issue_type" in result["classification"]
    
    def test_classify_booking_issue_medium_confidence(self):
        """Test classification of booking-related query."""
        state = {
            "messages": [HumanMessage(content="I need to check my yoga class reservation")],
            "ticket_metadata": {"ticket_id": "TEST-002"}
        }
        
        with patch('agentic.agents.classifier.ChatOpenAI') as mock_chat:
            mock_instance = Mock()
            mock_instance.invoke.return_value = AIMessage(
                content="classification: booking, confidence: medium"
            )
            mock_chat.return_value = mock_instance
            
            classifier = create_classifier_agent()
            result = classifier.invoke(state)
            
            assert result["classification"] is not None
    
    def test_classify_billing_dispute_high_urgency(self):
        """Test classification of billing dispute with high urgency."""
        state = {
            "messages": [HumanMessage(
                content="I was charged three times for my subscription! This is unacceptable!"
            )],
            "ticket_metadata": {"ticket_id": "TEST-003"}
        }
        
        with patch('agentic.agents.classifier.ChatOpenAI') as mock_chat:
            mock_instance = Mock()
            mock_instance.invoke.return_value = AIMessage(
                content="classification: billing, confidence: high, urgency: high"
            )
            mock_chat.return_value = mock_instance
            
            classifier = create_classifier_agent()
            result = classifier.invoke(state)
            
            assert result["classification"] is not None
    
    def test_classify_ambiguous_query_low_confidence(self):
        """Test classification of ambiguous or unclear query."""
        state = {
            "messages": [HumanMessage(content="It's not working")],
            "ticket_metadata": {"ticket_id": "TEST-004"}
        }
        
        with patch('agentic.agents.classifier.ChatOpenAI') as mock_chat:
            mock_instance = Mock()
            mock_instance.invoke.return_value = AIMessage(
                content="classification: unclear, confidence: low"
            )
            mock_chat.return_value = mock_instance
            
            classifier = create_classifier_agent()
            result = classifier.invoke(state)
            
            assert result["classification"] is not None
    
    def test_classify_multiple_issues(self):
        """Test classification when user has multiple related issues."""
        state = {
            "messages": [HumanMessage(
                content="I can't login and I also need to cancel my booking"
            )],
            "ticket_metadata": {"ticket_id": "TEST-005"}
        }
        
        with patch('agentic.agents.classifier.ChatOpenAI') as mock_chat:
            mock_instance = Mock()
            mock_instance.invoke.return_value = AIMessage(
                content="classification: multiple_issues, confidence: medium"
            )
            mock_chat.return_value = mock_instance
            
            classifier = create_classifier_agent()
            result = classifier.invoke(state)
            
            assert result["classification"] is not None


class TestResolverAgent:
    """Test suite for the Resolver Agent (RAG-based)."""
    
    def test_resolve_with_high_confidence_rag_match(self):
        """Test resolution when RAG finds high-confidence match."""
        state = {
            "messages": [HumanMessage(content="How do I reset my password?")],
            "classification": {"issue_type": "login", "confidence": "high"},
            "ticket_metadata": {"ticket_id": "TEST-006"}
        }
        
        with patch('agentic.agents.resolver.initialize_rag_system') as mock_rag:
            mock_retriever = Mock()
            mock_retriever.invoke.return_value = [
                {"page_content": "Password reset guide: Go to login page, click 'Forgot Password'"}
            ]
            mock_rag.return_value = mock_retriever
            
            with patch('agentic.agents.resolver.ChatOpenAI') as mock_chat:
                mock_instance = Mock()
                mock_instance.invoke.return_value = AIMessage(
                    content="To reset your password, go to the login page and click 'Forgot Password'"
                )
                mock_chat.return_value = mock_instance
                
                resolver = create_resolver_agent()
                result = resolver.invoke(state)
                
                assert result["resolution"] is not None
    
    def test_resolve_no_rag_match_low_confidence(self):
        """Test resolution when RAG doesn't find good match."""
        state = {
            "messages": [HumanMessage(content="Can you help with my obscure issue?")],
            "classification": {"issue_type": "unclear", "confidence": "low"},
            "ticket_metadata": {"ticket_id": "TEST-007"}
        }
        
        with patch('agentic.agents.resolver.initialize_rag_system') as mock_rag:
            mock_retriever = Mock()
            mock_retriever.invoke.return_value = []
            mock_rag.return_value = mock_retriever
            
            with patch('agentic.agents.resolver.ChatOpenAI') as mock_chat:
                mock_instance = Mock()
                mock_instance.invoke.return_value = AIMessage(
                    content="I need more information to help you"
                )
                mock_chat.return_value = mock_instance
                
                resolver = create_resolver_agent()
                result = resolver.invoke(state)
                
                assert result["resolution"] is not None
    
    def test_resolve_with_context_from_previous_messages(self):
        """Test resolution using conversation history."""
        state = {
            "messages": [
                HumanMessage(content="I have a problem with my subscription"),
                AIMessage(content="Can you describe the issue?"),
                HumanMessage(content="I can't access premium classes")
            ],
            "classification": {"issue_type": "subscription", "confidence": "medium"},
            "ticket_metadata": {"ticket_id": "TEST-008"}
        }
        
        with patch('agentic.agents.resolver.initialize_rag_system') as mock_rag:
            mock_retriever = Mock()
            mock_retriever.invoke.return_value = [
                {"page_content": "Premium class access requires active subscription"}
            ]
            mock_rag.return_value = mock_retriever
            
            with patch('agentic.agents.resolver.ChatOpenAI') as mock_chat:
                mock_instance = Mock()
                mock_instance.invoke.return_value = AIMessage(
                    content="Premium classes require an active subscription"
                )
                mock_chat.return_value = mock_instance
                
                resolver = create_resolver_agent()
                result = resolver.invoke(state)
                
                assert result["resolution"] is not None


class TestToolAgent:
    """Test suite for the Tool Agent (database operations)."""
    
    def test_tool_agent_query_subscription_status(self):
        """Test querying user subscription status."""
        state = {
            "messages": [HumanMessage(content="What's my subscription status?")],
            "classification": {"issue_type": "subscription", "confidence": "high"},
            "ticket_metadata": {"ticket_id": "TEST-009", "account_id": "cultpass"}
        }
        
        with patch('agentic.tools.cultpass_read_tools.get_user_by_email') as mock_tool:
            mock_tool.return_value = {
                "user_id": 1,
                "email": "test@example.com",
                "subscription": {"status": "active", "tier": "premium"}
            }
            
            with patch('agentic.agents.tool_agent.ChatOpenAI') as mock_chat:
                mock_instance = Mock()
                mock_instance.bind_tools = Mock(return_value=mock_instance)
                mock_instance.invoke.return_value = AIMessage(
                    content="Your subscription is active and premium tier"
                )
                mock_chat.return_value = mock_instance
                
                tool_agent = create_tool_agent()
                result = tool_agent.invoke(state)
                
                assert result["tool_results"] is not None
    
    def test_tool_agent_query_reservations(self):
        """Test querying user reservations."""
        state = {
            "messages": [HumanMessage(content="Show my upcoming reservations")],
            "classification": {"issue_type": "booking", "confidence": "high"},
            "ticket_metadata": {"ticket_id": "TEST-010", "account_id": "cultpass"}
        }
        
        with patch('agentic.tools.cultpass_read_tools.get_user_reservations') as mock_tool:
            mock_tool.return_value = [
                {"experience": "Yoga Class", "date": "2025-11-10", "status": "confirmed"}
            ]
            
            with patch('agentic.agents.tool_agent.ChatOpenAI') as mock_chat:
                mock_instance = Mock()
                mock_instance.bind_tools = Mock(return_value=mock_instance)
                mock_instance.invoke.return_value = AIMessage(
                    content="You have a Yoga Class on Nov 10"
                )
                mock_chat.return_value = mock_instance
                
                tool_agent = create_tool_agent()
                result = tool_agent.invoke(state)
                
                assert result["tool_results"] is not None
    
    def test_tool_agent_handles_database_error(self):
        """Test tool agent handling database errors gracefully."""
        state = {
            "messages": [HumanMessage(content="Check my account")],
            "classification": {"issue_type": "account", "confidence": "high"},
            "ticket_metadata": {"ticket_id": "TEST-011", "account_id": "cultpass"}
        }
        
        with patch('agentic.tools.cultpass_read_tools.get_user_by_email') as mock_tool:
            mock_tool.side_effect = Exception("Database connection error")
            
            with patch('agentic.agents.tool_agent.ChatOpenAI') as mock_chat:
                mock_instance = Mock()
                mock_instance.bind_tools = Mock(return_value=mock_instance)
                mock_instance.invoke.return_value = AIMessage(
                    content="I'm having trouble accessing the database right now"
                )
                mock_chat.return_value = mock_instance
                
                tool_agent = create_tool_agent()
                result = tool_agent.invoke(state)
                
                assert "error" in str(result).lower() or result["tool_results"] is not None


class TestEscalationAgent:
    """Test suite for the Escalation Agent."""
    
    def test_escalate_low_confidence_resolution(self):
        """Test escalation when resolver has low confidence."""
        state = {
            "messages": [HumanMessage(content="Complex billing dispute")],
            "classification": {"issue_type": "billing", "confidence": "medium"},
            "resolution": {"resolved": False, "confidence": "low"},
            "ticket_metadata": {"ticket_id": "TEST-012"}
        }
        
        with patch('agentic.agents.escalation.ChatOpenAI') as mock_chat:
            mock_instance = Mock()
            mock_instance.invoke.return_value = AIMessage(
                content="Escalating to human agent for complex billing issue"
            )
            mock_chat.return_value = mock_instance
            
            escalation_agent = create_escalation_agent()
            result = escalation_agent.invoke(state)
            
            assert result["escalation"] is not None
            assert result["escalation"]["escalated"] is True
    
    def test_escalate_user_requested_human(self):
        """Test escalation when user explicitly requests human agent."""
        state = {
            "messages": [
                HumanMessage(content="I need to speak to a real person about this")
            ],
            "ticket_metadata": {"ticket_id": "TEST-013"}
        }
        
        with patch('agentic.agents.escalation.ChatOpenAI') as mock_chat:
            mock_instance = Mock()
            mock_instance.invoke.return_value = AIMessage(
                content="Connecting you with a human agent"
            )
            mock_chat.return_value = mock_instance
            
            escalation_agent = create_escalation_agent()
            result = escalation_agent.invoke(state)
            
            assert result["escalation"] is not None
            assert result["escalation"]["escalated"] is True
    
    def test_escalate_preserves_context(self):
        """Test that escalation preserves full conversation context."""
        state = {
            "messages": [
                HumanMessage(content="I have a problem"),
                AIMessage(content="What's the issue?"),
                HumanMessage(content="It's very complicated and urgent")
            ],
            "classification": {"issue_type": "complex", "urgency": "high"},
            "resolution": {"resolved": False, "confidence": "low"},
            "ticket_metadata": {"ticket_id": "TEST-014"}
        }
        
        with patch('agentic.agents.escalation.ChatOpenAI') as mock_chat:
            mock_instance = Mock()
            mock_instance.invoke.return_value = AIMessage(
                content="Escalating with full context preserved"
            )
            mock_chat.return_value = mock_instance
            
            escalation_agent = create_escalation_agent()
            result = escalation_agent.invoke(state)
            
            assert result["escalation"] is not None
            assert len(result["messages"]) >= 3  # All messages preserved


class TestAgentIntegration:
    """Test interactions between agents."""
    
    def test_classifier_to_resolver_pipeline(self):
        """Test data flow from classifier to resolver."""
        # First, classify
        initial_state = {
            "messages": [HumanMessage(content="How do I reset my password?")],
            "ticket_metadata": {"ticket_id": "TEST-015"}
        }
        
        with patch('agentic.agents.classifier.ChatOpenAI') as mock_chat:
            mock_instance = Mock()
            mock_instance.invoke.return_value = AIMessage(
                content="classification: login, confidence: high"
            )
            mock_chat.return_value = mock_instance
            
            classifier = create_classifier_agent()
            classified_state = classifier.invoke(initial_state)
            
            assert classified_state["classification"] is not None
        
        # Then, resolve using classified state
        with patch('agentic.agents.resolver.initialize_rag_system') as mock_rag:
            mock_retriever = Mock()
            mock_retriever.invoke.return_value = [
                {"page_content": "Password reset instructions"}
            ]
            mock_rag.return_value = mock_retriever
            
            with patch('agentic.agents.resolver.ChatOpenAI') as mock_chat:
                mock_instance = Mock()
                mock_instance.invoke.return_value = AIMessage(
                    content="Here are the password reset instructions"
                )
                mock_chat.return_value = mock_instance
                
                resolver = create_resolver_agent()
                resolved_state = resolver.invoke(classified_state)
                
                assert resolved_state["resolution"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
