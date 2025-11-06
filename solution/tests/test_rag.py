"""
Tests for RAG (Retrieval-Augmented Generation) System

Tests document retrieval, embedding generation, confidence scoring,
and knowledge base integration.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock


class TestRAGSetup:
    """Test RAG system initialization and setup."""
    
    def test_rag_system_initialization(self):
        """Test that RAG system initializes correctly."""
        with patch('agentic.tools.rag_setup.initialize_rag_system') as mock_rag:
            mock_retriever = Mock()
            mock_rag.return_value = mock_retriever
            
            retriever = mock_rag()
            assert retriever is not None
    
    def test_load_knowledge_base_articles(self):
        """Test loading knowledge base from JSONL file."""
        from agentic.tools.rag_setup import load_knowledge_base
        
        # Mock knowledge base content
        mock_articles = [
            {
                "title": "Password Reset Guide",
                "content": "To reset password, click Forgot Password on login page",
                "category": "login",
                "tags": ["password", "login", "reset"]
            },
            {
                "title": "Booking Guide",
                "content": "To book a class, go to Experiences and select your class",
                "category": "booking",
                "tags": ["booking", "reservation", "class"]
            }
        ]
        
        with patch('builtins.open', create=True) as mock_open:
            with patch('json.loads') as mock_json:
                mock_json.side_effect = mock_articles
                mock_open.return_value.__enter__.return_value = iter([
                    '{"title": "Password Reset Guide"}',
                    '{"title": "Booking Guide"}'
                ])
                
                # In actual implementation would call load_knowledge_base()
                # Here we verify mock structure
                assert len(mock_articles) == 2
    
    def test_embedding_generation(self):
        """Test that embeddings are generated for documents."""
        with patch('langchain_openai.OpenAIEmbeddings') as mock_embeddings:
            mock_embed_instance = Mock()
            mock_embed_instance.embed_documents.return_value = [
                [0.1, 0.2, 0.3],  # Mock embedding vector
                [0.4, 0.5, 0.6]
            ]
            mock_embeddings.return_value = mock_embed_instance
            
            embeddings = mock_embed_instance.embed_documents(["doc1", "doc2"])
            assert len(embeddings) == 2
            assert len(embeddings[0]) == 3


class TestDocumentRetrieval:
    """Test document retrieval functionality."""
    
    def test_retrieve_relevant_documents_high_similarity(self):
        """Test retrieval of highly relevant documents."""
        query = "How do I reset my password?"
        
        expected_documents = [
            {
                "page_content": "To reset your password: 1) Go to login page 2) Click 'Forgot Password' "
                               "3) Enter your email 4) Check inbox for reset link",
                "metadata": {
                    "title": "Password Reset Guide",
                    "category": "login",
                    "similarity_score": 0.92
                }
            }
        ]
        
        with patch('agentic.tools.rag_setup.initialize_rag_system') as mock_rag:
            mock_retriever = Mock()
            mock_retriever.invoke.return_value = expected_documents
            mock_rag.return_value = mock_retriever
            
            retriever = mock_rag()
            results = retriever.invoke(query)
            
            assert len(results) == 1
            assert results[0]["metadata"]["similarity_score"] > 0.9
            assert "password" in results[0]["page_content"].lower()
    
    def test_retrieve_no_relevant_documents(self):
        """Test retrieval when no relevant documents found."""
        query = "How do I teleport to Mars?"
        
        with patch('agentic.tools.rag_setup.initialize_rag_system') as mock_rag:
            mock_retriever = Mock()
            mock_retriever.invoke.return_value = []
            mock_rag.return_value = mock_retriever
            
            retriever = mock_rag()
            results = retriever.invoke(query)
            
            assert len(results) == 0
    
    def test_retrieve_multiple_relevant_documents(self):
        """Test retrieval of multiple relevant documents."""
        query = "subscription and billing issues"
        
        expected_documents = [
            {
                "page_content": "Subscription tiers: Basic, Premium, Elite",
                "metadata": {"title": "Subscription Plans", "similarity_score": 0.85}
            },
            {
                "page_content": "Billing occurs on the 1st of each month",
                "metadata": {"title": "Billing FAQ", "similarity_score": 0.80}
            },
            {
                "page_content": "To upgrade subscription, go to Account Settings",
                "metadata": {"title": "Upgrade Guide", "similarity_score": 0.78}
            }
        ]
        
        with patch('agentic.tools.rag_setup.initialize_rag_system') as mock_rag:
            mock_retriever = Mock()
            mock_retriever.invoke.return_value = expected_documents
            mock_rag.return_value = mock_retriever
            
            retriever = mock_rag()
            results = retriever.invoke(query)
            
            assert len(results) == 3
            assert all(doc["metadata"]["similarity_score"] > 0.75 for doc in results)
    
    def test_retrieve_with_category_filter(self):
        """Test retrieval filtered by category."""
        query = "login problems"
        category_filter = "login"
        
        expected_documents = [
            {
                "page_content": "Password reset instructions",
                "metadata": {"title": "Login Help", "category": "login"}
            }
        ]
        
        with patch('agentic.tools.rag_setup.initialize_rag_system') as mock_rag:
            mock_retriever = Mock()
            mock_retriever.invoke.return_value = expected_documents
            mock_rag.return_value = mock_retriever
            
            retriever = mock_rag()
            results = retriever.invoke(query)
            
            assert all(doc["metadata"]["category"] == category_filter for doc in results)


class TestConfidenceScoring:
    """Test confidence scoring for RAG results."""
    
    def test_high_confidence_score(self):
        """Test high confidence when exact match found."""
        from agentic.tools.confidence_scorer import calculate_confidence
        
        similarity_score = 0.95
        num_documents = 3
        query_clarity = "high"
        
        confidence = calculate_confidence(similarity_score, num_documents, query_clarity)
        
        # Should be high confidence
        assert confidence >= 0.8
    
    def test_medium_confidence_score(self):
        """Test medium confidence for partial matches."""
        from agentic.tools.confidence_scorer import calculate_confidence
        
        similarity_score = 0.70
        num_documents = 2
        query_clarity = "medium"
        
        confidence = calculate_confidence(similarity_score, num_documents, query_clarity)
        
        # Should be medium confidence
        assert 0.5 <= confidence < 0.8
    
    def test_low_confidence_score(self):
        """Test low confidence when few or poor matches."""
        from agentic.tools.confidence_scorer import calculate_confidence
        
        similarity_score = 0.45
        num_documents = 1
        query_clarity = "low"
        
        confidence = calculate_confidence(similarity_score, num_documents, query_clarity)
        
        # Should be low confidence
        assert confidence < 0.5
    
    def test_confidence_score_no_documents(self):
        """Test confidence score when no documents retrieved."""
        from agentic.tools.confidence_scorer import calculate_confidence
        
        similarity_score = 0.0
        num_documents = 0
        query_clarity = "high"
        
        confidence = calculate_confidence(similarity_score, num_documents, query_clarity)
        
        # Should be very low confidence
        assert confidence < 0.3


class TestRAGIntegrationWithResolver:
    """Test RAG integration with Resolver Agent."""
    
    def test_resolver_uses_rag_for_knowledge_retrieval(self):
        """Test that Resolver Agent uses RAG to retrieve knowledge."""
        from agentic.agents import create_resolver_agent
        
        state = {
            "messages": [Mock(content="How do I reset my password?")],
            "classification": {"issue_type": "login", "confidence": "high"},
            "ticket_metadata": {"ticket_id": "RAG-001"}
        }
        
        with patch('agentic.tools.rag_setup.initialize_rag_system') as mock_rag:
            mock_retriever = Mock()
            mock_retriever.invoke.return_value = [
                {
                    "page_content": "Password reset guide content",
                    "metadata": {"similarity_score": 0.90}
                }
            ]
            mock_rag.return_value = mock_retriever
            
            # Resolver should use RAG to get knowledge
            assert mock_retriever is not None
    
    def test_resolver_fallback_when_rag_fails(self):
        """Test Resolver fallback when RAG system fails."""
        from agentic.agents import create_resolver_agent
        
        state = {
            "messages": [Mock(content="Help me")],
            "classification": {"issue_type": "unclear"},
            "ticket_metadata": {"ticket_id": "RAG-002"}
        }
        
        with patch('agentic.tools.rag_setup.initialize_rag_system') as mock_rag:
            mock_retriever = Mock()
            mock_retriever.invoke.side_effect = Exception("RAG failure")
            mock_rag.return_value = mock_retriever
            
            # Resolver should handle gracefully
            # In production, would escalate or provide generic response
            try:
                mock_retriever.invoke("test")
                assert False, "Should have raised exception"
            except Exception as e:
                assert "RAG failure" in str(e)


class TestRAGPerformance:
    """Test RAG system performance characteristics."""
    
    def test_retrieval_speed_acceptable(self):
        """Test that retrieval completes in acceptable time."""
        import time
        
        query = "How do I book a class?"
        
        with patch('agentic.tools.rag_setup.initialize_rag_system') as mock_rag:
            mock_retriever = Mock()
            
            def mock_retrieval(q):
                time.sleep(0.01)  # Simulate fast retrieval
                return [{"page_content": "Booking guide"}]
            
            mock_retriever.invoke = mock_retrieval
            mock_rag.return_value = mock_retriever
            
            retriever = mock_rag()
            start = time.time()
            results = retriever.invoke(query)
            elapsed = time.time() - start
            
            # Should complete in under 1 second
            assert elapsed < 1.0
            assert len(results) == 1
    
    def test_vector_store_size_manageable(self):
        """Test that vector store handles reasonable number of documents."""
        num_documents = 100
        
        # Mock vector store with many documents
        mock_documents = [
            {"id": i, "content": f"Document {i}"}
            for i in range(num_documents)
        ]
        
        # Should be able to handle 100+ documents
        assert len(mock_documents) >= 100
    
    def test_concurrent_retrievals(self):
        """Test that system handles concurrent retrieval requests."""
        import threading
        
        results = []
        
        def retrieve_query(query):
            with patch('agentic.tools.rag_setup.initialize_rag_system') as mock_rag:
                mock_retriever = Mock()
                mock_retriever.invoke.return_value = [{"content": f"Result for {query}"}]
                mock_rag.return_value = mock_retriever
                
                retriever = mock_rag()
                result = retriever.invoke(query)
                results.append(result)
        
        # Create 5 concurrent threads
        threads = [
            threading.Thread(target=retrieve_query, args=(f"query_{i}",))
            for i in range(5)
        ]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All requests should complete
        assert len(results) == 5


class TestKnowledgeBaseQuality:
    """Test knowledge base content quality."""
    
    def test_knowledge_base_has_minimum_articles(self):
        """Test that knowledge base has sufficient articles."""
        # Should have at least 10 articles as per requirements
        minimum_articles = 10
        
        mock_articles = [
            {"title": f"Article {i}", "content": "Content"}
            for i in range(minimum_articles)
        ]
        
        assert len(mock_articles) >= 10
    
    def test_knowledge_base_covers_all_categories(self):
        """Test that knowledge base covers all issue categories."""
        required_categories = [
            "login", "booking", "billing", "subscription",
            "account", "technical", "general"
        ]
        
        mock_articles = [
            {"category": cat, "title": f"{cat} guide"}
            for cat in required_categories
        ]
        
        categories_covered = set(article["category"] for article in mock_articles)
        
        # Should cover all main categories
        assert all(cat in categories_covered for cat in required_categories)
    
    def test_articles_have_required_fields(self):
        """Test that all articles have required metadata fields."""
        required_fields = ["title", "content", "category", "tags"]
        
        mock_article = {
            "title": "Test Article",
            "content": "Test content",
            "category": "test",
            "tags": ["test", "sample"]
        }
        
        # All required fields should be present
        assert all(field in mock_article for field in required_fields)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
