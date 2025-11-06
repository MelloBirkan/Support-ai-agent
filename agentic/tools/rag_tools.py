"""
RAG Tools for Knowledge Retrieval and Confidence Scoring

This module provides tools for:
- Initializing the RAG system with knowledge base articles
- Calculating confidence scores for resolutions
- Determining if tickets should be escalated
"""

from typing import List, TYPE_CHECKING
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.tools.retriever import create_retriever_tool
from langchain_core.documents import Document
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from data.models.udahub import Knowledge

if TYPE_CHECKING:
    from agentic.agents.state import UDAHubState


def initialize_rag_system(account_id: str = "cultpass"):
    """
    Initialize RAG system for knowledge retrieval.

    Steps:
    1. Load articles from database
    2. Create embeddings
    3. Build vector store
    4. Create retriever
    5. Create retriever tool

    Args:
        account_id: Account to load articles for (default: "cultpass")

    Returns:
        retriever_tool: LangChain tool for knowledge retrieval
    """

    # 1. Load articles from database
    engine = create_engine("sqlite:///data/core/udahub.db")
    Session = sessionmaker(bind=engine)
    session = Session()

    articles = session.query(Knowledge).filter(
        Knowledge.account_id == account_id
    ).all()

    # Convert to documents
    documents = [
        Document(
            page_content=f"Title: {article.title}\n\nContent: {article.content}",
            metadata={
                "article_id": article.article_id,
                "title": article.title,
                "tags": article.tags,
                "account_id": article.account_id
            }
        )
        for article in articles
    ]

    session.close()

    # 2. Create embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 3. Build vector store
    vectorstore = InMemoryVectorStore.from_documents(
        documents=documents,
        embedding=embeddings
    )

    # 4. Create retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 3
        }
    )

    # 5. Create retriever tool
    retriever_tool = create_retriever_tool(
        retriever=retriever,
        name="knowledge_retriever",
        description=(
            "Search the CultPass knowledge base for articles about common issues. "
            "Use this tool when you need information about: "
            "login problems, reservations, subscriptions, billing, account management, "
            "or technical issues. Input should be a search query describing the user's issue."
        )
    )

    return retriever_tool


def calculate_confidence(
    similarity_scores: List[float],
    answer_length: int,
    has_suggested_phrasing: bool,
    classification_match: bool
) -> float:
    """
    Calculate confidence score for resolution.
    
    Args:
        similarity_scores: List of similarity scores from retrieved documents
        answer_length: Length of generated answer
        has_suggested_phrasing: Whether article has suggested phrasing
        classification_match: Whether classification matches article tags
        
    Returns:
        Confidence score between 0 and 1
    """

    # Similarity score (40%)
    avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
    similarity_factor = 0.4 * avg_similarity

    # Completeness score (30%)
    if answer_length > 50:
        completeness_factor = 0.3 * 1.0
    elif answer_length > 20:
        completeness_factor = 0.3 * 0.7
    else:
        completeness_factor = 0.3 * 0.3

    # Quality score (20%)
    quality_factor = 0.2 * (1.0 if has_suggested_phrasing else 0.7)

    # Context match (10%)
    context_factor = 0.1 * (1.0 if classification_match else 0.5)

    # Final confidence
    confidence = (
        similarity_factor +
        completeness_factor +
        quality_factor +
        context_factor
    )

    return round(confidence, 2)


def should_escalate(state: "UDAHubState") -> bool:
    """
    Determine if ticket should be escalated to human agent.
    
    Args:
        state: Current workflow state
        
    Returns:
        True if ticket should be escalated
    """

    # Explicit user request
    if state.get("messages"):
        for msg in state["messages"]:
            if hasattr(msg, "content") and "human agent" in msg.content.lower():
                return True

    # Low resolver confidence
    if state.get("resolution"):
        if state["resolution"].get("confidence", 0) < 0.7:
            return True

    # No relevant knowledge
    if state.get("resolution"):
        if len(state["resolution"].get("articles_used", [])) == 0:
            return True

    # Multiple failed attempts
    if state.get("messages"):
        resolution_attempts = [
            msg for msg in state["messages"]
            if hasattr(msg, "name") and msg.name == "resolver_agent"
        ]
        if len(resolution_attempts) > 2:
            return True

    # Policy exception needed
    if state.get("tool_results"):
        if "requires_approval" in str(state["tool_results"]):
            return True

    return False
