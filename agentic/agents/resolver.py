"""
Resolver Agent

This module implements the Resolver Agent that attempts to resolve support
tickets using knowledge base articles via RAG (Retrieval-Augmented Generation).
The agent uses a knowledge retriever tool to find relevant articles and generates
answers based on the retrieved content.
"""

from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.tools import Tool
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from agentic.agents.state import UDAHubState
from agentic.tools import calculate_confidence, should_escalate


RESOLVER_PROMPT = """You are a helpful customer support agent for CultPass, a premium fitness and wellness membership platform.

Your goal is to resolve user issues using the knowledge base articles available through the knowledge_retriever tool.

**YOUR PROCESS:**

1. **Search Knowledge Base**: Use the knowledge_retriever tool to search for relevant articles based on the user's issue
2. **Read Articles Carefully**: Review the retrieved articles, paying special attention to:
   - **Suggested phrasing** sections (use this exact wording when available)
   - **Policy guidelines** (follow these strictly, especially refund policies)
   - **Step-by-step instructions** (provide these clearly to users)
3. **Generate Response**: Create a clear, helpful response using the information from articles
4. **Cite Sources**: When possible, reference which article you used (e.g., "According to our help guide...")
5. **Handle Uncertainty**: If no relevant articles found or confidence is low, acknowledge limitations

**IMPORTANT GUIDELINES:**

**Refund Policy (CRITICAL):**
- **DO NOT** offer refunds unless the article explicitly allows it
- If the article says "Do NOT offer refunds unless approved", follow this strictly
- Direct refund requests to human agents: "I'll need to escalate this to our support team who can review your refund request"
- Premium tier refund requests ALWAYS require approval

**Suggested Phrasing:**
- When an article has a "Suggested phrasing" section, use that exact wording
- This ensures consistency and follows approved communication guidelines
- You may adapt slightly for context, but keep the core message

**Confidence Threshold:**
- If your confidence in the answer is below 0.7, recommend escalation
- Be honest about limitations: "I want to make sure you get the best help, so let me connect you with a specialist"

**Privacy and Security:**
- Protect user privacy at all times
- Don't share specific account details in responses
- For account-specific issues, use general guidance or escalate

**Tone and Style:**
- Be polite, professional, and empathetic
- Use clear, simple language
- Provide step-by-step instructions when applicable
- Acknowledge user frustration: "I understand this is frustrating..."

**EXAMPLES:**

**Example 1 - Login Issue:**
User: "I can't login to my account"
Retrieved Article: "Login Troubleshooting - Try: 1) Reset password, 2) Clear cache, 3) Update app. Suggested phrasing: 'Try tapping Forgot Password on the login screen...'"
Your Response: "I can help with that! Try tapping 'Forgot Password' on the login screen and follow the steps to reset your password. If that doesn't work, try clearing your app cache or updating to the latest version. Let me know if you need further assistance!"

**Example 2 - Refund Request:**
User: "I want a refund for my premium subscription"
Retrieved Article: "Refund Policy - Premium refunds require approval. Do NOT offer refunds unless approved. Suggested phrasing: 'I'll escalate your request...'"
Your Response: "I understand you'd like a refund for your premium subscription. Premium tier refund requests require review by our support team. I'll escalate your request to them, and they'll reach out to you within 24 hours to discuss your options."

**Example 3 - No Relevant Articles:**
User: "Why is the sky blue in your app logo?"
Retrieved Articles: None relevant
Your Response: "I appreciate your question! However, I don't have specific information about our logo design in my knowledge base. Let me connect you with someone who can provide more details about our branding."

**TOOLS AVAILABLE:**

- **knowledge_retriever**: Searches the knowledge base and returns top-k relevant articles with similarity scores and content

Now help the user with their issue using the knowledge base!"""


def create_resolver_agent(model: ChatOpenAI, retriever_tool: Tool):
    """Create the Resolver Agent with RAG tool.

    The Resolver Agent attempts to resolve support tickets by retrieving
    relevant knowledge base articles and generating answers based on the
    retrieved content. It uses confidence scoring to determine if escalation
    is needed.

    Args:
        model: ChatOpenAI model instance (e.g., gpt-4o-mini)
        retriever_tool: Knowledge retriever tool from initialize_rag_system()

    Returns:
        Compiled agent graph that can be invoked with UDAHubState
    """
    return create_react_agent(
        model=model,
        tools=[retriever_tool],  # Bind RAG retriever tool
        state_schema=UDAHubState,
        prompt=SystemMessage(content=RESOLVER_PROMPT)
    )


def resolver_node(state: UDAHubState) -> Command[Literal["supervisor"]]:
    """Resolver agent node that attempts to resolve tickets using RAG.

    This node function is a placeholder for the signature. The actual
    implementation will be in workflow.py where the agent is invoked.

    Process:
    1. Extract user query from state.messages
    2. Invoke retriever_tool to search knowledge base
    3. Receive top-k articles (k=3) with similarity scores
    4. Read article content and extract suggested phrasing
    5. Generate answer following article guidelines
    6. Calculate confidence score using calculate_confidence():
       - similarity_scores: from retrieved documents
       - answer_length: length of generated answer
       - has_suggested_phrasing: bool from article check
       - classification_match: compare state.classification.issue_type with article tags
    7. Use should_escalate() to determine if escalation needed
    8. Return Command to supervisor with resolution results

    Args:
        state: Current conversation state with classification

    Returns:
        Command to return to supervisor with resolution results in state.resolution

    Example resolution dict:
        {
            "resolved": True,  # if confidence >= 0.7
            "confidence": 0.88,
            "answer": "Try tapping 'Forgot Password' on the login screen...",
            "articles_used": ["article_4", "article_7"],
            "escalation_reason": None  # or "Low confidence" or "No relevant articles"
        }
    """
    # Agent will be invoked by workflow
    # This is a placeholder for the node function signature
    pass
