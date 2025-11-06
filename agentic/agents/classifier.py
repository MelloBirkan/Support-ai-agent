"""
Classifier Agent

This module implements the Classifier Agent that categorizes support tickets
by issue type, urgency, and complexity. The agent uses pure LLM reasoning
(no tools) to analyze ticket content and metadata.
"""

from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from agentic.agents.state import UDAHubState
from agentic.logging import get_logger

logger = get_logger()


CLASSIFIER_PROMPT = """You are a ticket classification specialist for UDA-Hub, a unified support system.

Your task is to analyze support tickets and classify them by:
1. Issue Type
2. Urgency Level
3. Complexity

**ISSUE TYPES:**

- **technical**: Login issues, app crashes, performance problems, QR code scanning, bugs, connectivity
- **billing**: Payment failures, refunds, invoices, pricing questions, subscription charges
- **account**: Profile management, security concerns, account deletion, password reset, data privacy
- **booking**: Reservations, cancellations, experience scheduling, availability, waitlists
- **general**: Feature requests, general questions, feedback, suggestions, "how-to" questions

**URGENCY LEVELS:**

- **critical**: Security breach, service completely down, data loss, payment system failure
- **high**: Cannot access key features, urgent event within 24 hours, billing dispute, account blocked
- **medium**: Minor bugs, general support questions, non-urgent booking changes
- **low**: Feature requests, informational questions, suggestions

**COMPLEXITY:**

- **simple**: Single, well-defined issue covered by knowledge base, straightforward resolution
- **moderate**: Multiple related issues, requires data lookup, needs context from multiple sources
- **complex**: Requires human judgment, edge cases, policy exceptions, multiple systems involved

**YOUR PROCESS:**

1. Read the user's message carefully
2. Identify the primary issue (if multiple, choose the most critical)
3. Determine issue_type from the categories above
4. Assess urgency based on impact and timeline
5. Evaluate complexity based on resolution difficulty
6. Extract relevant tags (lowercase keywords like "login", "refund", "reservation")
7. Calculate confidence score (0-1) based on clarity of the request

**OUTPUT FORMAT:**

Return a structured classification with these fields:
- issue_type: One of [technical, billing, account, booking, general]
- urgency: One of [low, medium, high, critical]
- complexity: One of [simple, moderate, complex]
- tags: List of lowercase keywords relevant for search
- confidence: Float between 0-1

**GUIDELINES:**

- Be consistent and deterministic in your classifications
- Use keywords and patterns to identify issue types
- Consider user's tone and language for urgency (e.g., "URGENT", "ASAP", "can't", "won't")
- Tags should be specific and searchable (e.g., ["login", "password"] not ["help", "issue"])
- Confidence should be high (>0.8) for clear requests, lower for ambiguous ones
- Complete classification in a single pass - be thorough but efficient

**EXAMPLES:**

User: "I can't login to my account, tried resetting password but didn't receive email"
→ issue_type: technical, urgency: high, complexity: moderate, tags: ["login", "password", "email"], confidence: 0.95

User: "How do I book a yoga class?"
→ issue_type: general, urgency: low, complexity: simple, tags: ["booking", "yoga", "how-to"], confidence: 0.98

User: "Charged twice for my premium subscription, need refund immediately"
→ issue_type: billing, urgency: high, complexity: moderate, tags: ["refund", "duplicate", "charge"], confidence: 0.92

User: "App crashes when I try to scan QR code at the gym"
→ issue_type: technical, urgency: medium, complexity: moderate, tags: ["crash", "qr-code", "scanning"], confidence: 0.90

Now analyze the incoming ticket and provide your classification."""


def create_classifier_agent(model: ChatOpenAI):
    """Create the Classifier Agent.

    The Classifier Agent categorizes support tickets by issue type, urgency,
    and complexity. It uses pure LLM reasoning without any tools.

    Args:
        model: ChatOpenAI model instance (e.g., gpt-4o-mini with temperature=0)

    Returns:
        Compiled agent graph that can be invoked with UDAHubState
    """
    return create_react_agent(
        model=model,
        tools=[],  # No tools required
        state_schema=UDAHubState,
        prompt=SystemMessage(content=CLASSIFIER_PROMPT)
    )


def classifier_node(state: UDAHubState) -> Command[Literal["supervisor"]]:
    """Classifier agent node that categorizes tickets.

    This node function is a placeholder for the signature. The actual
    implementation will be in workflow.py where the agent is invoked.

    Process:
    1. Analyze ticket content from state.messages
    2. Classify by issue_type, urgency, complexity
    3. Extract relevant tags
    4. Calculate confidence score
    5. Return Command to supervisor with classification results

    Args:
        state: Current conversation state with ticket content

    Returns:
        Command to return to supervisor with classification results in state.classification
    """
    # Agent will be invoked by workflow
    # This is a placeholder for the node function signature
    pass
