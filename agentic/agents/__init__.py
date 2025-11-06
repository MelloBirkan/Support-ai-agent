"""
UDA-Hub Specialized Agents

This module provides specialized agents for the UDA-Hub multi-agent system:
- Classifier Agent: Categorizes tickets by type, urgency, and complexity
- Resolver Agent: Resolves tickets using knowledge base via RAG
- Tool Agent: Executes database operations on CultPass system
- Escalation Agent: Handles cases requiring human intervention

Each agent is created using LangGraph's create_react_agent and uses
the shared UDAHubState schema for state management.

Example usage:
    from agentic.agents import (
        UDAHubState,
        create_classifier_agent,
        create_resolver_agent,
        create_tool_agent,
        create_escalation_agent
    )
    from langchain_openai import ChatOpenAI
    from agentic.tools import initialize_rag_system

    # Initialize model
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Initialize RAG system
    retriever_tool = initialize_rag_system(account_id="cultpass")

    # Create agents
    classifier = create_classifier_agent(model)
    resolver = create_resolver_agent(model, retriever_tool)
    tool_agent = create_tool_agent(model)
    escalation = create_escalation_agent(model)
"""

from agentic.agents.state import UDAHubState
from agentic.agents.classifier import (
    create_classifier_agent,
    classifier_node,
    CLASSIFIER_PROMPT
)
from agentic.agents.resolver import (
    create_resolver_agent,
    resolver_node,
    RESOLVER_PROMPT
)
from agentic.agents.tool_agent import (
    create_tool_agent,
    tool_agent_node,
    TOOL_AGENT_PROMPT
)
from agentic.agents.escalation import (
    create_escalation_agent,
    escalation_node,
    ESCALATION_PROMPT
)

__all__ = [
    # State schema
    "UDAHubState",

    # Classifier Agent
    "create_classifier_agent",
    "classifier_node",
    "CLASSIFIER_PROMPT",

    # Resolver Agent
    "create_resolver_agent",
    "resolver_node",
    "RESOLVER_PROMPT",

    # Tool Agent
    "create_tool_agent",
    "tool_agent_node",
    "TOOL_AGENT_PROMPT",

    # Escalation Agent
    "create_escalation_agent",
    "escalation_node",
    "ESCALATION_PROMPT",
]
