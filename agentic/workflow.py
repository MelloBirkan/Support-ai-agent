"""
UDA-Hub Supervisor Workflow

This module implements the supervisor-based multi-agent workflow for ticket routing
and resolution. The supervisor coordinates specialized agents (Classifier, Resolver,
Tool Agent, Escalation) to handle customer support tickets efficiently.

Architecture:
    START → Supervisor → Classifier → Supervisor → Resolver/Tool Agent → Supervisor → END/Escalation
"""

from typing import Literal, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

# Import agents
from agentic.agents import (
    UDAHubState,
    create_classifier_agent,
    create_resolver_agent,
    create_tool_agent,
    create_escalation_agent,
)

# Import tools
from agentic.tools import initialize_rag_system

# Import memory
from agentic.memory import CustomerMemoryStore

# Import logging
from agentic.logging import get_logger

# Import configuration
from agentic.config import create_llm

# Initialize logger
logger = get_logger()


def _analyze_needs_tool_execution(state: UDAHubState) -> bool:
    """Determine if the ticket needs database tool execution.
    
    Returns True if:
    - Issue type is booking/billing/account and no tool results yet
    - Resolver explicitly mentioned needing data
    - Message contains requests for user data, subscriptions, reservations
    
    Args:
        state: Current workflow state
        
    Returns:
        True if tool execution is needed, False otherwise
    """
    # Check if we already have tool results
    if state.get("tool_results"):
        return False
    
    # Check classification for issue types that typically need data
    classification = state.get("classification")
    if classification:
        issue_type = classification.get("issue_type", "")
        if issue_type in ["booking", "billing", "account"]:
            # Check if resolver attempted but needs data
            resolution = state.get("resolution")
            if resolution and not resolution.get("resolved"):
                return True
    
    # Check last message for data-related keywords
    if state.get("messages"):
        last_msg = state["messages"][-1].content.lower()
        data_keywords = [
            "check subscription", "view reservation", "find user",
            "booking details", "cancel reservation", "refund",
            "account status", "subscription status"
        ]
        if any(keyword in last_msg for keyword in data_keywords):
            return True
    
    return False


def _user_requested_human(state: UDAHubState) -> bool:
    """Check if user explicitly requested human agent.
    
    Args:
        state: Current workflow state
        
    Returns:
        True if user requested human intervention
    """
    if not state.get("messages"):
        return False
    
    last_message = state["messages"][-1].content.lower()
    human_keywords = [
        "speak to a person", "human agent", "talk to someone",
        "real person", "customer service representative",
        "speak with agent", "connect me to", "transfer to"
    ]
    
    return any(keyword in last_message for keyword in human_keywords)


def supervisor_node(state: UDAHubState) -> Command:
    """Supervisor node that routes tickets to appropriate agents.
    
    Routing logic:
    1. User requests human → Escalation
    2. No classification → Classifier
    3. Has resolution → Check confidence:
       - High confidence (≥0.7) and resolved → END
       - Low confidence or unresolved → Escalation
    4. Needs tool execution → Tool Agent (limit to 2 consecutive attempts)
    5. Classified but no resolution → Resolver
    
    Args:
        state: Current workflow state with ticket data
        
    Returns:
        Command object directing to next agent or END
    """
    # Get thread_id for logging
    thread_id = state.get("ticket_metadata", {}).get("ticket_id", "unknown")
    
    # Get routing history
    routing_history = state.get("routing_history", [])
    
    # Priority 1: Check for explicit human request
    if _user_requested_human(state):
        logger.log_routing(
            from_agent="supervisor",
            to_agent="escalation",
            reason="User explicitly requested human agent",
            thread_id=thread_id,
            state_summary={"has_classification": bool(state.get("classification"))}
        )
        return Command(goto="escalation", update={"routing_history": routing_history + ["escalation"]})
    
    # Priority 2: New tickets need classification
    if not state.get("classification"):
        logger.log_routing(
            from_agent="supervisor",
            to_agent="classifier",
            reason="No classification present, initiating ticket analysis",
            thread_id=thread_id,
            state_summary={"message_count": len(state.get("messages", []))}
        )
        return Command(goto="classifier", update={"routing_history": routing_history + ["classifier"]})
    
    # Priority 3: Check if we have a resolution attempt
    resolution = state.get("resolution")
    if resolution:
        confidence = resolution.get("confidence", 0.0)
        resolved = resolution.get("resolved", False)
        
        # High confidence resolution → End workflow
        if resolved and confidence >= 0.7:
            logger.log_routing(
                from_agent="supervisor",
                to_agent="END",
                reason=f"High confidence resolution achieved (confidence: {confidence:.2f})",
                thread_id=thread_id,
                state_summary={
                    "resolved": True,
                    "confidence": confidence,
                    "classification": state.get("classification", {}).get("issue_type")
                }
            )
            return Command(goto=END)
        
        # Low confidence or unresolved → Escalate
        if confidence < 0.7 or not resolved:
            reason = f"Low confidence ({confidence:.2f})" if confidence < 0.7 else "Unresolved issue"
            logger.log_routing(
                from_agent="supervisor",
                to_agent="escalation",
                reason=reason,
                thread_id=thread_id,
                state_summary={
                    "resolved": resolved,
                    "confidence": confidence,
                    "tool_results": bool(state.get("tool_results")),
                    "routing_attempts": len(routing_history)
                }
            )
            return Command(goto="escalation", update={"routing_history": routing_history + ["escalation"]})
    
    # Priority 4: Check if tool execution is needed
    # Count consecutive tool_agent attempts
    consecutive_tool_attempts = 0
    for agent in reversed(routing_history):
        if agent == "tool_agent":
            consecutive_tool_attempts += 1
        else:
            break
    
    if _analyze_needs_tool_execution(state):
        # Limit consecutive tool attempts to 2
        if consecutive_tool_attempts >= 2:
            logger.log_routing(
                from_agent="supervisor",
                to_agent="escalation",
                reason=f"Max tool_agent attempts reached ({consecutive_tool_attempts}), escalating",
                thread_id=thread_id,
                state_summary={
                    "classification": state.get("classification", {}).get("issue_type"),
                    "tool_attempts": consecutive_tool_attempts
                }
            )
            return Command(goto="escalation", update={"routing_history": routing_history + ["escalation"]})
        
        logger.log_routing(
            from_agent="supervisor",
            to_agent="tool_agent",
            reason="Database tool execution required for resolution",
            thread_id=thread_id,
            state_summary={
                "classification": state.get("classification", {}).get("issue_type"),
                "has_tool_results": bool(state.get("tool_results")),
                "tool_attempt": consecutive_tool_attempts + 1
            }
        )
        return Command(goto="tool_agent", update={"routing_history": routing_history + ["tool_agent"]})
    
    # Default: Attempt resolution with RAG
    logger.log_routing(
        from_agent="supervisor",
        to_agent="resolver",
        reason="Attempting resolution with knowledge base",
        thread_id=thread_id,
        state_summary={
            "classification": state.get("classification", {}),
            "has_resolution": bool(resolution)
        }
    )
    return Command(goto="resolver", update={"routing_history": routing_history + ["resolver"]})





def _extract_classification_from_response(messages: list) -> dict:
    """Extract classification dict from agent's response messages."""
    # Look for classification info in the last AI message
    for msg in reversed(messages):
        if hasattr(msg, 'content') and isinstance(msg.content, str):
            content = msg.content.lower()
            # Try to extract classification information
            classification = {
                "issue_type": "general",
                "urgency": "medium",
                "complexity": "simple",
                "tags": [],
                "confidence": 0.8
            }
            
            # Extract issue type
            if any(word in content for word in ["login", "crash", "bug", "qr", "technical"]):
                classification["issue_type"] = "technical"
            elif any(word in content for word in ["payment", "refund", "billing", "charge", "invoice"]):
                classification["issue_type"] = "billing"
            elif any(word in content for word in ["account", "password", "security", "profile"]):
                classification["issue_type"] = "account"
            elif any(word in content for word in ["booking", "reservation", "cancel", "experience"]):
                classification["issue_type"] = "booking"
            
            # Extract urgency
            if any(word in content for word in ["urgent", "critical", "emergency", "asap", "immediately"]):
                classification["urgency"] = "high"
            elif any(word in content for word in ["cannot", "can't", "won't", "doesn't work"]):
                classification["urgency"] = "high"
            
            return classification
    
    return {"issue_type": "general", "urgency": "medium", "complexity": "simple", "tags": [], "confidence": 0.7}


def _extract_resolution_from_response(messages: list, state: UDAHubState) -> dict:
    """Extract resolution dict from agent's response messages."""
    # Check if RAG tool was called successfully
    articles_used = []
    confidence = 0.6
    
    for msg in reversed(messages):
        if hasattr(msg, 'content'):
            content_str = str(msg.content)
            if "article" in content_str.lower():
                confidence = 0.85
            # Check for tool calls in message
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                articles_used = ["knowledge_article_used"]
                confidence = 0.80
    
    # Get the answer from last AI message
    answer = None
    for msg in reversed(messages):
        if hasattr(msg, 'content') and msg.type == "ai":
            answer = msg.content
            break
    
    resolved = confidence >= 0.7 and answer is not None
    escalation_reason = None if resolved else "Low confidence resolution"
    
    return {
        "resolved": resolved,
        "confidence": confidence,
        "answer": answer,
        "articles_used": articles_used,
        "escalation_reason": escalation_reason
    }


def _extract_tool_results_from_response(messages: list) -> dict:
    """Extract tool execution results from agent's response messages."""
    tool_results = {
        "tool_name": "unknown",
        "success": False,
        "data": None
    }
    
    # Look for tool calls and results
    for msg in messages:
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            tool_results["tool_name"] = msg.tool_calls[0].get("name", "unknown")
            tool_results["success"] = True
        if msg.type == "tool":
            tool_results["data"] = msg.content
            tool_results["success"] = True
    
    return tool_results


def _extract_escalation_from_response(messages: list, state: UDAHubState) -> dict:
    """Extract escalation dict from agent's response messages."""
    # Build escalation summary from state
    classification = state.get("classification", {})
    resolution = state.get("resolution", {})
    
    attempted_steps = []
    if classification:
        attempted_steps.append(f"Classified as {classification.get('issue_type', 'unknown')} issue ({classification.get('urgency', 'unknown')} urgency)")
    if resolution:
        attempted_steps.append(f"Attempted resolution with confidence {resolution.get('confidence', 0):.2f}")
    if state.get("tool_results"):
        attempted_steps.append(f"Executed {state['tool_results'].get('tool_name', 'tool')}")
    
    # Determine priority from classification urgency
    urgency = classification.get("urgency", "medium")
    priority_map = {"critical": "P1", "high": "P2", "medium": "P3", "low": "P4"}
    priority = priority_map.get(urgency, "P3")
    
    # Get summary from last AI message
    summary = "Issue requires human intervention"
    for msg in reversed(messages):
        if hasattr(msg, 'content') and msg.type == "ai":
            summary = msg.content[:200]  # First 200 chars as summary
            break
    
    return {
        "summary": summary,
        "attempted_steps": attempted_steps,
        "priority": priority,
        "recommended_action": "Review ticket and provide appropriate resolution",
        "context": {
            "classification": classification,
            "resolution": resolution,
            "tool_results": state.get("tool_results")
        }
    }


def create_orchestrator(
    model: ChatOpenAI | None = None,
    account_id: str = "cultpass",
    checkpointer: MemorySaver | None = None,
    store: InMemoryStore | None = None,
    enable_long_term_memory: bool = False,
    db_path: str = "data/core/udahub.db"
) -> StateGraph:
    """Create the complete supervisor workflow orchestrator.
    
    This function builds a LangGraph StateGraph that coordinates all specialized
    agents using a supervisor pattern. The supervisor makes routing decisions
    based on ticket state, classifications, and confidence scores.
    
    Workflow structure:
        START → supervisor → classifier → supervisor → resolver/tool_agent 
              → supervisor → END/escalation
    
    Args:
        model: LLM model to use for agents. Defaults to gpt-4o-mini.
        account_id: Account identifier for RAG system. Defaults to "cultpass".
        checkpointer: Checkpointer for conversation memory. Defaults to MemorySaver.
        store: In-memory store for cross-thread data. Defaults to InMemoryStore.
        enable_long_term_memory: Enable persistent database memory features.
        db_path: Path to database for long-term memory.
        
    Returns:
        Compiled StateGraph ready for invocation
        
    Example:
        >>> orchestrator = create_orchestrator(enable_long_term_memory=True)
        >>> state = {
        ...     "messages": [HumanMessage(content="I can't login")],
        ...     "ticket_metadata": {"ticket_id": "T123", "account_id": "cultpass"}
        ... }
        >>> result = orchestrator.invoke(state, {"configurable": {"thread_id": "T123"}})
    """
    # Initialize model if not provided
    # Uses centralized configuration (supports Vocareum Gateway via OPENAI_API_BASE)
    if model is None:
        model = create_llm(temperature=0)
    
    # Initialize checkpointer if not provided (short-term memory)
    if checkpointer is None:
        checkpointer = MemorySaver()
    
    # Initialize store if not provided (cross-thread memory)
    if store is None:
        store = InMemoryStore()
    
    # Initialize long-term memory store if enabled
    memory_store = None
    if enable_long_term_memory:
        memory_store = CustomerMemoryStore(db_path=db_path)
    
    # Initialize RAG system for resolver
    retriever_tool = initialize_rag_system(account_id=account_id)
    
    # Create all specialized agents
    classifier_agent = create_classifier_agent(model)
    resolver_agent = create_resolver_agent(model, retriever_tool)
    tool_agent = create_tool_agent(model)
    escalation_agent = create_escalation_agent(model)
    
    # Create wrapper functions that extract outputs into state
    def classifier_wrapper(state: UDAHubState) -> dict:
        """Wrapper that invokes classifier and extracts classification into state."""
        result = classifier_agent.invoke(state)
        classification = _extract_classification_from_response(result["messages"])
        logger.log_agent(
            agent_name="classifier",
            action="classification_complete",
            details=classification,
            thread_id=state.get("ticket_metadata", {}).get("ticket_id", "unknown")
        )
        return {
            "messages": result["messages"],
            "classification": classification
        }
    
    def resolver_wrapper(state: UDAHubState) -> dict:
        """Wrapper that invokes resolver and extracts resolution into state."""
        # Integrate memory context if available
        if memory_store and state.get("ticket_metadata", {}).get("user_id"):
            from agentic.memory import get_memory_context_for_agent
            user_id = state["ticket_metadata"]["user_id"]
            classification = state.get("classification", {})
            memory_context = get_memory_context_for_agent(
                user_id=user_id,
                current_issue_type=classification.get("issue_type"),
                current_tags=classification.get("tags", []),
                memory_store=memory_store
            )
            # Prepend memory context as a system message if present
            if memory_context:
                state = {**state, "messages": state["messages"] + [SystemMessage(content=memory_context)]}
        
        result = resolver_agent.invoke(state)
        resolution = _extract_resolution_from_response(result["messages"], state)
        logger.log_agent(
            agent_name="resolver",
            action="resolution_attempt",
            details=resolution,
            thread_id=state.get("ticket_metadata", {}).get("ticket_id", "unknown")
        )
        
        # Store resolution in long-term memory if enabled and resolved
        if memory_store and resolution.get("resolved") and state.get("ticket_metadata", {}).get("ticket_id"):
            memory_store.store_ticket_resolution(
                ticket_id=state["ticket_metadata"]["ticket_id"],
                resolution_method="rag",
                confidence=resolution.get("confidence", 0.0),
                articles_used=",".join(resolution.get("articles_used", [])),
                tool_results=None
            )
        
        return {
            "messages": result["messages"],
            "resolution": resolution
        }
    
    def tool_wrapper(state: UDAHubState) -> dict:
        """Wrapper that invokes tool agent and extracts tool_results into state."""
        result = tool_agent.invoke(state)
        tool_results = _extract_tool_results_from_response(result["messages"])
        logger.log_agent(
            agent_name="tool_agent",
            action="tool_execution",
            details=tool_results,
            thread_id=state.get("ticket_metadata", {}).get("ticket_id", "unknown")
        )
        return {
            "messages": result["messages"],
            "tool_results": tool_results
        }
    
    def escalation_wrapper(state: UDAHubState) -> dict:
        """Wrapper that invokes escalation agent and extracts escalation into state."""
        result = escalation_agent.invoke(state)
        escalation = _extract_escalation_from_response(result["messages"], state)
        logger.log_agent(
            agent_name="escalation",
            action="escalation_prepared",
            details=escalation,
            thread_id=state.get("ticket_metadata", {}).get("ticket_id", "unknown")
        )
        
        # Store conversation summary if enabled
        if memory_store and state.get("ticket_metadata", {}).get("ticket_id"):
            memory_store.store_conversation_summary(
                ticket_id=state["ticket_metadata"]["ticket_id"],
                summary=escalation.get("summary", ""),
                key_points=",".join(escalation.get("attempted_steps", [])),
                message_count=len(state.get("messages", []))
            )
        
        return {
            "messages": result["messages"],
            "escalation": escalation
        }
    
    # Create StateGraph with UDAHubState schema
    workflow = StateGraph(UDAHubState)
    
    # Add supervisor node
    workflow.add_node("supervisor", supervisor_node)
    
    # Add agent nodes (using wrapper functions)
    workflow.add_node("classifier", classifier_wrapper)
    workflow.add_node("resolver", resolver_wrapper)
    workflow.add_node("tool_agent", tool_wrapper)
    workflow.add_node("escalation", escalation_wrapper)
    
    # Define workflow edges
    # Start with supervisor
    workflow.add_edge(START, "supervisor")
    
    # Agents route back to supervisor after completion
    # (except escalation which ends the workflow)
    workflow.add_edge("classifier", "supervisor")
    workflow.add_edge("resolver", "supervisor")
    workflow.add_edge("tool_agent", "supervisor")
    workflow.add_edge("escalation", END)
    
    # Compile workflow with checkpointer and store for memory capabilities
    compiled_graph = workflow.compile(
        checkpointer=checkpointer,
        store=store
    )
    
    # Store memory_store reference for later use
    compiled_graph.memory_store = memory_store
    
    return compiled_graph


# Create default orchestrator instance
orchestrator = create_orchestrator()


__all__ = [
    "orchestrator",
    "create_orchestrator",
    "UDAHubState",
    "supervisor_node",
    "CustomerMemoryStore",
    "InMemoryStore",
    "MemorySaver"
]