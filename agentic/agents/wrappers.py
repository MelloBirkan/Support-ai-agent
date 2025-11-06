"""
Agent Wrapper Functions with Logging

This module provides wrapper functions that add structured logging
to agent invocations. These wrappers capture agent decisions, tool usage,
and outcomes for searchable logging.
"""

from typing import Dict, Any
import json
from langchain_core.messages import AIMessage
from langgraph.prebuilt.chat_agent_executor import AgentState
from agentic.agents.state import UDAHubState
from agentic.logging import get_logger

logger = get_logger()


def wrap_classifier_with_logging(classifier_agent):
    """
    Wrap classifier agent to add logging.
    
    Args:
        classifier_agent: The compiled classifier agent
        
    Returns:
        Wrapped agent function with logging
    """
    def logged_classifier(state: UDAHubState) -> UDAHubState:
        thread_id = state.get("ticket_metadata", {}).get("ticket_id", "unknown")
        user_message = state.get("messages", [])[-1].content if state.get("messages") else ""
        
        try:
            # Invoke classifier
            result = classifier_agent.invoke(state)
            
            # Extract classification from agent response
            if result.get("messages"):
                last_message = result["messages"][-1]
                if isinstance(last_message, AIMessage):
                    # Try to parse classification from message
                    content = last_message.content
                    
                    # Look for JSON in the response
                    try:
                        # Try to find and parse JSON object in the content
                        start_idx = content.find('{')
                        end_idx = content.rfind('}') + 1
                        if start_idx != -1 and end_idx > start_idx:
                            classification_data = json.loads(content[start_idx:end_idx])
                            
                            # Update state with classification
                            result["classification"] = classification_data
                            
                            # Log the classification
                            logger.log_classification(
                                thread_id=thread_id,
                                classification=classification_data,
                                confidence=classification_data.get("confidence", 0.0),
                                user_message=user_message
                            )
                    except json.JSONDecodeError:
                        # If parsing fails, log with minimal info
                        logger.log_error(
                            thread_id=thread_id,
                            agent_name="classifier",
                            error_type="ParseError",
                            error_message="Failed to parse classification JSON from response",
                            recovery_action="Using default classification"
                        )
            
            return result
            
        except Exception as e:
            logger.log_error(
                thread_id=thread_id,
                agent_name="classifier",
                error_type=type(e).__name__,
                error_message=str(e),
                stack_trace=str(e),
                recovery_action="Returning state unchanged"
            )
            return state
    
    return logged_classifier


def wrap_resolver_with_logging(resolver_agent):
    """
    Wrap resolver agent to add logging.
    
    Args:
        resolver_agent: The compiled resolver agent
        
    Returns:
        Wrapped agent function with logging
    """
    def logged_resolver(state: UDAHubState) -> UDAHubState:
        thread_id = state.get("ticket_metadata", {}).get("ticket_id", "unknown")
        
        try:
            # Invoke resolver
            result = resolver_agent.invoke(state)
            
            # Extract resolution info
            if result.get("messages"):
                last_message = result["messages"][-1]
                if isinstance(last_message, AIMessage):
                    response = last_message.content
                    
                    # Calculate basic confidence (can be enhanced)
                    # For now, use a simple heuristic
                    confidence = 0.8 if len(response) > 100 else 0.5
                    resolved = confidence >= 0.7
                    
                    # Update state with resolution
                    resolution_data = {
                        "resolved": resolved,
                        "confidence": confidence,
                        "response": response
                    }
                    result["resolution"] = resolution_data
                    
                    # Log the resolution
                    logger.log_resolution(
                        thread_id=thread_id,
                        resolved=resolved,
                        confidence=confidence,
                        response=response,
                        rag_articles=result.get("rag_articles", [])
                    )
            
            return result
            
        except Exception as e:
            logger.log_error(
                thread_id=thread_id,
                agent_name="resolver",
                error_type=type(e).__name__,
                error_message=str(e),
                stack_trace=str(e),
                recovery_action="Returning state unchanged"
            )
            return state
    
    return logged_resolver


def wrap_tool_agent_with_logging(tool_agent):
    """
    Wrap tool agent to add logging.
    
    Args:
        tool_agent: The compiled tool agent
        
    Returns:
        Wrapped agent function with logging
    """
    def logged_tool_agent(state: UDAHubState) -> UDAHubState:
        thread_id = state.get("ticket_metadata", {}).get("ticket_id", "unknown")
        
        try:
            import time
            start_time = time.time()
            
            # Invoke tool agent
            result = tool_agent.invoke(state)
            
            execution_time = (time.time() - start_time) * 1000  # ms
            
            # Extract tool execution info
            if result.get("messages"):
                # Look for tool calls in messages
                for msg in result["messages"]:
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            logger.log_tool_execution(
                                thread_id=thread_id,
                                tool_name=tool_call.get('name', 'unknown'),
                                tool_input=tool_call.get('args', {}),
                                tool_output="Tool executed",
                                success=True,
                                execution_time=execution_time
                            )
            
            # Update state with tool results
            if not result.get("tool_results"):
                result["tool_results"] = {"executed": True, "timestamp": time.time()}
            
            return result
            
        except Exception as e:
            logger.log_error(
                thread_id=thread_id,
                agent_name="tool_agent",
                error_type=type(e).__name__,
                error_message=str(e),
                stack_trace=str(e),
                recovery_action="Returning state unchanged"
            )
            return state
    
    return logged_tool_agent


def wrap_escalation_with_logging(escalation_agent):
    """
    Wrap escalation agent to add logging.
    
    Args:
        escalation_agent: The compiled escalation agent
        
    Returns:
        Wrapped agent function with logging
    """
    def logged_escalation(state: UDAHubState) -> UDAHubState:
        thread_id = state.get("ticket_metadata", {}).get("ticket_id", "unknown")
        
        try:
            # Extract context before escalation
            classification = state.get("classification", {})
            resolution = state.get("resolution", {})
            
            attempted_steps = []
            if classification:
                attempted_steps.append(f"Classified as {classification.get('issue_type')}")
            if resolution:
                attempted_steps.append(f"Resolution attempted (confidence: {resolution.get('confidence', 0):.2f})")
            if state.get("tool_results"):
                attempted_steps.append("Database tools executed")
            
            # Determine escalation reason
            reason = "Unknown"
            if resolution and resolution.get("confidence", 0) < 0.7:
                reason = f"Low confidence resolution ({resolution.get('confidence', 0):.2f})"
            elif not resolution:
                reason = "Unable to resolve"
            elif state.get("user_requested_human"):
                reason = "User requested human agent"
            
            # Invoke escalation agent
            result = escalation_agent.invoke(state)
            
            # Log the escalation
            priority = classification.get("urgency", "medium") if classification else "medium"
            logger.log_escalation(
                thread_id=thread_id,
                reason=reason,
                priority=priority,
                context={
                    "classification": classification,
                    "resolution": resolution,
                    "tool_results": bool(state.get("tool_results"))
                },
                attempted_steps=attempted_steps
            )
            
            return result
            
        except Exception as e:
            logger.log_error(
                thread_id=thread_id,
                agent_name="escalation",
                error_type=type(e).__name__,
                error_message=str(e),
                stack_trace=str(e),
                recovery_action="Returning state unchanged"
            )
            return state
    
    return logged_escalation


__all__ = [
    "wrap_classifier_with_logging",
    "wrap_resolver_with_logging",
    "wrap_tool_agent_with_logging",
    "wrap_escalation_with_logging"
]
