"""
State Inspector Utility for UDA-Hub

Provides utilities for inspecting workflow state, conversation history,
agent decisions, and tool usage by thread_id.
"""

from typing import Optional, Any
from datetime import datetime
from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import AnyMessage


class StateInspector:
    """Inspector for examining workflow state and history."""
    
    def __init__(self, orchestrator: CompiledStateGraph):
        """Initialize state inspector.
        
        Args:
            orchestrator: Compiled workflow graph
        """
        self.orchestrator = orchestrator
    
    def get_conversation_history(
        self,
        thread_id: str
    ) -> list[dict]:
        """Get conversation history for a thread.
        
        Args:
            thread_id: Thread/ticket identifier
            
        Returns:
            List of message dictionaries with role and content
        """
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            state = self.orchestrator.get_state(config)
            messages = state.values.get("messages", [])
            
            history = []
            for msg in messages:
                history.append({
                    "role": msg.type,
                    "content": msg.content,
                    "timestamp": getattr(msg, "timestamp", None)
                })
            
            return history
        except Exception as e:
            print(f"Error retrieving conversation history: {e}")
            return []
    
    def get_state_snapshot(
        self,
        thread_id: str
    ) -> dict:
        """Get current state snapshot for a thread.
        
        Args:
            thread_id: Thread/ticket identifier
            
        Returns:
            Dictionary containing current state values
        """
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            state = self.orchestrator.get_state(config)
            
            return {
                "messages": len(state.values.get("messages", [])),
                "ticket_metadata": state.values.get("ticket_metadata"),
                "classification": state.values.get("classification"),
                "resolution": state.values.get("resolution"),
                "tool_results": state.values.get("tool_results"),
                "escalation": state.values.get("escalation"),
                "next_node": state.next,
                "checkpoint_id": state.config.get("configurable", {}).get("checkpoint_id")
            }
        except Exception as e:
            print(f"Error retrieving state snapshot: {e}")
            return {}
    
    def get_state_history(
        self,
        thread_id: str,
        limit: Optional[int] = None
    ) -> list[dict]:
        """Get state history (checkpoints) for a thread.
        
        Args:
            thread_id: Thread/ticket identifier
            limit: Maximum number of checkpoints to return
            
        Returns:
            List of checkpoint dictionaries
        """
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            checkpoints = []
            for checkpoint in self.orchestrator.get_state_history(config):
                checkpoints.append({
                    "checkpoint_id": checkpoint.config.get("configurable", {}).get("checkpoint_id"),
                    "messages_count": len(checkpoint.values.get("messages", [])),
                    "classification": checkpoint.values.get("classification"),
                    "resolution": checkpoint.values.get("resolution"),
                    "next_node": checkpoint.next,
                    "parent_checkpoint": checkpoint.parent_config
                })
                
                if limit and len(checkpoints) >= limit:
                    break
            
            return checkpoints
        except Exception as e:
            print(f"Error retrieving state history: {e}")
            return []
    
    def get_agent_decisions(
        self,
        thread_id: str
    ) -> list[dict]:
        """Extract agent decision history from checkpoints.
        
        Args:
            thread_id: Thread/ticket identifier
            
        Returns:
            List of agent decision dictionaries
        """
        history = self.get_state_history(thread_id)
        
        decisions = []
        for i, checkpoint in enumerate(history):
            if checkpoint.get("classification"):
                decisions.append({
                    "step": i,
                    "agent": "classifier",
                    "decision": checkpoint["classification"]
                })
            
            if checkpoint.get("resolution"):
                decisions.append({
                    "step": i,
                    "agent": "resolver",
                    "decision": checkpoint["resolution"]
                })
            
            if checkpoint.get("next_node"):
                decisions.append({
                    "step": i,
                    "agent": "supervisor",
                    "decision": {"next_node": checkpoint["next_node"]}
                })
        
        return decisions
    
    def get_tool_usage(
        self,
        thread_id: str
    ) -> list[dict]:
        """Get tool execution history for a thread.
        
        Args:
            thread_id: Thread/ticket identifier
            
        Returns:
            List of tool usage dictionaries
        """
        history = self.get_state_history(thread_id)
        
        tool_usage = []
        for i, checkpoint in enumerate(history):
            if checkpoint.get("tool_results"):
                tool_usage.append({
                    "step": i,
                    "results": checkpoint["tool_results"]
                })
        
        return tool_usage
    
    def get_full_workflow_trace(
        self,
        thread_id: str
    ) -> dict:
        """Get complete workflow trace with all details.
        
        Args:
            thread_id: Thread/ticket identifier
            
        Returns:
            Dictionary with complete trace information
        """
        return {
            "thread_id": thread_id,
            "conversation_history": self.get_conversation_history(thread_id),
            "current_state": self.get_state_snapshot(thread_id),
            "state_history": self.get_state_history(thread_id),
            "agent_decisions": self.get_agent_decisions(thread_id),
            "tool_usage": self.get_tool_usage(thread_id)
        }
    
    def print_workflow_summary(
        self,
        thread_id: str
    ) -> None:
        """Print human-readable workflow summary.
        
        Args:
            thread_id: Thread/ticket identifier
        """
        trace = self.get_full_workflow_trace(thread_id)
        
        print(f"\n{'='*60}")
        print(f"Workflow Summary for Thread: {thread_id}")
        print(f"{'='*60}\n")
        
        # Conversation
        print(f"Conversation Messages: {len(trace['conversation_history'])}")
        for i, msg in enumerate(trace['conversation_history'][:3]):
            content = msg['content'][:50] + "..." if len(msg['content']) > 50 else msg['content']
            print(f"  [{i+1}] {msg['role']}: {content}")
        if len(trace['conversation_history']) > 3:
            print(f"  ... and {len(trace['conversation_history']) - 3} more messages")
        
        # Current State
        print(f"\nCurrent State:")
        current = trace['current_state']
        if current.get('classification'):
            print(f"  Classification: {current['classification'].get('issue_type')} "
                  f"(confidence: {current['classification'].get('confidence', 0):.2f})")
        if current.get('resolution'):
            print(f"  Resolution: {'Resolved' if current['resolution'].get('resolved') else 'Unresolved'} "
                  f"(confidence: {current['resolution'].get('confidence', 0):.2f})")
        if current.get('next_node'):
            print(f"  Next Node: {current['next_node']}")
        
        # Agent Decisions
        print(f"\nAgent Decisions: {len(trace['agent_decisions'])}")
        for decision in trace['agent_decisions'][:5]:
            print(f"  Step {decision['step']}: {decision['agent']} -> {decision['decision']}")
        
        # Tool Usage
        if trace['tool_usage']:
            print(f"\nTool Executions: {len(trace['tool_usage'])}")
            for usage in trace['tool_usage']:
                print(f"  Step {usage['step']}: Tools executed")
        
        print(f"\n{'='*60}\n")


def inspect_thread_state(
    orchestrator: CompiledStateGraph,
    thread_id: str,
    verbose: bool = True
) -> dict:
    """Convenience function to inspect thread state.
    
    Args:
        orchestrator: Compiled workflow graph
        thread_id: Thread/ticket identifier
        verbose: Print summary if True
        
    Returns:
        Complete workflow trace dictionary
    """
    inspector = StateInspector(orchestrator)
    
    if verbose:
        inspector.print_workflow_summary(thread_id)
    
    return inspector.get_full_workflow_trace(thread_id)


def get_thread_messages(
    orchestrator: CompiledStateGraph,
    thread_id: str
) -> list[AnyMessage]:
    """Get raw message objects for a thread.
    
    Args:
        orchestrator: Compiled workflow graph
        thread_id: Thread/ticket identifier
        
    Returns:
        List of message objects
    """
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        state = orchestrator.get_state(config)
        return state.values.get("messages", [])
    except Exception:
        return []


def get_thread_classification(
    orchestrator: CompiledStateGraph,
    thread_id: str
) -> Optional[dict]:
    """Get classification for a thread.
    
    Args:
        orchestrator: Compiled workflow graph
        thread_id: Thread/ticket identifier
        
    Returns:
        Classification dictionary or None
    """
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        state = orchestrator.get_state(config)
        return state.values.get("classification")
    except Exception:
        return None


def get_thread_resolution(
    orchestrator: CompiledStateGraph,
    thread_id: str
) -> Optional[dict]:
    """Get resolution for a thread.
    
    Args:
        orchestrator: Compiled workflow graph
        thread_id: Thread/ticket identifier
        
    Returns:
        Resolution dictionary or None
    """
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        state = orchestrator.get_state(config)
        return state.values.get("resolution")
    except Exception:
        return None


__all__ = [
    "StateInspector",
    "inspect_thread_state",
    "get_thread_messages",
    "get_thread_classification",
    "get_thread_resolution"
]
