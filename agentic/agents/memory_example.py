"""
Example Usage of Memory Capabilities in UDA-Hub

This file demonstrates how to use the memory features including:
- Short-term memory (MemorySaver checkpointer)
- Long-term memory (CustomerMemoryStore)
- State inspection (StateInspector)
"""

from langchain_core.messages import HumanMessage
from agentic.workflow import create_orchestrator
from agentic.inspector import StateInspector, inspect_thread_state
from agentic.memory import CustomerMemoryStore, get_memory_context_for_agent


def example_basic_memory():
    """Example: Basic short-term memory with MemorySaver."""
    print("\n" + "="*60)
    print("Example 1: Basic Short-Term Memory")
    print("="*60 + "\n")
    
    # Create orchestrator with default MemorySaver
    orchestrator = create_orchestrator()
    
    # First message
    ticket_id = "ticket_001"
    config = {"configurable": {"thread_id": ticket_id}}
    
    state1 = {
        "messages": [HumanMessage(content="I can't log into my account")],
        "ticket_metadata": {
            "ticket_id": ticket_id,
            "account_id": "cultpass",
            "user_id": "user_123",
            "channel": "chat"
        }
    }
    
    print("First message: 'I can't log into my account'")
    result1 = orchestrator.invoke(state1, config)
    print(f"Response received. Messages in state: {len(result1['messages'])}\n")
    
    # Follow-up message (same thread_id)
    state2 = {
        "messages": [HumanMessage(content="I already tried resetting my password")]
    }
    
    print("Follow-up: 'I already tried resetting my password'")
    result2 = orchestrator.invoke(state2, config)
    print(f"Response received. Messages in state: {len(result2['messages'])}")
    print("✓ Conversation history automatically persisted!\n")


def example_long_term_memory():
    """Example: Long-term memory with CustomerMemoryStore."""
    print("\n" + "="*60)
    print("Example 2: Long-Term Memory")
    print("="*60 + "\n")
    
    # Create orchestrator with long-term memory enabled
    orchestrator = create_orchestrator(
        enable_long_term_memory=True,
        db_path="data/core/udahub.db"
    )
    
    # Access memory store
    memory_store = orchestrator.memory_store
    
    # Get user history
    user_id = "user_123"
    history = memory_store.get_user_ticket_history(user_id, limit=5)
    print(f"User {user_id} ticket history: {len(history)} tickets")
    for ticket in history:
        print(f"  - {ticket['ticket_id']}: {ticket['issue_type']} ({ticket['status']})")
    
    # Get resolved issues
    resolved = memory_store.get_resolved_issues(user_id, limit=5)
    print(f"\nResolved issues: {len(resolved)} tickets")
    
    # Get user preferences
    preferences = memory_store.get_user_preferences(user_id)
    print(f"User preferences: {preferences}")
    
    # Set a preference
    memory_store.set_user_preference(user_id, "preferred_contact", "email")
    print("✓ Preference set: preferred_contact = email\n")


def example_memory_context():
    """Example: Building memory context for agents."""
    print("\n" + "="*60)
    print("Example 3: Memory Context for Agents")
    print("="*60 + "\n")
    
    memory_store = CustomerMemoryStore()
    
    # Get memory context for agent
    user_id = "user_123"
    context = get_memory_context_for_agent(
        user_id=user_id,
        current_issue_type="technical",
        current_tags=["login", "authentication"],
        memory_store=memory_store
    )
    
    print("Generated memory context for agent:")
    print(context)


def example_state_inspection():
    """Example: Inspecting workflow state."""
    print("\n" + "="*60)
    print("Example 4: State Inspection")
    print("="*60 + "\n")
    
    # Create orchestrator
    orchestrator = create_orchestrator()
    
    # Run a ticket through workflow
    ticket_id = "ticket_002"
    config = {"configurable": {"thread_id": ticket_id}}
    
    state = {
        "messages": [HumanMessage(content="How do I cancel my subscription?")],
        "ticket_metadata": {
            "ticket_id": ticket_id,
            "account_id": "cultpass",
            "user_id": "user_456",
            "channel": "email"
        }
    }
    
    result = orchestrator.invoke(state, config)
    
    # Inspect state
    inspector = StateInspector(orchestrator)
    
    # Get conversation history
    history = inspector.get_conversation_history(ticket_id)
    print(f"Conversation history: {len(history)} messages")
    
    # Get current state snapshot
    snapshot = inspector.get_state_snapshot(ticket_id)
    print(f"Current state:")
    print(f"  - Classification: {snapshot.get('classification')}")
    print(f"  - Resolution: {snapshot.get('resolution')}")
    print(f"  - Next node: {snapshot.get('next_node')}")
    
    # Get agent decisions
    decisions = inspector.get_agent_decisions(ticket_id)
    print(f"\nAgent decisions: {len(decisions)}")
    for decision in decisions[:3]:
        print(f"  Step {decision['step']}: {decision['agent']}")
    
    # Print full summary
    print("\nFull workflow summary:")
    inspector.print_workflow_summary(ticket_id)


def example_similar_issues():
    """Example: Finding similar past issues."""
    print("\n" + "="*60)
    print("Example 5: Finding Similar Issues")
    print("="*60 + "\n")
    
    memory_store = CustomerMemoryStore()
    
    user_id = "user_123"
    issue_type = "technical"
    tags = ["login", "password"]
    
    # Find similar past issues
    similar = memory_store.find_similar_issues(
        user_id=user_id,
        issue_type=issue_type,
        tags=tags,
        limit=5
    )
    
    print(f"Similar past issues for {user_id}:")
    for issue in similar:
        print(f"  - {issue['ticket_id']}: {issue['resolution_method']} "
              f"(confidence: {issue['confidence']}, overlap: {issue['tag_overlap']} tags)")


def example_recurring_issues():
    """Example: Identifying recurring issues."""
    print("\n" + "="*60)
    print("Example 6: Recurring Issues Detection")
    print("="*60 + "\n")
    
    memory_store = CustomerMemoryStore()
    
    user_id = "user_123"
    
    # Get recurring issues
    recurring = memory_store.get_recurring_issues(user_id, days_back=90)
    
    print(f"Recurring issues for {user_id} (last 90 days):")
    for issue_type, count in sorted(recurring.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {issue_type}: {count} times")
    
    if recurring:
        print("\n✓ Pattern detected! User may need proactive support.")


def example_store_resolution():
    """Example: Storing ticket resolution."""
    print("\n" + "="*60)
    print("Example 7: Storing Resolution")
    print("="*60 + "\n")
    
    memory_store = CustomerMemoryStore()
    
    # Store resolution
    memory_store.store_ticket_resolution(
        ticket_id="ticket_003",
        resolution_method="knowledge_base",
        confidence=0.85,
        articles_used="art_001,art_002",
        tool_results=None
    )
    print("✓ Resolution stored for ticket_003")
    
    # Store conversation summary
    memory_store.store_conversation_summary(
        ticket_id="ticket_003",
        summary="User couldn't login. Resolved by password reset instructions.",
        key_points="login issue, password reset, resolved",
        message_count=5
    )
    print("✓ Conversation summary stored")
    
    # Retrieve summary
    summary = memory_store.get_conversation_summary("ticket_003")
    if summary:
        print(f"\nRetrieved summary:")
        print(f"  Summary: {summary['summary']}")
        print(f"  Key points: {summary['key_points']}")
        print(f"  Messages: {summary['message_count']}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("UDA-Hub Memory Capabilities Examples")
    print("="*60)
    
    # Run examples
    # example_basic_memory()
    # example_long_term_memory()
    # example_memory_context()
    # example_state_inspection()
    # example_similar_issues()
    # example_recurring_issues()
    # example_store_resolution()
    
    print("\nTo run examples, uncomment the desired function calls above.")
    print("Note: Some examples require existing data in the database.\n")
