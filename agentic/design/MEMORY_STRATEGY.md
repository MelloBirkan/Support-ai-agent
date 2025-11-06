# Memory Strategy

This document describes the memory and state management implementation for UDA-Hub, covering both short-term (session-level) and long-term (cross-session) memory.

---

## Overview

UDA-Hub implements a **two-tier memory system**:

1. **Short-Term Memory**: Session-level conversation tracking using LangGraph checkpointers
2. **Long-Term Memory**: Cross-session customer history and preferences using persistent storage

This dual approach enables:
- Contextual conversations within a single ticket/session
- Customer knowledge retention across multiple interactions
- Personalized support based on history
- Pattern detection for proactive assistance

---

## Short-Term Memory (Session-Level)

### Implementation: MemorySaver Checkpointer

**Purpose**: Maintain conversation state within a single ticket/session

**Technology**: LangGraph's `MemorySaver` checkpointer

**Import**:
```python
from langgraph.checkpoint.memory import MemorySaver
```

### How It Works

**1. Graph Compilation with Checkpointer**

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph

# Create checkpointer
checkpointer = MemorySaver()

# Compile workflow with checkpointer
orchestrator = workflow.compile(checkpointer=checkpointer)
```

**2. Invocation with Thread ID**

```python
# Each ticket gets unique thread_id
thread_id = ticket_id  # e.g., "ticket_123"
config = {"configurable": {"thread_id": thread_id}}

# First invocation
result = orchestrator.invoke(
    {"messages": [HumanMessage(content="I can't login")]},
    config=config
)
```

**3. Automatic State Persistence**

- State automatically saved after each node execution
- Checkpointer stores full state snapshot
- Includes messages, metadata, agent outputs

**4. Subsequent Invocations Load Previous State**

```python
# Follow-up message (same thread_id)
result = orchestrator.invoke(
    {"messages": [HumanMessage(content="I tried resetting password")]},
    config=config  # Same thread_id loads previous context
)

# Previous messages automatically included
# Full conversation history available to agents
```

### State Schema

**Base State**: `MessagesState` from LangGraph

```python
from typing import Annotated
import operator
from langchain_core.messages import AnyMessage
from langgraph.graph import MessagesState

class UDAHubState(MessagesState):
    """State schema for UDA-Hub multi-agent system."""

    # Messages accumulated using operator.add
    messages: Annotated[list[AnyMessage], operator.add]

    # Other state fields (replaced on update)
    ticket_metadata: dict
    classification: dict | None = None
    resolution: dict | None = None
    tool_results: dict | None = None
    escalation: dict | None = None
```

**Key Points**:

1. **Messages Annotation**: `Annotated[list[AnyMessage], operator.add]`
   - Messages are accumulated, not replaced
   - Each agent adds to the message list
   - Full conversation history maintained

2. **Other Fields**: Replaced on update
   - `classification`: Latest classification
   - `resolution`: Latest resolution attempt
   - `tool_results`: Latest tool execution results

### State Persistence

**What Gets Saved**:
- All messages in conversation
- Ticket metadata
- Classification results
- Resolution attempts
- Tool execution results
- Escalation summaries

**When State is Saved**:
- After each node execution
- Before routing to next agent
- On workflow completion

**State Storage**:
- In-memory by default (MemorySaver)
- Can be persisted to SQLite (SqliteSaver)
- Can be persisted to Postgres (PostgresSaver)

### Thread ID Strategy

**Thread ID = Ticket ID**

**Why This Works**:
- Each ticket is a separate conversation
- All messages in a ticket share the same thread
- Multi-turn conversations maintain context automatically
- Thread persists until ticket is closed

**Example**:
```python
# Ticket created
ticket_id = "ticket_123"
thread_id = ticket_id

# First message
config = {"configurable": {"thread_id": "ticket_123"}}
orchestrator.invoke({"messages": [...]}, config)

# Follow-up messages use same thread_id
orchestrator.invoke({"messages": [...]}, config)

# All messages accessible in state["messages"]
```

### State Inspection

**Accessing Conversation History**:

```python
# Get current state
config = {"configurable": {"thread_id": "ticket_123"}}
current_state = orchestrator.get_state(config)

# Access messages
messages = current_state.values["messages"]
for msg in messages:
    print(f"{msg.type}: {msg.content}")

# Access metadata
ticket_metadata = current_state.values["ticket_metadata"]
classification = current_state.values["classification"]
```

**Get State History**:

```python
# Get all checkpoints for a thread
checkpoints = orchestrator.get_state_history(config)

for checkpoint in checkpoints:
    print(f"Checkpoint: {checkpoint.config}")
    print(f"Messages: {len(checkpoint.values['messages'])}")
```

**Use Cases for State Inspection**:
- Debugging agent decisions
- Reviewing conversation flow
- Analyzing resolution attempts
- Generating escalation summaries
- Monitoring performance

---

## Long-Term Memory (Cross-Session)

### Implementation: Database + Semantic Search

**Purpose**: Remember customer preferences, past issues, and resolutions across different tickets

**Technology**:
- SQLAlchemy for persistent storage
- Vector embeddings for semantic search
- In-memory store for cross-thread access (optional)

### Storage Strategy

#### 1. Ticket History Storage

**Tables Used** (from `data/models/udahub.py`):

1. **Ticket**: All tickets for a user
   - `ticket_id`, `user_id`, `account_id`, `channel`, `created_at`

2. **TicketMessage**: All messages across all tickets
   - `message_id`, `ticket_id`, `role`, `content`, `created_at`

3. **TicketMetadata**: Status and classification history
   - `ticket_id`, `status`, `main_issue_type`, `tags`, `urgency`

**Query Pattern: Get User's Previous Tickets**

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from data.models.udahub import Ticket, TicketMetadata
from datetime import datetime, timedelta

def get_user_ticket_history(user_id: str, limit: int = 5) -> list[dict]:
    """Get user's previous tickets."""
    engine = create_engine("sqlite:///data/core/udahub.db")
    Session = sessionmaker(bind=engine)
    session = Session()

    # Query tickets
    tickets = session.query(Ticket).filter(
        Ticket.user_id == user_id
    ).order_by(
        Ticket.created_at.desc()
    ).limit(limit).all()

    # Build result
    history = []
    for ticket in tickets:
        metadata = session.query(TicketMetadata).filter_by(
            ticket_id=ticket.ticket_id
        ).first()

        history.append({
            "ticket_id": ticket.ticket_id,
            "created_at": ticket.created_at,
            "status": metadata.status if metadata else "unknown",
            "issue_type": metadata.main_issue_type if metadata else None
        })

    session.close()
    return history
```

**Query Pattern: Get Resolved Issues**

```python
def get_resolved_issues(user_id: str, limit: int = 10) -> list[dict]:
    """Get user's previously resolved issues."""
    session = Session()

    resolved = session.query(Ticket, TicketMetadata).join(
        TicketMetadata, Ticket.ticket_id == TicketMetadata.ticket_id
    ).filter(
        Ticket.user_id == user_id,
        TicketMetadata.status == "resolved"
    ).order_by(
        Ticket.created_at.desc()
    ).limit(limit).all()

    results = []
    for ticket, metadata in resolved:
        results.append({
            "ticket_id": ticket.ticket_id,
            "issue_type": metadata.main_issue_type,
            "tags": metadata.tags,
            "created_at": ticket.created_at
        })

    session.close()
    return results
```

#### 2. Customer Preferences Storage

**Approach**: Store preferences as structured data

**Option A: Extend User Table**

Add `preferences` JSON field to User model:

```python
from sqlalchemy import JSON

class User(Base):
    __tablename__ = "users"
    # ... existing fields ...
    preferences = Column(JSON, nullable=True)
```

**Store Preferences**:
```python
user.preferences = {
    "preferred_contact": "email",
    "common_issues": ["login", "booking"],
    "subscription_tier": "premium"
}
```

**Option B: Separate Preferences Table**

```python
class UserPreference(Base):
    __tablename__ = "user_preferences"

    preference_id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.user_id"))
    preference_key = Column(String, nullable=False)
    preference_value = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.now)
```

**Store Preferences**:
```python
pref = UserPreference(
    preference_id=generate_id(),
    user_id="user_123",
    preference_key="preferred_contact",
    preference_value="email"
)
session.add(pref)
session.commit()
```

**Option C: In-Memory Store (LangGraph Platform)**

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()

# Store preference
store.put(
    namespace=(user_id, "preferences"),
    key="preferred_contact",
    value={"method": "email", "updated_at": datetime.now().isoformat()}
)

# Retrieve preferences
preferences = store.search(
    namespace=(user_id, "preferences"),
    query="contact preferences",
    limit=5
)
```

**Recommendation**: Option A (JSON field) for simplicity, Option C for LangGraph Platform deployment

### Semantic Search for Past Resolutions

**Purpose**: Find similar past issues to inform current resolution

**Implementation Steps**:

**Step 1: Create Embeddings for Resolved Tickets**

```python
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def create_ticket_embedding(ticket_id: str):
    """Create embedding for resolved ticket."""
    session = Session()

    # Get ticket details
    ticket = session.query(Ticket).filter_by(ticket_id=ticket_id).first()
    metadata = session.query(TicketMetadata).filter_by(ticket_id=ticket_id).first()
    messages = session.query(TicketMessage).filter_by(ticket_id=ticket_id).all()

    # Build text representation
    ticket_text = f"""
    Issue Type: {metadata.main_issue_type}
    Tags: {metadata.tags}
    Question: {messages[0].content}
    Resolution: {messages[-1].content if messages[-1].role == 'ai' else 'N/A'}
    """

    # Generate embedding
    embedding = embeddings.embed_query(ticket_text)

    session.close()
    return embedding
```

**Step 2: Search Similar Issues**

```python
def search_similar_tickets(
    user_id: str,
    current_issue: str,
    limit: int = 3
) -> list[dict]:
    """Find similar past tickets using semantic search."""

    # Get user's resolved tickets
    resolved = get_resolved_issues(user_id, limit=20)

    # Build vector store
    documents = []
    for ticket in resolved:
        doc = Document(
            page_content=f"{ticket['issue_type']} {ticket['tags']}",
            metadata={"ticket_id": ticket["ticket_id"]}
        )
        documents.append(doc)

    vectorstore = InMemoryVectorStore.from_documents(
        documents=documents,
        embedding=embeddings
    )

    # Search
    similar = vectorstore.similarity_search(
        query=current_issue,
        k=limit
    )

    return [doc.metadata["ticket_id"] for doc in similar]
```

**Step 3: Use in Resolution**

```python
def resolver_agent_with_history(state: MessagesState):
    user_id = state["ticket_metadata"]["user_id"]
    current_issue = state["messages"][0].content

    # Search similar past tickets
    similar_tickets = search_similar_tickets(
        user_id=user_id,
        current_issue=current_issue,
        limit=3
    )

    # Build context
    if similar_tickets:
        context = "Previous similar issues:\n"
        for ticket_id in similar_tickets:
            # Get resolution
            resolution = get_ticket_resolution(ticket_id)
            context += f"- {resolution}\n"

        # Include in prompt
        enhanced_prompt = f"{context}\n\nCurrent issue: {current_issue}"
    else:
        enhanced_prompt = current_issue

    # Proceed with resolution
    # ...
```

### Memory Retrieval in Agents

**Accessing Long-Term Memory in Node Functions**:

```python
from langgraph.store.base import BaseStore
from langgraph.config import RunnableConfig

def resolver_agent(
    state: MessagesState,
    config: RunnableConfig,
    *,
    store: BaseStore  # Access to in_memory_store
):
    """Resolver agent with access to long-term memory."""

    # Get user_id from state
    user_id = state["ticket_metadata"]["user_id"]

    # Retrieve past tickets (database)
    past_tickets = get_user_ticket_history(user_id, limit=5)

    # Retrieve preferences (in_memory_store)
    preferences = store.search(
        namespace=(user_id, "preferences"),
        query="support preferences",
        limit=3
    )

    # Use in resolution
    context = f"""
    User has {len(past_tickets)} previous tickets.
    Common issues: {[t['issue_type'] for t in past_tickets]}
    Preferences: {preferences}
    """

    # Include in LLM prompt
    # ...
```

**When to Use Long-Term Memory**:

1. **Returning Customers**: Load previous ticket history
2. **Recurring Issues**: Check if user has had similar problems
3. **Personalization**: Use preferences for communication style
4. **Pattern Detection**: Identify chronic issues for proactive support

---

## Memory Lifecycle

### Session Memory Lifecycle

```
Ticket Created
  │
  ▼
thread_id = ticket_id
  │
  ▼
Checkpointer initialized
  │
  ▼
Conversation flows through agents
  │ (state saved after each node)
  ▼
Checkpointer persists state
  │
  ▼
Ticket Resolved/Escalated
  │
  ▼
Final state saved
  │
  ▼
Thread remains accessible for inspection
  │ (optional)
  ▼
Thread archived after 30 days
```

### Long-Term Memory Lifecycle

```
Ticket Resolved
  │
  ▼
Extract key information:
  - Issue type
  - Resolution method
  - User satisfaction (if available)
  │
  ▼
Create memory entry:
  - Store in database
  - Generate embedding (optional)
  - Tag with metadata
  │
  ▼
Memory available for future tickets
  │
  ▼
Periodic cleanup:
  - Archive old tickets (>1 year)
  - Maintain embeddings for search
```

**Implementation**:

```python
def archive_ticket_to_memory(ticket_id: str):
    """Archive resolved ticket to long-term memory."""
    session = Session()

    # Get ticket
    ticket = session.query(Ticket).filter_by(ticket_id=ticket_id).first()
    metadata = session.query(TicketMetadata).filter_by(ticket_id=ticket_id).first()

    # Verify resolved
    if metadata.status != "resolved":
        return

    # Extract key info
    key_info = {
        "ticket_id": ticket_id,
        "user_id": ticket.user_id,
        "issue_type": metadata.main_issue_type,
        "tags": metadata.tags,
        "created_at": ticket.created_at
    }

    # Generate embedding (for semantic search)
    embedding = create_ticket_embedding(ticket_id)

    # Store (implementation depends on chosen storage)
    # ...

    session.close()
```

---

## Memory Integration Points

### 1. Supervisor Agent

**Uses Memory For**:
- Check if user has open tickets (prevent duplicates)
- Review past escalations (pattern detection)
- Load user preferences for routing decisions

**Implementation**:

```python
def supervisor(
    state: MessagesState,
    config: RunnableConfig,
    *,
    store: BaseStore
):
    user_id = state["ticket_metadata"]["user_id"]

    # Check for duplicate tickets
    open_tickets = get_open_tickets(user_id)
    if len(open_tickets) > 1:
        # Note in state
        state["ticket_metadata"]["duplicate_warning"] = True

    # Load preferences
    preferences = store.search(
        namespace=(user_id, "preferences"),
        query="routing preferences",
        limit=3
    )

    # Make routing decision with context
    # ...
```

### 2. Resolver Agent

**Uses Memory For**:
- Find similar past resolutions
- Check if user has recurring issues
- Personalize response based on history

**Implementation**:

```python
def resolver_agent(
    state: MessagesState,
    config: RunnableConfig,
    *,
    store: BaseStore
):
    user_id = state["ticket_metadata"]["user_id"]
    current_issue = state["messages"][0].content

    # Search similar past tickets
    similar_tickets = search_similar_tickets(
        user_id=user_id,
        current_issue=current_issue,
        limit=3
    )

    # Include in context for LLM
    if similar_tickets:
        context = f"User has had similar issues before:\n"
        for ticket_id in similar_tickets:
            context += f"- {ticket_id}\n"

        # Add to system prompt
        # ...
```

### 3. Escalation Agent

**Uses Memory For**:
- Include full ticket history in escalation summary
- Note recurring issues for human agent
- Provide context on past resolutions

**Implementation**:

```python
def escalation_agent(state: MessagesState):
    user_id = state["ticket_metadata"]["user_id"]

    # Get full history
    ticket_history = get_user_ticket_history(user_id, limit=10)

    # Count stats
    total_tickets = len(ticket_history)
    resolved = [t for t in ticket_history if t["status"] == "resolved"]
    escalated = [t for t in ticket_history if t["status"] == "escalated"]

    # Include in escalation summary
    summary = f"""
    User History:
    - Total tickets: {total_tickets}
    - Resolved: {len(resolved)}
    - Escalated: {len(escalated)}
    - Common issues: {get_common_issues(ticket_history)}
    """

    # ...
```

---

## Memory Configuration

### Workflow Compilation with Memory

**Basic Setup** (Short-term only):

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
orchestrator = workflow.compile(checkpointer=checkpointer)
```

**Advanced Setup** (Short-term + Long-term):

```python
from langgraph.checkpoint.memory import MemorySaver, InMemoryStore

checkpointer = MemorySaver()
in_memory_store = InMemoryStore()

orchestrator = workflow.compile(
    checkpointer=checkpointer,
    store=in_memory_store
)
```

**Production Setup** (with persistence):

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# Use SQLite checkpointer for persistence
checkpointer = SqliteSaver.from_conn_string("checkpoints.db")
orchestrator = workflow.compile(checkpointer=checkpointer)
```

### Configuration for LangGraph Platform

**langgraph.json**:

```json
{
  "store": {
    "index": {
      "embed": "openai:text-embedding-3-small",
      "dims": 1536,
      "fields": ["$"]
    }
  }
}
```

This enables:
- Semantic search in in_memory_store
- Automatic embedding generation
- Cross-thread memory access

---

## Memory Best Practices

### 1. Context Window Management

**Problem**: LLM context windows have limits

**Solution**:
- Keep active context under 10 messages
- Summarize older messages if conversation exceeds limit
- Store full history in database for reference

**Implementation**:

```python
def truncate_messages(messages: list[AnyMessage], max_messages: int = 10) -> list[AnyMessage]:
    if len(messages) <= max_messages:
        return messages

    # Keep recent messages
    recent = messages[-max_messages:]

    # Summarize older messages
    older = messages[:-max_messages]
    summary = summarize_messages(older)

    return [
        SystemMessage(content=f"Previous conversation summary: {summary}"),
        *recent
    ]
```

### 2. Privacy and Data Retention

**Considerations**:
- Anonymize sensitive data in long-term memory
- Implement data retention policies (e.g., delete after 1 year)
- Comply with GDPR/privacy regulations

**Implementation**:

```python
def anonymize_ticket(ticket_id: str):
    """Anonymize sensitive data in ticket."""
    session = Session()

    messages = session.query(TicketMessage).filter_by(ticket_id=ticket_id).all()
    for msg in messages:
        # Remove PII
        msg.content = remove_pii(msg.content)

    session.commit()
    session.close()

def cleanup_old_tickets():
    """Delete tickets older than 1 year."""
    cutoff = datetime.now() - timedelta(days=365)

    session = Session()
    old_tickets = session.query(Ticket).filter(
        Ticket.created_at < cutoff
    ).all()

    for ticket in old_tickets:
        session.delete(ticket)

    session.commit()
    session.close()
```

### 3. Performance Optimization

**Strategies**:
- Cache frequently accessed memories
- Use database indexes on user_id and created_at
- Limit semantic search to recent tickets (e.g., last 6 months)

**Implementation**:

```python
# Database indexes
CREATE INDEX idx_tickets_user_id ON tickets(user_id);
CREATE INDEX idx_tickets_created_at ON tickets(created_at);
CREATE INDEX idx_ticket_metadata_status ON ticket_metadata(status);

# Caching
from functools import lru_cache

@lru_cache(maxsize=100)
def get_user_ticket_history_cached(user_id: str, limit: int = 5):
    return get_user_ticket_history(user_id, limit)
```

### 4. Memory Quality

**Guidelines**:
- Only store resolved tickets in long-term memory
- Validate memory entries before storage
- Periodically review and clean up stale memories

**Implementation**:

```python
def validate_memory_entry(ticket_id: str) -> bool:
    """Validate ticket before storing in memory."""
    session = Session()

    metadata = session.query(TicketMetadata).filter_by(ticket_id=ticket_id).first()

    # Must be resolved
    if metadata.status != "resolved":
        return False

    # Must have classification
    if not metadata.main_issue_type:
        return False

    session.close()
    return True
```

---

## Testing Memory

### Short-Term Memory Tests

```python
def test_multi_turn_conversation():
    """Test that messages persist across invocations."""
    config = {"configurable": {"thread_id": "test_123"}}

    # First message
    result1 = orchestrator.invoke(
        {"messages": [HumanMessage(content="Hello")]},
        config
    )

    # Second message
    result2 = orchestrator.invoke(
        {"messages": [HumanMessage(content="Follow-up")]},
        config
    )

    # Verify message accumulation
    assert len(result2["messages"]) > len(result1["messages"])

def test_state_persistence():
    """Test that state is persisted."""
    config = {"configurable": {"thread_id": "test_456"}}

    # Invoke workflow
    orchestrator.invoke(
        {"messages": [...], "ticket_metadata": {"ticket_id": "test_456"}},
        config
    )

    # Get state
    state = orchestrator.get_state(config)

    # Verify state exists
    assert state.values["ticket_metadata"] is not None
```

### Long-Term Memory Tests

```python
def test_store_and_retrieve_preference():
    """Test preference storage."""
    user_id = "test_user"

    # Store preference
    store.put(
        namespace=(user_id, "preferences"),
        key="test_key",
        value={"value": "test"}
    )

    # Retrieve
    result = store.search(
        namespace=(user_id, "preferences"),
        query="test"
    )

    assert len(result) > 0

def test_similar_ticket_search():
    """Test semantic search for similar tickets."""
    user_id = "test_user"

    # Search
    similar = search_similar_tickets(
        user_id=user_id,
        current_issue="login issue",
        limit=3
    )

    assert len(similar) <= 3
    assert isinstance(similar, list)
```

---

## Related Documentation

- **System Overview**: See `ARCHITECTURE.md`
- **Agent Details**: See `AGENT_SPECIFICATIONS.md`
- **Data Flow**: See `DATA_FLOW.md`
- **RAG**: See `RAG_IMPLEMENTATION.md`
- **Diagrams**: See `DIAGRAMS.md`
