# Workflow Documentation

## Overview

UDA-Hub uses a **Supervisor Pattern** with LangGraph to coordinate multiple specialized agents. This document explains the complete workflow from ticket ingestion to resolution or escalation.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Input                            │
│                  (Support Ticket)                        │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │   Supervisor Agent    │
         │  (Orchestrator)       │
         └───────┬───────────────┘
                 │
      ┌──────────┼──────────┬──────────┐
      │          │          │          │
      ▼          ▼          ▼          ▼
┌──────────┐ ┌────────┐ ┌────────┐ ┌──────────┐
│Classifier│ │Resolver│ │  Tool  │ │Escalation│
│  Agent   │ │ Agent  │ │ Agent  │ │  Agent   │
└────┬─────┘ └───┬────┘ └───┬────┘ └────┬─────┘
     │           │          │          │
     │      ┌────┴────┐     │          │
     │      │   RAG   │     │          │
     │      │ System  │     │          │
     │      └─────────┘     │          │
     │                 ┌────┴────┐     │
     │                 │CultPass │     │
     │                 │   DB    │     │
     │                 └─────────┘     │
     │                                 │
     └─────────────┬───────────────────┘
                   ▼
         ┌──────────────────┐
         │  User Response   │
         │ or Escalation    │
         └──────────────────┘
```

---

## Workflow States

### State Schema

```python
{
    "messages": [
        HumanMessage(content="User query"),
        AIMessage(content="Agent response"),
        ...
    ],
    "classification": {
        "issue_type": str,
        "confidence": str,  # high|medium|low
        "urgency": str,  # high|medium|low
        "category": str,
        "sentiment": str
    },
    "resolution": {
        "resolved": bool,
        "confidence": str,
        "answer": str,
        "rag_documents_used": int
    },
    "tool_results": [
        {"tool": str, "result": dict}
    ],
    "escalation": {
        "escalated": bool,
        "reason": str,
        "priority": str,
        "assigned_to": str
    },
    "ticket_metadata": {
        "ticket_id": str,
        "account_id": str,
        "user_email": str,
        "channel": str,
        "created_at": str
    }
}
```

---

## Workflow Steps

### Step 1: Ticket Ingestion

**Input**: User message via email, chat, phone, or app

**Process**:
1. Create ticket in UDA-Hub database
2. Generate unique ticket_id
3. Extract user information
4. Initialize state with message and metadata
5. Route to Supervisor

**State After Step 1**:
```python
{
    "messages": [HumanMessage(content="I can't login")],
    "ticket_metadata": {
        "ticket_id": "TKT-12345",
        "account_id": "cultpass",
        "user_email": "user@example.com",
        "channel": "email"
    }
}
```

---

### Step 2: Supervisor Routes to Classifier

**Supervisor Decision**: No classification exists → Route to Classifier

**Classifier Process**:
1. Analyze user message(s)
2. Use LLM to identify issue type
3. Determine confidence level
4. Assess urgency
5. Detect sentiment
6. Add classification to state

**State After Step 2**:
```python
{
    "messages": [HumanMessage(content="I can't login")],
    "classification": {
        "issue_type": "login",
        "confidence": "high",
        "urgency": "medium",
        "category": "technical",
        "sentiment": "neutral"
    },
    "ticket_metadata": {...}
}
```

**Logged Event**:
```json
{
    "event_type": "CLASSIFICATION",
    "ticket_id": "TKT-12345",
    "data": {
        "issue_type": "login",
        "confidence": "high",
        "processing_time_ms": 250
    }
}
```

---

### Step 3: Supervisor Routes Based on Classification

**Supervisor Decision Logic**:

```python
if user_requested_human(state):
    return "escalation"
elif needs_tool_execution(state):
    return "tool_agent"
elif classification.confidence == "high":
    return "resolver"
else:
    return "escalation"
```

**Possible Routes**:
- **High confidence, knowledge-based** → Resolver (RAG)
- **Requires data lookup** → Tool Agent
- **Low confidence or complex** → Escalation
- **User requested human** → Escalation

---

### Step 4A: Resolver Path (RAG-Based Resolution)

**When**: Issue can be answered from knowledge base

**Resolver Process**:
1. Extract query from user message
2. Invoke RAG retriever
3. Evaluate retrieved documents
4. Generate answer using LLM + context
5. Calculate confidence score
6. Add resolution to state

**RAG Retrieval**:
```python
# Query: "How do I reset my password?"
retrieved_docs = [
    {
        "page_content": "To reset your password: 1) Go to login page...",
        "metadata": {
            "title": "Password Reset Guide",
            "similarity_score": 0.92
        }
    }
]
```

**State After Resolver**:
```python
{
    "messages": [
        HumanMessage(content="I can't login"),
        AIMessage(content="To reset your password, go to...")
    ],
    "classification": {...},
    "resolution": {
        "resolved": True,
        "confidence": "high",
        "answer": "To reset your password...",
        "rag_documents_used": 2
    },
    "ticket_metadata": {...}
}
```

**Outcome**: 
- If confidence high → END (ticket resolved)
- If confidence low → Route to Escalation

---

### Step 4B: Tool Agent Path (Database Operations)

**When**: Issue requires user-specific data from database

**Tool Agent Process**:
1. Analyze what data is needed
2. Select appropriate tool(s)
3. Execute database queries
4. Format results
5. Generate response
6. Add tool_results to state

**Example: Booking Inquiry**

User: "Show me my upcoming reservations"

**Tool Selection**:
```python
# Tool Agent decides:
# 1. get_user_by_email("user@example.com")
# 2. get_user_reservations(user_id, status="upcoming")
```

**Tool Execution**:
```python
user = get_user_by_email("user@example.com")
# Returns: {"user_id": 1, "email": "user@example.com", ...}

reservations = get_user_reservations(user_id=1, status="upcoming")
# Returns: [
#     {"experience": "Yoga", "date": "2025-11-10", "time": "18:00"},
#     {"experience": "Gym", "date": "2025-11-12", "time": "07:00"}
# ]
```

**State After Tool Agent**:
```python
{
    "messages": [
        HumanMessage(content="Show my reservations"),
        AIMessage(content="You have 2 upcoming reservations: ...")
    ],
    "classification": {...},
    "tool_results": [
        {
            "tool": "get_user_reservations",
            "success": True,
            "data": [...]
        }
    ],
    "ticket_metadata": {...}
}
```

**Outcome**: Ticket resolved with data → END

---

### Step 4C: Escalation Path

**When**: 
- Low confidence resolution
- Complex issue (billing dispute, security)
- User explicitly requests human
- Multiple failed attempts
- Tool execution failed

**Escalation Agent Process**:
1. Determine escalation reason
2. Assess priority level
3. Select appropriate team
4. Prepare context summary
5. Create handoff message
6. Add escalation to state

**State After Escalation**:
```python
{
    "messages": [
        HumanMessage(content="I was charged twice!"),
        AIMessage(content="I'm connecting you with our billing team...")
    ],
    "classification": {"issue_type": "billing_dispute", "urgency": "high"},
    "escalation": {
        "escalated": True,
        "reason": "Complex billing dispute requires human review",
        "priority": "high",
        "assigned_to": "billing_team",
        "context_summary": "User reports duplicate charge..."
    },
    "ticket_metadata": {...}
}
```

**Outcome**: Human agent receives ticket with full context

---

## Multi-Turn Conversations

### Conversation Flow

**Turn 1**: Initial query
```python
User: "I have a problem with my account"
State: {"messages": [HumanMessage(...)]}
```

**Turn 2**: Clarification
```python
Agent: "What specific issue are you experiencing?"
User: "I can't book premium classes"
State: {"messages": [HumanMessage(), AIMessage(), HumanMessage()]}
```

**Turn 3**: Resolution
```python
Agent: "Let me check your subscription status..."
[Tool Agent checks subscription]
Agent: "Your subscription is active. Try logging out and back in."
```

### Memory Persistence

**Session Memory** (MemorySaver):
- Maintains conversation within a ticket
- Uses thread_id = ticket_id
- Preserves state across multiple exchanges

**Cross-Session Memory** (CustomerMemoryStore):
- Recalls previous tickets from same user
- Identifies recurring issues
- Personalizes responses

**Example**:
```python
# User returns with similar issue
previous_context = memory.get_user_history(user_email)
# Returns: [
#     {"ticket_id": "TKT-100", "issue": "login", "resolved": True},
#     {"ticket_id": "TKT-150", "issue": "login", "resolved": True}
# ]

# Agent: "I see you've had login issues before. Let me escalate 
# this to investigate why it keeps happening."
```

---

## Error Handling

### Classification Failure
```python
try:
    classification = classifier.invoke(state)
except Exception as e:
    logger.error("Classification failed", error=str(e))
    # Fallback: Route to human agent
    return escalation_agent.invoke(state)
```

### RAG Retrieval Failure
```python
try:
    documents = retriever.invoke(query)
except Exception as e:
    logger.error("RAG retrieval failed", error=str(e))
    # Fallback: Generic response + escalation
    return {
        "resolution": {"resolved": False, "confidence": "low"},
        "escalation_suggested": True
    }
```

### Database Tool Failure
```python
try:
    result = get_user_by_email(email)
except DatabaseError as e:
    logger.error("Database query failed", error=str(e))
    # Fallback: Apologize + escalate
    return {
        "tool_results": {"error": "Unable to access database"},
        "escalation": {"reason": "Technical failure"}
    }
```

---

## Routing Decision Tree

```
START
  │
  ├─ Has classification? NO → Classifier
  │                      YES ↓
  │
  ├─ User requested human? YES → Escalation
  │                        NO ↓
  │
  ├─ Confidence < 0.5? YES → Escalation
  │                    NO ↓
  │
  ├─ Needs database data? YES → Tool Agent
  │                       NO ↓
  │
  ├─ Has resolution attempt? NO → Resolver
  │                         YES ↓
  │
  ├─ Resolution confidence high? YES → END
  │                              NO → Escalation
  │
END (or Escalation)
```

---

## Performance Targets

**Response Times**:
- Classification: <1 second
- RAG retrieval: <2 seconds
- Database query: <1 second
- End-to-end (simple): <5 seconds
- End-to-end (complex): <10 seconds

**Success Metrics**:
- Auto-resolution rate: >70%
- Classification accuracy: >90%
- User satisfaction: >80%
- Escalation rate: <30%

---

## Implementation

**Main Workflow File**: `agentic/workflow.py`

**Key Components**:
```python
from langgraph.graph import StateGraph, START, END
from agentic.agents import create_classifier_agent, create_resolver_agent
from langgraph.checkpoint.memory import MemorySaver

# Build graph
workflow = StateGraph(UDAHubState)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("classifier", create_classifier_agent())
workflow.add_node("resolver", create_resolver_agent())
workflow.add_node("tool_agent", create_tool_agent())
workflow.add_node("escalation", create_escalation_agent())

# Add edges based on supervisor routing
workflow.add_conditional_edges("supervisor", route_decision)

# Compile with checkpointer
orchestrator = workflow.compile(checkpointer=MemorySaver())
```

**Execution**:
```python
state = {
    "messages": [HumanMessage(content="User query")],
    "ticket_metadata": {"ticket_id": "TKT-001"}
}

config = {"configurable": {"thread_id": "TKT-001"}}

result = orchestrator.invoke(state, config)
```

---

## Testing Workflow

See `solution/tests/test_workflow.py` for integration tests:
- Simple resolution path
- Tool execution path
- Escalation path
- Multi-turn conversations
- Error handling

**Example Test**:
```python
def test_workflow_simple_resolution():
    state = {"messages": [HumanMessage(content="Reset password")]}
    result = orchestrator.invoke(state, config)
    
    assert result["classification"]["issue_type"] == "login"
    assert result["resolution"]["resolved"] is True
    assert result["escalation"] is None
```

---

For more details, see:
- `AGENTS.md` - Individual agent documentation
- `TOOLS.md` - Tool specifications
- `agentic/workflow.py` - Implementation
- `agentic/design/DATA_FLOW.md` - Detailed flow diagrams
