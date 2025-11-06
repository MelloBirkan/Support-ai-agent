# Data Flow and Decision Routing

This document describes the complete flow from ticket ingestion to resolution or escalation, including all routing logic and state transitions.

---

## Overview

UDA-Hub uses a supervisor-based routing pattern where the central Supervisor Agent makes all routing decisions based on:
- Ticket state (new, classified, resolving, etc.)
- Agent outputs (classification, resolution, tool results)
- Confidence scores (from Resolver Agent)
- User requests (explicit escalation)

---

## Ticket Ingestion Flow

### 1. Initial Ticket Creation

**Input Sources**:
- External support systems (Zendesk, Intercom, Freshdesk)
- Internal CRM systems
- Direct API calls
- Email forwarding
- Chat integrations

**Incoming Ticket Structure**:
```python
{
    "ticket_id": str,                    # Unique identifier
    "account_id": str,                   # Customer account (e.g., "cultpass")
    "user_id": str,                      # UDA-Hub user ID
    "external_user_id": str,             # Maps to CultPass user_id
    "channel": str,                      # "email", "chat", "api"
    "content": str,                      # User's message/question
    "metadata": {
        "urgency": str | None,           # Optional urgency indicator
        "tags": list[str] | None,        # Optional pre-tags
        "history": list[str] | None      # Previous ticket IDs
    }
}
```

### 2. Database Operations

**Step 1: Create Ticket Entry**
```sql
INSERT INTO tickets (ticket_id, account_id, user_id, channel, created_at)
VALUES (?, ?, ?, ?, ?);
```

**Step 2: Create Ticket Metadata**
```sql
INSERT INTO ticket_metadata (ticket_id, status, urgency, main_issue_type, tags)
VALUES (?, 'new', ?, ?, ?);
```

**Step 3: Create Initial Message**
```sql
INSERT INTO ticket_messages (message_id, ticket_id, role, content, created_at)
VALUES (?, ?, 'user', ?, ?);
```

### 3. Initialize State

**Thread ID Assignment**:
```python
thread_id = ticket_id  # Each ticket is a separate thread
config = {"configurable": {"thread_id": thread_id}}
```

**Initial State**:
```python
initial_state = {
    "messages": [
        HumanMessage(content=ticket.content)
    ],
    "ticket_metadata": {
        "ticket_id": ticket.ticket_id,
        "account_id": ticket.account_id,
        "user_id": ticket.user_id,
        "channel": ticket.channel,
        "urgency": ticket.metadata.get("urgency")
    },
    "classification": None,
    "resolution": None,
    "tool_results": None,
    "escalation": None
}
```

### 4. Invoke Workflow

```python
result = orchestrator.invoke(initial_state, config)
```

---

## Agent Routing Flow

### Complete Flow Diagram

```
START
  │
  ▼
┌─────────────────────┐
│ Supervisor (Initial) │
│ Analyze: New ticket  │
└──────────┬──────────┘
           │
           ▼
  ┌────────────────┐
  │ Classifier     │
  │ - Categorize   │
  │ - Assess urgency│
  └────────┬───────┘
           │ Returns classification
           ▼
┌──────────────────────────┐
│ Supervisor (Post-Class)  │
│ Decision: Route to       │
│ Resolver                 │
└──────────┬───────────────┘
           │
           ▼
  ┌────────────────────┐
  │ Resolver           │
  │ - Use RAG tool     │
  │ - Generate answer  │
  │ - Calculate conf.  │
  └──────┬──────┬──────┘
         │      │
         │      ▼ (if needs data)
         │  ┌──────────────┐
         │  │ Tool Agent   │
         │  │ - Query DB   │
         │  └──────┬───────┘
         │         │
         │         ▼
         │  ┌──────────────────┐
         │  │ Supervisor       │
         │  │ Route back to    │
         │  │ Resolver w/ data │
         │  └──────────────────┘
         │
         ▼ Returns resolution
┌──────────────────────────┐
│ Supervisor (Post-Resolve)│
│ Check confidence         │
└──────┬──────────┬────────┘
       │          │
       │          ▼ (if confidence < 0.7)
       │     ┌─────────────────┐
       │     │ Escalation      │
       │     │ - Summarize     │
       │     │ - Assign priority│
       │     └────────┬────────┘
       │              │
       │              ▼
       │         END (Escalated)
       │
       ▼ (if confidence >= 0.7)
   END (Resolved)
```

---

## Detailed Routing Logic

### Supervisor Decision Tree

**Decision Point 1: Initial Routing**

```python
def supervisor_initial_decision(state: UDAHubState) -> str:
    """Determine routing for newly ingested ticket."""

    # Check if classification exists
    if state.get("classification") is None:
        return "classifier"  # Route to Classifier Agent

    # Classification exists, check for resolution
    if state.get("resolution") is None:
        return "resolver"  # Route to Resolver Agent

    # Resolution exists, check confidence
    resolution = state["resolution"]
    if resolution["confidence"] >= 0.7:
        return "END"  # Ticket resolved

    # Low confidence, escalate
    return "escalation"
```

**Decision Point 2: Post-Resolver Routing**

```python
def supervisor_post_resolver_decision(state: UDAHubState) -> str:
    """Determine routing after Resolver attempt."""

    resolution = state["resolution"]

    # Check if resolver needs data
    if "need_data" in resolution:
        return "tool_agent"  # Route to Tool Agent

    # Check confidence
    if resolution["confidence"] >= 0.7:
        return "END"  # Successfully resolved

    # Low confidence
    if resolution["confidence"] < 0.7:
        return "escalation"  # Route to Escalation Agent

    # No relevant knowledge found
    if len(resolution["articles_used"]) == 0:
        return "escalation"
```

**Decision Point 3: Post-Tool Routing**

```python
def supervisor_post_tool_decision(state: UDAHubState) -> str:
    """Determine routing after Tool Agent execution."""

    tool_results = state["tool_results"]

    # Check for errors
    if "error" in tool_results:
        # If critical error, escalate
        if tool_results["error"] == "database_unavailable":
            return "escalation"

    # Data retrieved successfully, return to resolver
    return "resolver"
```

**Decision Point 4: Escalation Conditions**

```python
def should_escalate(state: UDAHubState) -> bool:
    """Determine if ticket should be escalated."""

    # Explicit user request
    for msg in state["messages"]:
        if "human agent" in msg.content.lower():
            return True

    # Low resolver confidence
    if state.get("resolution"):
        if state["resolution"]["confidence"] < 0.7:
            return True

    # No relevant knowledge
    if state.get("resolution"):
        if len(state["resolution"]["articles_used"]) == 0:
            return True

    # Multiple failed attempts
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
```

---

## State Transitions

### Ticket Status Flow

```
new → classifying → classified → resolving → resolved
                                            ↓
                                       escalated
```

**Status Definitions**:

1. **new**: Ticket created, no processing yet
   - Initial state after ingestion
   - Waiting for Supervisor routing

2. **classifying**: Classifier Agent is analyzing
   - Set when Supervisor routes to Classifier
   - Classifier analyzing content

3. **classified**: Classification complete
   - Classifier returned results
   - Ready for resolution

4. **resolving**: Resolver Agent attempting resolution
   - Set when Supervisor routes to Resolver
   - May involve RAG queries and tool calls

5. **resolved**: Successfully resolved by AI
   - High confidence answer provided
   - Ticket closed

6. **escalated**: Requires human intervention
   - Low confidence or complex issue
   - Assigned to human agent queue

### Status Update Timing

**Database Updates**:
```python
# Update status after each agent
def update_ticket_status(ticket_id: str, new_status: str):
    with get_session("data/core/udahub.db") as session:
        metadata = session.query(TicketMetadata).filter_by(
            ticket_id=ticket_id
        ).first()
        metadata.status = new_status
        metadata.updated_at = datetime.now()
        session.commit()

# Create system message for status change
def log_status_change(ticket_id: str, old_status: str, new_status: str):
    with get_session("data/core/udahub.db") as session:
        message = TicketMessage(
            message_id=generate_id(),
            ticket_id=ticket_id,
            role="system",
            content=f"Status changed: {old_status} → {new_status}",
            created_at=datetime.now()
        )
        session.add(message)
        session.commit()
```

---

## Message Flow

### Message Types and Roles

**From `data/models/udahub.py`**:
```python
class RoleEnum(str, Enum):
    user = "user"        # Messages from end user
    agent = "agent"      # Messages from human agents
    ai = "ai"            # Messages from AI agents
    system = "system"    # System messages (status, routing)
```

### Message Sequence Examples

**Example 1: Successful Resolution**

```
1. [user] "I can't log into my CultPass account"
   → ticket_id: "ticket_001"
   → Stored in TicketMessage

2. [system] "Status: new → classifying"
   → Supervisor routes to Classifier

3. [ai] "Classification: type=technical, urgency=high, confidence=0.92"
   → Classifier analysis complete

4. [system] "Status: classifying → classified"
   → Classification stored in TicketMetadata

5. [system] "Status: classified → resolving"
   → Supervisor routes to Resolver

6. [ai] "Searching knowledge base for login issues..."
   → Resolver invokes RAG tool

7. [ai] "Try tapping 'Forgot Password' on the login screen. Make sure you're using the email associated with your account. If the email doesn't arrive, check spam or try again in a few minutes."
   → Resolver generates answer
   → Confidence: 0.85

8. [system] "Status: resolving → resolved"
   → Supervisor marks as resolved
   → END

9. [system] "Ticket resolved with confidence 0.85"
   → Final status update
```

**Example 2: Escalation Flow**

```
1. [user] "I want a refund for my cancelled reservation"
   → ticket_id: "ticket_002"

2. [system] "Status: new → classifying"

3. [ai] "Classification: type=billing, urgency=medium, confidence=0.88"

4. [system] "Status: classifying → classified"

5. [system] "Status: classified → resolving"

6. [ai] "Searching knowledge base for refund policy..."

7. [ai] "Article found: 'Refund Policy' - Refunds require approval"
   → Confidence: 0.45 (low - policy exception needed)

8. [system] "Status: resolving → escalated"
   → Supervisor routes to Escalation

9. [ai] "ESCALATION SUMMARY:
   Issue: Refund request for cancelled reservation
   Reason: Refund policy requires human approval
   Priority: P2 (High - financial impact)
   Recommendation: Review refund request and approve/deny"

10. [system] "Ticket escalated to human agent queue"
    → END (escalated)
```

---

## Tool Invocation Flow

### When Tools Are Called

**Scenarios Requiring Tool Agent**:

1. **User Information Lookup**
   - User asks about their account status
   - Need to verify user identity
   - Check if user is blocked

2. **Subscription Status Check**
   - User asks about subscription details
   - Verify tier benefits
   - Check quota usage

3. **Experience Search**
   - User asks about available events
   - Check event availability
   - Filter by preferences

4. **Reservation Management**
   - User wants to see bookings
   - Check reservation status
   - Cancel reservation

### Tool Agent Workflow

```
1. Resolver determines need for data
   └─> Sets "need_data" flag in resolution

2. Supervisor detects need_data flag
   └─> Routes to Tool Agent

3. Tool Agent receives request
   └─> Analyzes request to select appropriate tool

4. Tool selection logic:
   - "check subscription" → subscription_management_tool
   - "find events" → experience_search_tool
   - "my reservations" → reservation_management_tool
   - "user info" → user_lookup_tool

5. Tool executes database query
   └─> Query CultPass DB using SQLAlchemy

6. Tool returns structured result
   └─> Added to state.tool_results

7. Supervisor receives tool results
   └─> Routes back to Resolver with data

8. Resolver uses data to generate final response
   └─> Includes tool data in answer
```

### Tool Selection Examples

**Example 1: Subscription Check**
```python
# User message: "What's my subscription status?"

# Resolver analysis:
{
    "need_data": True,
    "required_tool": "subscription_management",
    "parameters": {
        "user_id": "cultpass_user_123"
    }
}

# Tool Agent executes:
result = subscription_management_tool.invoke({
    "user_id": "cultpass_user_123"
})

# Returns:
{
    "status": "active",
    "tier": "premium",
    "monthly_quota": 4,
    "end_date": "2025-12-31"
}

# Resolver uses data:
"Your subscription is active with Premium tier. You have 4 experiences per month, and your subscription is valid until December 31, 2025."
```

**Example 2: Experience Search**
```python
# User message: "Are there any premium yoga events this weekend?"

# Resolver analysis:
{
    "need_data": True,
    "required_tool": "experience_search",
    "parameters": {
        "search_query": "yoga",
        "is_premium": True,
        "date_range": ("2025-11-08", "2025-11-10")
    }
}

# Tool Agent executes:
result = experience_search_tool.invoke({
    "search_query": "yoga",
    "is_premium": True
})

# Resolver uses data:
"I found 3 premium yoga experiences this weekend: ..."
```

---

## Confidence Scoring and Thresholds

### Confidence Calculation

**Formula**:
```python
confidence = (
    0.4 * retrieval_similarity_score +
    0.3 * answer_completeness_score +
    0.2 * article_quality_score +
    0.1 * context_match_score
)
```

**Factor Breakdown**:

1. **Retrieval Similarity** (40%)
   - Average of vector similarity scores from RAG
   - Range: 0.0 to 1.0
   - High: 0.8-1.0 (strong semantic match)
   - Medium: 0.6-0.79 (partial match)
   - Low: 0.0-0.59 (weak match)

2. **Answer Completeness** (30%)
   - Does answer address all parts of question?
   - Full answer (>50 chars): 1.0
   - Partial answer (20-50 chars): 0.7
   - Minimal answer (<20 chars): 0.3

3. **Article Quality** (20%)
   - Does article have suggested phrasing? 1.0
   - Article has tags? 0.7
   - Basic article: 0.5

4. **Context Match** (10%)
   - Classification matches article tags? 1.0
   - Partial match: 0.5

### Routing Thresholds

**Threshold Decision Table**:

| Confidence | Action | Description |
|-----------|--------|-------------|
| ≥ 0.8 | Resolve immediately | High confidence, direct answer |
| 0.7-0.79 | Resolve with disclaimer | Medium confidence, answer with caveat |
| 0.5-0.69 | Escalate with summary | Low confidence, provide context |
| < 0.5 | Escalate immediately | Very low confidence, no good answer |

**Implementation**:
```python
def route_based_on_confidence(confidence: float) -> str:
    if confidence >= 0.8:
        return "END"  # Resolve
    elif confidence >= 0.7:
        return "END"  # Resolve with disclaimer
    elif confidence >= 0.5:
        return "escalation"  # Escalate with summary
    else:
        return "escalation"  # Escalate immediately
```

---

## Error Handling and Fallbacks

### Error Scenarios

**1. RAG Retrieval Fails**

```python
try:
    articles = retriever_tool.invoke(query)
except Exception as e:
    logger.error(f"RAG retrieval failed: {e}")
    return {
        "error": "knowledge_base_unavailable",
        "action": "escalate",
        "escalation_reason": "Knowledge base temporarily unavailable"
    }
```

**2. Tool Execution Fails**

```python
try:
    result = tool.invoke(params)
except DatabaseError as e:
    logger.error(f"Database error: {e}")
    # Retry once
    try:
        time.sleep(1)
        result = tool.invoke(params)
    except:
        return {
            "error": "database_unavailable",
            "action": "escalate",
            "escalation_reason": "Unable to access user data"
        }
```

**3. LLM Timeout**

```python
try:
    response = agent.invoke(state, timeout=30)
except TimeoutError:
    logger.error("Agent timeout")
    # Retry with shorter context
    truncated_state = truncate_messages(state, max_messages=5)
    try:
        response = agent.invoke(truncated_state, timeout=30)
    except:
        return {
            "error": "timeout",
            "action": "escalate",
            "escalation_reason": "System timeout"
        }
```

**4. Invalid Classification**

```python
classification = classifier_agent.invoke(state)

# Validate classification
valid_types = ["technical", "billing", "account", "booking", "general"]
if classification["issue_type"] not in valid_types:
    logger.warning(f"Invalid classification: {classification['issue_type']}")
    classification["issue_type"] = "general"  # Default
```

### Fallback Strategies

**Priority Order**:

1. **Retry Once**: For transient failures (network, timeout)
2. **Use Cache**: For tool failures, use cached data if available
3. **Degrade Gracefully**: Provide partial answer with disclaimer
4. **Escalate**: If all else fails, escalate with context

**Implementation**:
```python
def resolve_with_fallback(state: UDAHubState) -> dict:
    # Try RAG
    try:
        articles = retriever.invoke(query)
        if len(articles) > 0:
            return generate_answer(articles)
    except:
        logger.warning("RAG failed, trying cache")

    # Try cache
    cached_answer = check_cache(query)
    if cached_answer:
        return {
            "resolved": True,
            "confidence": 0.6,
            "answer": cached_answer + " (from cache)"
        }

    # Escalate
    return {
        "resolved": False,
        "confidence": 0.0,
        "escalation_reason": "Unable to retrieve knowledge"
    }
```

---

## Logging and Observability

### Required Logging Points

**1. Routing Decisions**

```python
logger.info({
    "event": "routing_decision",
    "timestamp": datetime.now().isoformat(),
    "ticket_id": ticket_id,
    "from_agent": "supervisor",
    "to_agent": "resolver",
    "reason": "classification complete, attempting resolution",
    "confidence": 1.0
})
```

**2. Agent Execution**

```python
logger.info({
    "event": "agent_execution",
    "timestamp": datetime.now().isoformat(),
    "agent_name": "resolver",
    "ticket_id": ticket_id,
    "duration_ms": 1234,
    "success": True,
    "output_summary": "Generated answer with confidence 0.85"
})
```

**3. Tool Invocations**

```python
logger.info({
    "event": "tool_invocation",
    "timestamp": datetime.now().isoformat(),
    "tool_name": "subscription_management_tool",
    "ticket_id": ticket_id,
    "user_id": user_id,
    "result_status": "success",
    "duration_ms": 234
})
```

**4. Confidence Scores**

```python
logger.info({
    "event": "confidence_calculation",
    "timestamp": datetime.now().isoformat(),
    "ticket_id": ticket_id,
    "confidence": 0.85,
    "factors": {
        "similarity": 0.92,
        "completeness": 0.85,
        "quality": 1.0,
        "context": 1.0
    },
    "threshold": 0.7,
    "decision": "resolve"
})
```

**5. Escalations**

```python
logger.info({
    "event": "escalation",
    "timestamp": datetime.now().isoformat(),
    "ticket_id": ticket_id,
    "trigger_reason": "low_confidence",
    "confidence": 0.45,
    "priority": "P2",
    "summary": "Refund request requires approval"
})
```

### Monitoring Metrics

**Key Metrics to Track**:

1. **Resolution Rate**: % of tickets resolved without escalation
   - Target: > 70%

2. **Average Confidence**: Mean confidence for resolved tickets
   - Target: > 0.8

3. **Escalation Rate**: % of tickets escalated
   - Target: < 30%

4. **Tool Usage**: Frequency of each tool
   - Monitor for patterns

5. **Response Time**: Average time from creation to resolution
   - Target: < 2 minutes

6. **Agent Performance**: Success rate per agent
   - Classifier accuracy
   - Resolver success rate

**Metrics Collection**:
```python
class MetricsCollector:
    def record_resolution(self, ticket_id: str, confidence: float, escalated: bool):
        # Store in metrics database
        pass

    def get_resolution_rate(self, time_period: str) -> float:
        # Calculate resolution rate
        pass

    def get_avg_confidence(self, time_period: str) -> float:
        # Calculate average confidence
        pass
```

---

## Multi-Turn Conversations

### Handling Follow-Up Questions

**Scenario**: User asks follow-up after initial resolution

**Process**:
```
1. New message arrives with same thread_id
   └─> Checkpointer loads previous state

2. Supervisor analyzes message in context
   └─> Checks if related to previous issue

3. Routing decision:
   - Related to previous → Route to Resolver with full context
   - New issue → Route to Classifier for new classification
   - Clarification → Route to Resolver directly

4. Agents have access to full conversation history
   └─> state["messages"] contains all messages

5. Resolution considers previous context
   └─> Builds on previous answer
```

**Example**:
```
# Initial conversation
User: "How do I reset my password?"
AI: "Tap 'Forgot Password' on the login screen..."
Status: Resolved (confidence: 0.85)

# Follow-up (same thread_id)
User: "I didn't receive the email"
→ Supervisor detects follow-up
→ Routes to Resolver with full context
AI: "Check your spam folder. The email comes from noreply@cultpass.com. If still not there, wait 5 minutes and try again."
Status: Resolved (confidence: 0.80)
```

### Context Window Management

**Strategy**:
- Keep last 10 messages in active context
- Summarize older messages if > 10 turns
- Store full history in database

**Implementation**:
```python
def prepare_context(messages: list[AnyMessage], max_messages: int = 10) -> list[AnyMessage]:
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

---

## Integration with External Systems

### Webhook Flow

```
┌─────────────────────┐
│ External System     │
│ (Zendesk, etc.)     │
└──────────┬──────────┘
           │ HTTP POST (webhook)
           ▼
┌─────────────────────┐
│ UDA-Hub API         │
│ /webhooks/ticket    │
└──────────┬──────────┘
           │ Create Ticket
           ▼
┌─────────────────────┐
│ Database            │
│ - Ticket            │
│ - TicketMetadata    │
│ - TicketMessage     │
└──────────┬──────────┘
           │ Initialize State
           ▼
┌─────────────────────┐
│ Workflow Graph      │
│ invoke(state, config)│
└──────────┬──────────┘
           │ Process
           ▼
┌─────────────────────┐
│ Return Response     │
│ (JSON)              │
└──────────┬──────────┘
           │ HTTP 200
           ▼
┌─────────────────────┐
│ External System     │
│ (receives response) │
└─────────────────────┘
```

### Response Format

**Successful Resolution**:
```json
{
  "ticket_id": "ticket_123",
  "status": "resolved",
  "response": "Try tapping 'Forgot Password' on the login screen...",
  "confidence": 0.85,
  "articles_used": ["article_4"],
  "processing_time_ms": 1234
}
```

**Escalation**:
```json
{
  "ticket_id": "ticket_456",
  "status": "escalated",
  "escalation_summary": {
    "issue": "Refund request for cancelled reservation",
    "priority": "P2",
    "reason": "Requires approval",
    "recommended_action": "Review and approve/deny refund"
  },
  "context": {
    "user_email": "user@example.com",
    "subscription_tier": "premium"
  },
  "processing_time_ms": 2345
}
```

---

## Related Documentation

- **System Overview**: See `ARCHITECTURE.md`
- **Agent Details**: See `AGENT_SPECIFICATIONS.md`
- **Memory**: See `MEMORY_STRATEGY.md`
- **RAG**: See `RAG_IMPLEMENTATION.md`
- **Diagrams**: See `DIAGRAMS.md`
