# Agent Documentation

## Overview

The UDA-Hub system uses a supervisor pattern with specialized agents coordinated by a central supervisor. Each agent has specific responsibilities and decision-making capabilities.

---

## Supervisor Agent

**Purpose**: Central orchestrator that routes tickets to appropriate specialized agents.

**Responsibilities**:
- Receive incoming tickets and manage workflow
- Route to Classifier for initial categorization
- Make routing decisions based on classification and state
- Determine when resolution is complete or escalation needed
- Manage conversation flow and state transitions

**Routing Logic**:
1. **No classification** → Route to Classifier
2. **Classification exists, no resolution attempt** → Route to Resolver
3. **Resolver attempted, needs data** → Route to Tool Agent
4. **Low confidence or user requested human** → Route to Escalation
5. **High confidence resolution** → END (complete)

**Decision Criteria**:
- Classification confidence level (high/medium/low)
- Issue type and urgency
- Previous agent outcomes
- User sentiment and explicit requests

---

## Classifier Agent

**Purpose**: Analyze and categorize incoming support tickets.

**Input**: 
- User message(s)
- Ticket metadata

**Output**:
- Issue type (login, booking, billing, subscription, account, etc.)
- Confidence level (high/medium/low)
- Urgency (high/medium/low)
- Category (technical, billing, general)
- Sentiment (positive/neutral/negative)

**Classification Categories**:

### Login Issues
- Password reset requests
- Account access problems
- Two-factor authentication issues
- Account locked/suspended

### Booking Issues
- View/check reservations
- Cancel/modify bookings
- Class availability queries
- Booking errors

### Billing Issues
- Subscription status inquiries
- Payment problems
- Refund requests
- Duplicate charges
- Billing disputes

### Account Management
- Profile updates
- Email/phone changes
- Account settings
- Notification preferences

### General Inquiries
- How-to questions
- Feature information
- General support

**Confidence Thresholds**:
- **High (≥0.8)**: Clear, specific issue identified
- **Medium (0.5-0.79)**: Likely category but some ambiguity
- **Low (<0.5)**: Unclear or multiple potential issues

**Implementation**:
```python
from agentic.agents import create_classifier_agent

classifier = create_classifier_agent()
result = classifier.invoke(state)
# result["classification"] contains issue_type, confidence, urgency, etc.
```

---

## Resolver Agent

**Purpose**: Resolve issues using RAG-based knowledge retrieval.

**Input**:
- User message(s)
- Classification result
- Conversation history

**Output**:
- Resolution answer
- Confidence level
- RAG documents used
- Resolved status (True/False)

**Process**:
1. Analyze issue from classification
2. Query RAG system for relevant knowledge base articles
3. Evaluate retrieved documents for relevance
4. Generate response based on knowledge
5. Calculate confidence score
6. Determine if issue is resolved

**RAG Integration**:
- Uses OpenAI embeddings (text-embedding-3-small)
- Searches knowledge base with semantic similarity
- Retrieves top-k relevant documents
- Confidence based on similarity scores

**Confidence Scoring**:
- **High**: RAG found highly relevant articles (similarity > 0.85)
- **Medium**: Partial match found (similarity 0.6-0.85)
- **Low**: No good match or uncertain answer (similarity < 0.6)

**When Resolver Succeeds**:
- Clear knowledge base article matches issue
- High confidence answer generated
- User likely to be satisfied

**When Resolver Fails**:
- No relevant knowledge found
- Complex issue requiring data lookup
- Issue requires human judgment
- → Routes to Tool Agent or Escalation

**Implementation**:
```python
from agentic.agents import create_resolver_agent

resolver = create_resolver_agent()
result = resolver.invoke(state)
# result["resolution"] contains answer, confidence, resolved status
```

---

## Tool Agent

**Purpose**: Execute database operations to retrieve or modify data.

**Input**:
- User query requiring data
- Classification
- Account/user identifiers

**Output**:
- Tool execution results
- Data retrieved from database
- Operation status (success/failure)

**Available Tools**:

### Read Tools (CultPass Database)
1. **get_user_by_email(email)**: Retrieve user information
2. **get_user_subscriptions(user_id)**: Get subscription details
3. **get_user_reservations(user_id, status)**: List reservations
4. **get_experience_details(experience_id)**: Experience information
5. **check_subscription_status(user_id)**: Subscription validation

### Write Tools (CultPass Database)
1. **cancel_reservation(reservation_id, user_id)**: Cancel booking
2. **update_user_preferences(user_id, preferences)**: Update settings

**Tool Selection Logic**:
- **Subscription queries** → get_user_subscriptions
- **Booking queries** → get_user_reservations
- **Cancellation requests** → cancel_reservation
- **User info** → get_user_by_email

**Error Handling**:
- Database connection failures → Escalate
- Invalid parameters → Ask for clarification
- Permission issues → Escalate to human
- Data not found → Inform user, offer alternatives

**Example Scenarios**:
- "What's my subscription status?" → get_user_subscriptions
- "Show my reservations" → get_user_reservations
- "Cancel my booking tomorrow" → cancel_reservation

**Implementation**:
```python
from agentic.agents import create_tool_agent

tool_agent = create_tool_agent()
result = tool_agent.invoke(state)
# result["tool_results"] contains data from database
```

---

## Escalation Agent

**Purpose**: Handle cases requiring human intervention.

**Input**:
- Ticket state
- Reason for escalation
- Full conversation context

**Output**:
- Escalation confirmation
- Priority level
- Assignment queue
- Context summary for human agent

**Escalation Triggers**:

### Automatic Escalation
1. **Low confidence resolution** (confidence < 0.5)
2. **Complex billing disputes** (refunds, disputes)
3. **Security issues** (account locked, suspicious activity)
4. **Technical failures** (system errors, data issues)
5. **Multiple failed resolution attempts**

### User-Requested Escalation
1. Explicit request for human agent
2. Dissatisfaction with automated response
3. Urgent/sensitive matters

**Escalation Priorities**:
- **High**: Billing disputes, security issues, angry users
- **Medium**: Complex technical issues, account problems
- **Low**: General inquiries, feature requests

**Escalation Teams**:
- **Billing Team**: Payment, refund, subscription issues
- **Technical Team**: System bugs, technical failures
- **Security Team**: Account security, suspicious activity
- **General Support**: Other escalations

**Handoff Information**:
- Full conversation history
- Classification and attempted resolutions
- User sentiment and urgency
- Relevant database information
- Recommended next actions

**Implementation**:
```python
from agentic.agents import create_escalation_agent

escalation_agent = create_escalation_agent()
result = escalation_agent.invoke(state)
# result["escalation"] contains escalated status, reason, priority
```

---

## Agent Communication

**State Schema**:
All agents communicate through a shared state:

```python
{
    "messages": [HumanMessage(...), AIMessage(...)],  # Conversation
    "classification": {...},  # From Classifier
    "resolution": {...},  # From Resolver
    "tool_results": [...],  # From Tool Agent
    "escalation": {...},  # From Escalation
    "ticket_metadata": {...}  # Persistent metadata
}
```

**Message Flow**:
1. User message added to state
2. Classifier adds classification
3. Resolver adds resolution attempt
4. Tool Agent adds data results
5. Escalation adds handoff info
6. AI responses added to messages

---

## Best Practices

### For Classifier
- Always provide confidence and urgency
- Identify sentiment for escalation decisions
- Handle multi-issue tickets by prioritizing

### For Resolver
- Cite knowledge base sources
- Be honest about confidence levels
- Suggest escalation when uncertain

### For Tool Agent
- Validate user identity before operations
- Handle errors gracefully
- Log all database operations

### For Escalation
- Preserve full context
- Set appropriate priority
- Provide clear handoff summary

---

## Testing Agents

See `solution/tests/test_agents.py` for comprehensive unit tests covering:
- Classification accuracy
- RAG retrieval quality
- Tool execution
- Escalation logic
- Agent integration

---

## Performance Metrics

**Target Metrics**:
- Classification accuracy: >90%
- Resolution rate: >70% without escalation
- RAG confidence: >0.8 for resolvable issues
- Tool execution success: >95%
- Escalation rate: <30%

Monitor these metrics using the logging system documented in `LOGGING.md`.
