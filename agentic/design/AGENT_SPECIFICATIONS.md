# Agent Specifications

This document provides implementation-ready specifications for each agent in the UDA-Hub system.

---

## 1. Supervisor Agent

### Purpose
Central orchestrator for ticket routing and agent coordination

### Input State
```python
{
    "messages": list[AnyMessage],              # Chat message history
    "ticket_metadata": {
        "ticket_id": str,
        "account_id": str,
        "user_id": str,
        "channel": str,                        # email, chat, api
        "urgency": str | None                  # low, medium, high, critical
    },
    "classification": dict | None,             # From Classifier Agent
    "resolution": dict | None,                 # From Resolver Agent
    "tool_results": dict | None,               # From Tool Agent
    "escalation": dict | None                  # From Escalation Agent
}
```

### Output State
```python
{
    "messages": list[AnyMessage],              # Updated with supervisor decisions
    "next_agent": str,                         # Name of next agent to invoke
    "routing_reason": str                      # Explanation for routing decision
}
```

### Decision Logic

**Decision Tree**:
```
1. IF ticket.status == "new" AND classification is None:
     → Route to Classifier Agent

2. ELSE IF classification exists AND resolution is None:
     → Route to Resolver Agent

3. ELSE IF resolution.confidence < 0.7:
     → Route to Escalation Agent

4. ELSE IF resolution.resolved == True:
     → Route to END (mark ticket as resolved)

5. ELSE IF resolver indicates need for data:
     → Route to Tool Agent
     → After tool execution, route back to Resolver Agent
```

### Implementation Details

**LLM Configuration**:
- Use OpenAI GPT-4o-mini with structured output
- Temperature: 0 (deterministic decisions)
- Response schema: `SupervisorDecision`

**Structured Output Schema**:
```python
from pydantic import BaseModel, Field

class SupervisorDecision(BaseModel):
    next_agent: Literal["classifier", "resolver", "tool_agent", "escalation", "END"]
    reasoning: str = Field(description="Explanation for routing decision")
    confidence: float = Field(description="Confidence in routing decision (0-1)")
```

**System Prompt** (key points):
- Emphasize customer satisfaction and accuracy over speed
- Consider full conversation history when routing
- Maintain context between agent handoffs
- Log all routing decisions with clear reasoning

**Handoff Tools**:
- Bind handoff tools for each specialized agent
- Tools created using LangGraph's handoff mechanism
- Each tool represents a route to specific agent

**Node Function Signature**:
```python
from langgraph.types import Command
from typing import Literal

def supervisor(state: MessagesState) -> Command[Literal["classifier", "resolver", "tool_agent", "escalation", END]]:
    """
    Supervisor agent that routes tickets to appropriate specialized agents.

    Args:
        state: Current conversation state with messages and metadata

    Returns:
        Command object with goto target and optional state updates
    """
    # Implementation
    pass
```

---

## 2. Classifier Agent

### Purpose
Categorize tickets by type, urgency, and complexity

### Input State
```python
{
    "messages": list[AnyMessage],              # Ticket content and user messages
    "ticket_metadata": dict                    # Basic ticket information
}
```

### Output State
```python
{
    "messages": list[AnyMessage],              # Updated with classification
    "classification": {
        "issue_type": str,                     # technical, billing, account, booking, general
        "urgency": str,                        # low, medium, high, critical
        "complexity": str,                     # simple, moderate, complex
        "tags": list[str],                     # Relevant tags
        "confidence": float                    # 0-1
    }
}
```

### Classification Criteria

**Issue Types**:

1. **Technical**:
   - Login problems, authentication failures
   - App crashes, freezes, performance issues
   - Compatibility problems (device, OS)
   - QR code scanning issues
   - Feature not working as expected

2. **Billing**:
   - Payment method issues
   - Refund requests
   - Invoice generation
   - Subscription charges, unexpected charges
   - Failed payment transactions

3. **Account**:
   - Profile updates, personal information changes
   - Security settings, privacy concerns
   - Password reset, forgot credentials
   - Account deletion requests
   - Email address changes

4. **Booking/Reservations**:
   - How to reserve experiences
   - Cancellation requests
   - Rescheduling events
   - Waitlist inquiries
   - No-show policy questions
   - Attendance confirmation

5. **General**:
   - Questions about service features
   - How-to inquiries
   - Subscription tier information
   - General feedback
   - Feature requests

### Urgency Determination

**Critical** (Immediate attention required):
- Account security breach
- Service completely unavailable
- Payment processing failure
- Data loss or corruption

**High** (Same-day response needed):
- Cannot access key features
- Event happening soon (within 24 hours)
- Billing dispute with immediate impact
- Blocked account without clear reason

**Medium** (Response within 1-2 days):
- Minor bugs affecting user experience
- General questions with time sensitivity
- Feature working but suboptimally
- Non-urgent billing questions

**Low** (Response within 3-5 days):
- Feature requests
- General information inquiries
- Suggestions for improvement
- Documentation clarification

### Complexity Assessment

**Simple**:
- Single issue with clear resolution path
- Covered by existing knowledge articles
- No data lookup required

**Moderate**:
- Multiple related issues
- May require data lookup
- Some ambiguity in user description

**Complex**:
- Multiple unrelated issues
- Requires policy exception
- Needs human judgment
- Edge cases not covered by knowledge base

### Implementation Details

**Agent Creation**:
```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

classifier_agent = create_react_agent(
    model=model,
    tools=[],  # No tools required for classification
    state_schema=UDAHubState,
    prompt=SystemMessage(content=CLASSIFIER_PROMPT)
)
```

**System Prompt** (key points):
```
You are a ticket classification specialist for UDA-Hub.

Your task is to analyze support tickets and classify them accurately.

For each ticket:
1. Read the user's message carefully
2. Identify the primary issue type
3. Assess urgency based on impact and timeline
4. Determine complexity based on resolution difficulty
5. Extract relevant tags for search

Return structured classification data.

Be thorough but efficient - this should complete in one pass.
```

**Structured Output**:
- Use Pydantic model for type safety
- Validate field values (e.g., issue_type in allowed list)
- Confidence score based on clarity of user message

**Database Updates**:
- Store classification in `TicketMetadata` table
- Update `main_issue_type` field
- Store tags as comma-separated string
- Update `status` to "classified"

**Node Function Signature**:
```python
def classifier_agent(state: MessagesState) -> Command[Literal["supervisor"]]:
    """
    Classifier agent that categorizes tickets by type, urgency, and complexity.

    Args:
        state: Current conversation state

    Returns:
        Command to return to supervisor with classification results
    """
    # Implementation
    pass
```

---

## 3. Resolver Agent

### Purpose
Resolve tickets using knowledge base articles via RAG

### Input State
```python
{
    "messages": list[AnyMessage],              # Ticket content and conversation
    "classification": dict,                    # Ticket classification
    "ticket_metadata": dict,                   # Ticket information
    "tool_results": dict | None                # Data from Tool Agent (if needed)
}
```

### Output State
```python
{
    "messages": list[AnyMessage],              # Updated with resolution attempt
    "resolution": {
        "resolved": bool,                      # Whether ticket was resolved
        "confidence": float,                   # 0-1
        "answer": str | None,                  # Generated response
        "articles_used": list[str],            # Article IDs used
        "escalation_reason": str | None        # Why escalation is needed
    }
}
```

### Resolution Process

**Step-by-Step**:

1. **Invoke Retriever Tool**
   - Pass ticket content as query
   - Retriever searches knowledge base
   - Returns top-k relevant articles (k=3)

2. **Receive Retrieved Articles**
   - Articles include title, content, tags
   - Sorted by similarity score
   - Filtered by account_id (only CultPass articles)

3. **Generate Answer**
   - Read retrieved articles carefully
   - Use **suggested phrasing** from articles when available
   - Follow article guidelines (e.g., refund policies)
   - Combine information from multiple articles if needed

4. **Calculate Confidence**
   - **Retrieval Similarity** (40%): Average similarity score from vector search
   - **Answer Completeness** (30%): Does answer fully address the question?
   - **Article Quality** (20%): Do articles have suggested phrasing?
   - **Context Match** (10%): Does classification match article tags?

5. **Decision**
   - If confidence >= 0.7: Mark as resolved
   - If confidence < 0.7: Recommend escalation

### Tools Required

**Knowledge Retriever Tool**:
```python
from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever=retriever,
    name="knowledge_retriever",
    description=(
        "Search the CultPass knowledge base for articles about common issues. "
        "Use this tool when you need information about: "
        "login problems, reservations, subscriptions, billing, account management, "
        "or technical issues. Input should be a search query describing the user's issue."
    )
)
```

### Implementation Details

**Agent Creation**:
```python
resolver_agent = create_react_agent(
    model=model,
    tools=[retriever_tool],  # Bind retriever tool
    state_schema=UDAHubState,
    prompt=SystemMessage(content=RESOLVER_PROMPT)
)
```

**System Prompt** (key points):
```
You are a helpful customer support agent for CultPass.

Your goal is to resolve user issues using the knowledge base.

When a user asks a question:
1. Use the knowledge_retriever tool to search for relevant articles
2. Read the retrieved articles carefully
3. Generate a response using the **Suggested phrasing** from articles
4. If no relevant articles found, acknowledge inability to help
5. Always be polite and professional

IMPORTANT GUIDELINES:
- Follow article guidelines strictly (e.g., "Do NOT offer refunds unless approved")
- Use exact suggested phrasing when available
- If you're not confident (< 0.7), recommend escalation
- Cite article sources when possible

REFUND POLICY:
- Do NOT offer refunds unless article explicitly allows it
- Direct refund requests to human agents
```

**Confidence Scoring Implementation**:
```python
def calculate_confidence(
    similarity_scores: list[float],
    answer_length: int,
    has_suggested_phrasing: bool,
    classification_match: bool
) -> float:
    # Similarity score (40%)
    avg_similarity = sum(similarity_scores) / len(similarity_scores)
    similarity_factor = 0.4 * avg_similarity

    # Completeness score (30%)
    if answer_length > 50:
        completeness_factor = 0.3 * 1.0
    elif answer_length > 20:
        completeness_factor = 0.3 * 0.7
    else:
        completeness_factor = 0.3 * 0.3

    # Quality score (20%)
    quality_factor = 0.2 * (1.0 if has_suggested_phrasing else 0.7)

    # Context match (10%)
    context_factor = 0.1 * (1.0 if classification_match else 0.5)

    return similarity_factor + completeness_factor + quality_factor + context_factor
```

**Database Updates**:
- Store resolution in `TicketMessage` table with role="ai"
- Update `TicketMetadata.status` to "resolved" if successful
- Store articles_used for tracking

**Node Function Signature**:
```python
def resolver_agent(state: MessagesState) -> Command[Literal["supervisor"]]:
    """
    Resolver agent that attempts to resolve tickets using RAG.

    Args:
        state: Current conversation state with classification

    Returns:
        Command to return to supervisor with resolution results
    """
    # Implementation
    pass
```

---

## 4. Tool Agent

### Purpose
Execute database operations on CultPass external system

### Input State
```python
{
    "messages": list[AnyMessage],              # Request for data/action
    "ticket_metadata": dict                    # Ticket info with user context
}
```

### Output State
```python
{
    "messages": list[AnyMessage],              # Updated with tool results
    "tool_results": dict                       # Operation outcomes
}
```

### Available Tools

**1. user_lookup_tool**

**Purpose**: Retrieve user information from CultPass database

**Input**:
```python
{
    "email": str | None,       # User email address
    "user_id": str | None      # CultPass user ID
}
```

**Output**:
```python
{
    "user_id": str,
    "full_name": str,
    "email": str,
    "is_blocked": bool,
    "subscription": {          # If exists
        "status": str,
        "tier": str,
        "monthly_quota": int
    }
}
```

**Implementation**:
```python
from utils import get_session
from data.models.cultpass import User as CultPassUser, Subscription

def user_lookup(email: str = None, user_id: str = None) -> dict:
    with get_session("data/external/cultpass.db") as session:
        if email:
            user = session.query(CultPassUser).filter_by(email=email).first()
        elif user_id:
            user = session.query(CultPassUser).filter_by(user_id=user_id).first()
        else:
            return {"error": "Must provide email or user_id"}

        if not user:
            return {"error": "User not found"}

        # Get subscription
        subscription = session.query(Subscription).filter_by(user_id=user.user_id).first()

        return {
            "user_id": user.user_id,
            "full_name": user.full_name,
            "email": user.email,
            "is_blocked": user.is_blocked,
            "subscription": {
                "status": subscription.status,
                "tier": subscription.tier,
                "monthly_quota": subscription.monthly_quota
            } if subscription else None
        }
```

---

**2. subscription_management_tool**

**Purpose**: Check subscription details and status

**Input**:
```python
{
    "user_id": str             # CultPass user ID
}
```

**Output**:
```python
{
    "subscription_id": str,
    "status": str,             # active, expired, cancelled
    "tier": str,               # basic, premium, vip
    "monthly_quota": int,
    "start_date": str,
    "end_date": str
}
```

**Implementation**: Query `Subscription` table from `data/models/cultpass.py`

---

**3. experience_search_tool**

**Purpose**: Search available experiences

**Input**:
```python
{
    "search_query": str,            # Search text
    "is_premium": bool | None,      # Filter by premium flag
    "date_range": tuple | None      # (start_date, end_date)
}
```

**Output**:
```python
{
    "experiences": [
        {
            "experience_id": str,
            "title": str,
            "description": str,
            "available_slots": int,
            "is_premium": bool
        }
    ]
}
```

**Implementation**: Query `Experience` table with filters

---

**4. reservation_management_tool**

**Purpose**: View or cancel reservations

**Input**:
```python
{
    "user_id": str,
    "action": str,                  # "list" or "cancel"
    "reservation_id": str | None    # Required for cancel
}
```

**Output**:
```python
{
    "reservations": [
        {
            "reservation_id": str,
            "experience_id": str,
            "experience_title": str,
            "status": str,          # confirmed, cancelled, completed
            "reserved_at": str
        }
    ]
}
```

**Implementation**: Query `Reservation` table, join with `Experience` for details

---

**5. refund_processing_tool** (Restricted)

**Purpose**: Process refunds (requires approval)

**Input**:
```python
{
    "user_id": str,
    "reservation_id": str,
    "reason": str,
    "approved": bool                # Must be True to execute
}
```

**Output**:
```python
{
    "refund_status": str,           # "processed" or "requires_approval"
    "message": str
}
```

**Implementation**:
- Check `approved` flag
- If False, return "requires_approval"
- If True, update reservation status to "refunded"

---

### Implementation Details

**Agent Creation**:
```python
from langchain_core.tools import tool

# Define tools with @tool decorator
@tool
def user_lookup_tool(email: str = None, user_id: str = None) -> dict:
    """Look up user information from CultPass database."""
    return user_lookup(email, user_id)

# Create agent with all tools
tool_agent = create_react_agent(
    model=model,
    tools=[
        user_lookup_tool,
        subscription_management_tool,
        experience_search_tool,
        reservation_management_tool,
        refund_processing_tool
    ],
    state_schema=UDAHubState,
    prompt=SystemMessage(content=TOOL_AGENT_PROMPT)
)
```

**System Prompt** (key points):
```
You are a tool execution agent for CultPass support.

Your role is to retrieve data from the CultPass database.

When asked for information:
1. Identify which tool is needed
2. Extract required parameters from the request
3. Execute the appropriate tool
4. Return results in clear format

IMPORTANT:
- NEVER process refunds without approval flag
- Handle errors gracefully
- Protect user privacy
```

**Error Handling**:
```python
# User not found
{"error": "User not found", "suggestion": "Check email spelling"}

# Database error
{"error": "Database unavailable", "action": "escalate"}

# Permission error
{"error": "Refund requires approval", "action": "escalate"}
```

**Node Function Signature**:
```python
def tool_agent(state: MessagesState) -> Command[Literal["supervisor"]]:
    """
    Tool agent that executes database operations on CultPass system.

    Args:
        state: Current conversation state with data request

    Returns:
        Command to return to supervisor with tool results
    """
    # Implementation
    pass
```

---

## 5. Escalation Agent

### Purpose
Handle cases requiring human intervention

### Input State
```python
{
    "messages": list[AnyMessage],              # Full conversation history
    "classification": dict,                    # Ticket classification
    "resolution": dict,                        # Failed resolution details
    "ticket_metadata": dict                    # Ticket information
}
```

### Output State
```python
{
    "messages": list[AnyMessage],              # Updated with escalation summary
    "escalation": {
        "summary": str,                        # Concise issue summary
        "attempted_steps": list[str],          # Resolution attempts
        "priority": str,                       # P1, P2, P3, P4
        "recommended_action": str,             # Next steps for human
        "context": dict                        # Relevant user/ticket info
    }
}
```

### Escalation Triggers

1. **Low Confidence**: Resolver confidence < 0.7
2. **No Knowledge Found**: No relevant articles in knowledge base
3. **User Request**: User explicitly asks for human agent
4. **Policy Exception**: Requires judgment call (e.g., refund approval)
5. **Technical Issue**: Beyond knowledge base scope
6. **Multiple Failures**: Multiple resolution attempts failed

### Priority Levels

**P1 - Critical** (Immediate response):
- Service outage affecting multiple users
- Account security breach
- Payment processing complete failure
- Data loss or corruption

**P2 - High** (Within 2 hours):
- User cannot access service at all
- Event happening within 24 hours
- Billing dispute with financial impact
- Account blocked without clear reason

**P3 - Medium** (Within 1 business day):
- Feature not working, but workarounds exist
- General support needed
- Non-urgent billing questions
- Enhancement requests with business impact

**P4 - Low** (Within 3-5 business days):
- Feature requests
- General information inquiries
- Suggestions for improvement
- Documentation clarification

### Summary Format

**Structured Summary**:
```
ISSUE: [One-sentence description]

USER INFORMATION:
- Name: [Full name]
- Email: [Email address]
- Subscription: [Tier] ([Status])
- User ID: [CultPass user_id]

ATTEMPTED RESOLUTION:
1. [First attempt and outcome]
2. [Second attempt and outcome]
3. [Additional attempts...]

ESCALATION REASON:
[Clear explanation of why human intervention is needed]

RECOMMENDATION:
[Specific action for human agent to take]

CONTEXT:
- Classification: [Issue type, urgency]
- Confidence: [Resolver confidence score]
- Articles searched: [Article IDs]
- Tools used: [Tool names]
```

### Implementation Details

**Agent Creation**:
```python
escalation_agent = create_react_agent(
    model=model,
    tools=[],  # No tools required
    state_schema=UDAHubState,
    prompt=SystemMessage(content=ESCALATION_PROMPT)
)
```

**System Prompt** (key points):
```
You are an escalation specialist for UDA-Hub.

Your role is to prepare tickets for human agent handoff.

When escalating:
1. Summarize the issue concisely
2. Document all attempted resolution steps
3. Assign appropriate priority level
4. Provide actionable recommendation
5. Include all relevant context

Be thorough but concise - human agents need clear, actionable information.
```

**Database Updates**:
- Update `TicketMetadata.status` to "escalated"
- Store escalation summary in `TicketMessage` with role="system"
- Set priority flag for human agent queue

**Node Function Signature**:
```python
def escalation_agent(state: MessagesState) -> Command[Literal[END]]:
    """
    Escalation agent that prepares tickets for human intervention.

    Args:
        state: Current conversation state with full history

    Returns:
        Command to END with escalation summary
    """
    # Implementation
    pass
```

---

## Agent Communication Protocol

### Command Pattern

**All agents use the `Command` pattern for routing**:

**Agent Returns to Supervisor**:
```python
return Command(
    goto="supervisor",
    update={"messages": [...], "classification": {...}}
)
```

**Supervisor Routes to Agent**:
```python
return Command(
    goto="resolver",
    update={"messages": [...]}
)
```

**State Updates**:
- Updates are merged into `MessagesState`
- Messages accumulated using `operator.add`
- Other fields replaced with new values

**Routing Decisions**:
- All routing logged with reasoning
- Decisions include confidence scores
- Full audit trail in state history

---

## State Schema

**Shared State for All Agents**:

```python
from typing import Annotated
import operator
from langchain_core.messages import AnyMessage
from langgraph.graph import MessagesState

class UDAHubState(MessagesState):
    """State schema for UDA-Hub multi-agent system."""

    messages: Annotated[list[AnyMessage], operator.add]
    ticket_metadata: dict
    classification: dict | None = None
    resolution: dict | None = None
    tool_results: dict | None = None
    escalation: dict | None = None
```

---

## Related Documentation

- **System Overview**: See `ARCHITECTURE.md`
- **Data Flow**: See `DATA_FLOW.md` for routing logic
- **Memory**: See `MEMORY_STRATEGY.md` for state persistence
- **RAG**: See `RAG_IMPLEMENTATION.md` for knowledge retrieval
- **Diagrams**: See `DIAGRAMS.md` for visual representations
