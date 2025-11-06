# UDA-Hub Architecture

## System Overview

**Purpose**: Universal Decision Agent (UDA-Hub) for customer support automation

**First Customer**: CultPass - a cultural experiences subscription service

**Integration Points**:
- UDA-Hub core database (ticket management, knowledge base)
- CultPass external database (user data, subscriptions, reservations)

---

## Multi-Agent Architecture Pattern

### Pattern Type: Supervisor Pattern

**Centralized Decision-Making Architecture**

The supervisor pattern provides:
- **Clear task delegation**: Central orchestrator assigns work to specialized agents
- **Centralized routing logic**: Single point of decision-making for ticket flow
- **Better control over escalation**: Supervisor makes informed escalation decisions
- **Easier debugging**: All routing decisions flow through one agent

**Why Supervisor over Swarm?**
- Swarm patterns are better for collaborative tasks requiring peer-to-peer communication
- Support tickets require hierarchical decision-making and clear handoffs
- Supervisor provides better observability and control for customer-facing systems

### Implementation

**Technology**: `langgraph-supervisor` package with LangGraph StateGraph

**Core Mechanism**:
- Supervisor agent receives tickets and analyzes state
- Supervisor routes to specialized agents using `Command` objects
- Agents return results to supervisor
- Supervisor decides next step (continue processing or end)

---

## High-Level Architecture Components

### 1. Supervisor Agent

**Role**: Central orchestrator for ticket routing and agent coordination

**Responsibilities**:
- Analyze incoming ticket content and metadata
- Decide which specialized agent should handle each stage
- Coordinate handoffs between agents
- Make final escalation decisions based on confidence scores
- Maintain routing history for observability

**Implementation**:
- Uses LLM with structured output to return routing decisions
- Returns `Command` objects specifying next agent
- Considers ticket state, classification, and resolution attempts

**Decision Criteria**:
- New tickets → Route to Classifier
- Classified tickets → Route to Resolver
- Need data → Route to Tool Agent
- Low confidence → Route to Escalation
- High confidence → END (resolved)

---

### 2. Classifier Agent

**Role**: Categorizes tickets by type and urgency

**Responsibilities**:
- Analyze ticket content to determine issue type (technical, billing, account, booking, general)
- Extract urgency level from content and metadata
- Tag tickets with relevant categories
- Return structured classification results to supervisor

**Implementation**:
- Uses `create_react_agent` with classification-focused system prompt
- Outputs structured data using Pydantic models
- Single-shot classification (no tools required)

**Output Schema**:
```python
{
    "issue_type": str,      # technical, billing, account, booking, general
    "urgency": str,         # low, medium, high, critical
    "complexity": str,      # simple, moderate, complex
    "tags": list[str],      # relevant tags
    "confidence": float     # 0-1
}
```

---

### 3. Resolver Agent

**Role**: Attempts to resolve tickets using knowledge base via RAG

**Responsibilities**:
- Use RAG tool to retrieve relevant knowledge articles
- Generate responses based on retrieved context
- Follow suggested phrasing from articles
- Calculate confidence scores for answers
- Recommend escalation if confidence is low

**Implementation**:
- Uses `create_react_agent` with resolver-focused system prompt
- Has access to `knowledge_retriever` tool (RAG-based)
- Generates answers following article guidelines

**Tools**:
- `knowledge_retriever`: RAG tool created with `create_retriever_tool`
  - Searches Knowledge table for account-specific articles
  - Returns top-k relevant articles (k=3 recommended)
  - Includes article title, content, and tags

**Confidence Thresholds**:
- ≥ 0.8: High confidence → Resolve immediately
- 0.7-0.79: Medium confidence → Resolve with disclaimer
- 0.5-0.69: Low confidence → Escalate with summary
- < 0.5: Very low confidence → Escalate immediately

---

### 4. Tool Agent

**Role**: Executes database operations on CultPass external system

**Responsibilities**:
- Look up user information (email, blocked status)
- Check subscription details (status, tier, quota)
- Search and manage experience reservations
- Process refunds and cancellations (with approval)
- Return structured operation results

**Implementation**:
- Uses `create_react_agent` with tool-focused system prompt
- Has access to all CultPass database tools
- Tools use SQLAlchemy models from `data/models/cultpass.py`

**Available Tools**:
1. `user_lookup_tool`: Query user by email or ID
2. `subscription_management_tool`: Check subscription status and tier
3. `experience_search_tool`: Search available experiences
4. `reservation_management_tool`: View/cancel reservations
5. `refund_processing_tool`: Process refunds (restricted)

**Database Access**: Uses session management from `utils.py` (`get_session` context manager)

---

### 5. Escalation Agent

**Role**: Handles cases requiring human intervention

**Responsibilities**:
- Summarize ticket context and full conversation history
- Document all attempted resolution steps
- Assign priority level for human agents
- Store escalation metadata in database
- Provide actionable recommendations

**Implementation**:
- Uses `create_react_agent` with summarization-focused system prompt
- No tools required (pure LLM reasoning)
- Generates structured escalation summaries

**Escalation Triggers**:
- Low confidence from Resolver (< 0.7)
- No relevant knowledge articles found
- User explicitly requests human agent
- Complex issue requiring judgment call
- Refund or policy exception needed
- Technical issue beyond knowledge base scope

**Priority Levels**:
- **P1 (Critical)**: Service outage, account security, payment failure
- **P2 (High)**: Cannot use service, upcoming event issue, billing dispute
- **P3 (Medium)**: Feature not working, general support needed
- **P4 (Low)**: Feature request, general inquiry

---

## Technology Stack

### Orchestration & Agents
- **LangGraph**: StateGraph for workflow orchestration
- **langgraph-supervisor**: Supervisor pattern implementation
- **LangChain**: Agent creation and tool binding

### LLM & Embeddings
- **OpenAI GPT-4o-mini**: Primary language model (as shown in `agentic/workflow.py`)
- **OpenAI Embeddings**: text-embedding-3-small for RAG

### Data & Storage
- **SQLAlchemy**: ORM for database operations
- **SQLite**: Database engine
- **InMemoryVectorStore**: Vector storage for RAG
- **MemorySaver**: Checkpointing for session state

### Models
- **UDA-Hub Core**: `data/models/udahub.py` (Account, User, Ticket, TicketMetadata, TicketMessage, Knowledge)
- **CultPass External**: `data/models/cultpass.py` (User, Subscription, Experience, Reservation)

---

## Integration Points

### UDA-Hub Core Database

**Models from `data/models/udahub.py`**:

1. **Account**: Customer accounts (e.g., "cultpass")
   - `account_id`, `account_name`, `created_at`

2. **User**: UDA-Hub users linked to accounts
   - `user_id`, `account_id`, `external_user_id`, `user_name`

3. **Ticket**: Support tickets with channel information
   - `ticket_id`, `account_id`, `user_id`, `channel`, `created_at`

4. **TicketMetadata**: Status, issue type, tags
   - `ticket_id`, `status`, `main_issue_type`, `tags`, `urgency`

5. **TicketMessage**: Conversation history with roles
   - `message_id`, `ticket_id`, `role` (user, agent, ai, system), `content`, `created_at`

6. **Knowledge**: Knowledge base articles per account
   - `article_id`, `account_id`, `title`, `content`, `tags`

### CultPass External Database

**Models from `data/models/cultpass.py`**:

1. **User**: CultPass users
   - `user_id`, `full_name`, `email`, `is_blocked`

2. **Subscription**: User subscriptions
   - `subscription_id`, `user_id`, `status`, `tier`, `monthly_quota`, `start_date`, `end_date`

3. **Experience**: Bookable experiences
   - `experience_id`, `title`, `description`, `available_slots`, `is_premium`

4. **Reservation**: User reservations
   - `reservation_id`, `user_id`, `experience_id`, `status`, `reserved_at`

---

## Design Principles

### 1. Modularity
- Each agent is a separate module in `agentic/agents/`
- Agents are independent and reusable
- Clear interfaces between components

### 2. Tool Abstraction
- Database operations abstracted as tools in `agentic/tools/`
- Tools encapsulate database queries
- Consistent error handling across tools

### 3. State Management
- Centralized state using `MessagesState` pattern
- State accumulated using `operator.add` annotation
- Checkpointer persists state after each node

### 4. Confidence-Based Routing
- Decisions based on quantifiable confidence scores
- Transparent thresholds for escalation
- Confidence factors documented and tunable

### 5. Memory Persistence
- Session-level memory for conversations (MemorySaver)
- Cross-session memory for customer history (database + semantic search)
- Thread ID strategy: thread_id = ticket_id

### 6. Observability
- Comprehensive logging at all decision points
- Routing decisions tracked with reasoning
- Performance metrics for monitoring

---

## Data Flow Overview

```
External System
  ↓
Ticket Creation (Database)
  ↓
Initialize State (thread_id)
  ↓
Supervisor (Analyze)
  ↓
Classifier → Supervisor
  ↓
Resolver (+ RAG) → Supervisor
  ↓
Tool Agent (if needed) → Supervisor
  ↓
Escalation (if low confidence) → END
  OR
Resolved (if high confidence) → END
```

---

## Related Documentation

- **Agent Details**: See `AGENT_SPECIFICATIONS.md` for detailed agent designs
- **Routing Logic**: See `DATA_FLOW.md` for routing and decision flow
- **Memory Implementation**: See `MEMORY_STRATEGY.md` for state and memory management
- **Knowledge Retrieval**: See `RAG_IMPLEMENTATION.md` for RAG system details
- **Visual Diagrams**: See `DIAGRAMS.md` for Mermaid diagrams
