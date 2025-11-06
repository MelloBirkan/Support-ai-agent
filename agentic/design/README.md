# UDA-Hub Design Documentation

This folder contains comprehensive design documentation for the UDA-Hub multi-agent customer support system.

---

## Documentation Structure

### 1. [ARCHITECTURE.md](./ARCHITECTURE.md)
**System Overview and Multi-Agent Architecture**

- High-level system design
- Supervisor pattern explanation
- Agent roles and responsibilities overview
- Technology stack
- Database integration points (UDA-Hub core and CultPass external)
- Design principles

**Read this first** to understand the overall system architecture.

---

### 2. [AGENT_SPECIFICATIONS.md](./AGENT_SPECIFICATIONS.md)
**Detailed Agent Specifications**

- **Supervisor Agent**: Central orchestrator
- **Classifier Agent**: Ticket categorization
- **Resolver Agent**: Knowledge-based resolution
- **Tool Agent**: Database operations
- **Escalation Agent**: Human handoff

Each specification includes:
- Purpose and responsibilities
- Input/output state schemas
- Decision logic and criteria
- Implementation details
- Node function signatures

**Read this** for implementation-ready agent designs.

---

### 3. [DATA_FLOW.md](./DATA_FLOW.md)
**Data Flow and Decision Routing Logic**

- Ticket ingestion flow
- Agent routing flow with diagrams
- Supervisor decision tree
- State transitions
- Message flow and types
- Tool invocation flow
- Confidence scoring and thresholds
- Error handling and fallbacks
- Logging and observability
- Multi-turn conversation handling

**Read this** to understand how data flows through the system.

---

### 4. [MEMORY_STRATEGY.md](./MEMORY_STRATEGY.md)
**Memory and State Management**

- Short-term memory (session-level) using MemorySaver
- Long-term memory (cross-session) using database and semantic search
- Thread ID strategy
- State schema and persistence
- Memory lifecycle
- Integration points in agents
- Configuration and best practices

**Read this** to understand memory implementation.

---

### 5. [RAG_IMPLEMENTATION.md](./RAG_IMPLEMENTATION.md)
**RAG System for Knowledge Retrieval**

- RAG architecture and flow
- Knowledge base structure
- Document preprocessing
- Embedding generation with OpenAI
- Vector store setup (InMemoryVectorStore)
- Retriever tool creation
- Integration with Resolver Agent
- Confidence scoring logic
- Document grading (optional)
- Testing and optimization

**Read this** to understand knowledge retrieval implementation.

---

### 6. [DIAGRAMS.md](./DIAGRAMS.md)
**System Diagrams and Visualizations**

- High-level architecture diagram
- Supervisor routing flow
- Agent interaction sequence
- RAG system architecture
- Memory architecture
- State transition diagram
- Tool agent operations
- Confidence scoring flow
- Database schema
- Error handling flow

**Read this** for visual representations of the system.

---

## Quick Start Guide

### For System Architects
1. Read `ARCHITECTURE.md` for overall design
2. Review `DATA_FLOW.md` for routing logic
3. Check `MEMORY_STRATEGY.md` for state management
4. View `DIAGRAMS.md` for visual overview

### For Agent Developers
1. Read `AGENT_SPECIFICATIONS.md` for your assigned agent
2. Review `DATA_FLOW.md` for integration points
3. Check `MEMORY_STRATEGY.md` for state access patterns
4. Reference `DIAGRAMS.md` for agent interactions

### For Tool Developers
1. Read `AGENT_SPECIFICATIONS.md` (Tool Agent section)
2. Review `RAG_IMPLEMENTATION.md` for retriever tool
3. Check database models in `data/models/cultpass.py` and `data/models/udahub.py`
4. View `DIAGRAMS.md` for tool operation flow

### For RAG Developers
1. Read `RAG_IMPLEMENTATION.md` in detail
2. Review knowledge base structure in `data/external/cultpass_articles.jsonl`
3. Check `AGENT_SPECIFICATIONS.md` (Resolver Agent section)
4. View `DIAGRAMS.md` for RAG architecture

---

## Design Principles

### 1. Modularity
Each agent is independent and reusable, with clear separation of concerns.

### 2. Clarity
Clear interfaces between components make the system easy to understand and maintain.

### 3. Scalability
Architecture supports adding new agents, tools, and knowledge sources without major refactoring.

### 4. Observability
Comprehensive logging at all decision points enables monitoring and debugging.

### 5. Reliability
Error handling and fallback strategies ensure graceful degradation.

### 6. Privacy
Secure handling of customer data with anonymization and retention policies.

---

## Architecture Diagram

See `DIAGRAMS.md` for detailed diagrams. High-level overview:

```
┌─────────────────────────────────────────────────────────┐
│                    Supervisor Agent                      │
│              (Central Orchestrator)                      │
└────────┬────────────┬────────────┬────────────┬─────────┘
         │            │            │            │
         ▼            ▼            ▼            ▼
   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
   │Classifier│ │ Resolver │ │   Tool   │ │Escalation│
   │  Agent   │ │  Agent   │ │  Agent   │ │  Agent   │
   └──────────┘ └────┬─────┘ └────┬─────┘ └──────────┘
                     │            │
                     ▼            ▼
              ┌──────────┐  ┌──────────┐
              │   RAG    │  │ CultPass │
              │  System  │  │   DB     │
              └──────────┘  └──────────┘
```

---

## Technology Stack

### Orchestration & Agents
- **LangGraph**: StateGraph for workflow orchestration
- **langgraph-supervisor**: Supervisor pattern implementation
- **LangChain**: Agent creation and tool binding

### LLM & Embeddings
- **OpenAI GPT-4o-mini**: Primary language model
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
- **Account**: Customer accounts (e.g., CultPass)
- **User**: UDA-Hub users linked to accounts
- **Ticket**: Support tickets with channel information
- **TicketMetadata**: Status, issue type, tags
- **TicketMessage**: Conversation history with roles
- **Knowledge**: Knowledge base articles per account

### CultPass External Database
- **User**: CultPass users with email and blocked status
- **Subscription**: User subscriptions with status, tier, quota
- **Experience**: Bookable experiences with slots and premium flags
- **Reservation**: User reservations with status tracking

---

## Next Steps

After reviewing the design documentation:

1. **Expand Knowledge Base**: Add 10+ articles to `data/external/cultpass_articles.jsonl`
2. **Implement Tools**: Create database tools in `agentic/tools/`
3. **Implement RAG**: Build RAG system following `RAG_IMPLEMENTATION.md`
4. **Implement Agents**: Create agent modules in `agentic/agents/`
5. **Build Workflow**: Implement supervisor workflow in `agentic/workflow.py`
6. **Add Memory**: Integrate memory management
7. **Add Logging**: Implement structured logging
8. **Create Tests**: Develop comprehensive test cases

Refer to the project rubric for detailed requirements and submission guidelines.

---

## Document Roadmap

| Document | Purpose | Key Topics |
|----------|---------|------------|
| ARCHITECTURE.md | System overview | Multi-agent pattern, components, tech stack |
| AGENT_SPECIFICATIONS.md | Agent implementation | Supervisor, Classifier, Resolver, Tool, Escalation |
| DATA_FLOW.md | Routing and flow | Decision trees, state transitions, message flow |
| MEMORY_STRATEGY.md | State management | Short-term, long-term, lifecycle |
| RAG_IMPLEMENTATION.md | Knowledge retrieval | Embeddings, vector store, confidence scoring |
| DIAGRAMS.md | Visual reference | Architecture, flows, sequences |

---

## Contributing to Documentation

When updating these documents:
1. Maintain consistency in formatting and structure
2. Update cross-references when adding new sections
3. Include code examples for clarity
4. Add diagrams to `DIAGRAMS.md` for visual concepts
5. Keep examples aligned with actual implementation

---

## Version History

**Version 1.0** (2025-11-05)
- Initial comprehensive design documentation
- All six core documents created
- Multi-agent supervisor architecture defined
- RAG implementation detailed
- Memory strategy documented
