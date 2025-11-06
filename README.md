# UDA-Hub: Multi-Agent Customer Support System

An intelligent customer support system built with LangGraph's supervisor pattern, featuring specialized agents for ticket classification, RAG-based resolution, database operations, and intelligent escalation.

## Overview

UDA-Hub is a production-ready multi-agent system designed to handle customer support tickets for CultPass (a fitness subscription service). The system autonomously:

- **Classifies** incoming support tickets by issue type, urgency, and sentiment
- **Resolves** issues using a knowledge base (RAG) when possible
- **Executes** database operations for user-specific queries
- **Escalates** complex issues to human agents when necessary
- **Remembers** context across conversations for personalized support
- **Logs** all decisions for monitoring and analysis

## Features

✅ **Intelligent Ticket Classification** - Categorizes issues with high accuracy  
✅ **RAG-Based Knowledge Retrieval** - Answers questions from knowledge base  
✅ **Database Tool Integration** - Queries and modifies CultPass data  
✅ **Memory Management** - Session and cross-session context preservation  
✅ **Automatic Escalation** - Routes complex issues to humans  
✅ **Comprehensive Logging** - Structured JSON logging for all events  
✅ **Multi-Turn Conversations** - Maintains context across exchanges  
✅ **High Test Coverage** - Extensive unit, integration, and E2E tests

## Architecture

```
┌──────────────────────────────────────────────────┐
│              Supervisor Agent                     │
│         (Central Orchestrator)                    │
└─────────┬────────────┬────────────┬──────────────┘
          │            │            │            
          ▼            ▼            ▼            
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

**Supervisor Pattern**: Central orchestrator routes tickets to specialized agents based on classification, confidence, and state.

## Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key
- pip package manager

### Installation

```bash
# 1. Clone repository
git clone <repository-url>
cd UDA-Hub

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 5. Initialize databases
jupyter notebook 01_external_db_setup.ipynb  # Run all cells
jupyter notebook 02_core_db_setup.ipynb      # Run all cells

# 6. Run application
jupyter notebook 03_agentic_app.ipynb        # Run example scenarios
```

### Verify Installation

```bash
python solution/docs/test_installation.py
```

Expected output: `✅ All tests passed! Setup complete.`

## Dependencies

### Core Framework
- **langchain** (>=0.3.27) - Agent framework and orchestration
- **langchain-openai** (>=0.3.28) - OpenAI LLM and embeddings integration
- **langgraph** (>=0.5.4) - Workflow graph and state management
- **langgraph-supervisor** (>=0.0.28) - Supervisor pattern implementation

### Database & Storage
- **sqlalchemy** (>=2.0.41) - ORM for database operations
- SQLite (built-in) - Database engine

### Utilities
- **python-dotenv** (>=1.1.1) - Environment variable management
- **ipykernel** (>=6.30.0) - Jupyter notebook support

### AI Models
- **OpenAI GPT-4o-mini** - Language model for agents
- **OpenAI text-embedding-3-small** - Embeddings for RAG

See `requirements.txt` for complete list with version constraints.

## Project Structure

```
UDA-Hub/
├── agentic/                    # Core agent system
│   ├── agents/                 # Agent implementations
│   │   ├── classifier.py       # Classification agent
│   │   ├── resolver.py         # RAG-based resolution
│   │   ├── tool_agent.py       # Database operations
│   │   ├── escalation.py       # Human handoff
│   │   └── state.py            # State schema
│   ├── tools/                  # Tools and utilities
│   │   ├── rag_setup.py        # RAG system initialization
│   │   ├── cultpass_read_tools.py   # Database read ops
│   │   ├── cultpass_write_tools.py  # Database write ops
│   │   └── confidence_scorer.py     # Confidence calculation
│   ├── design/                 # Design documentation
│   │   ├── ARCHITECTURE.md     # System architecture
│   │   ├── AGENT_SPECIFICATIONS.md
│   │   ├── DATA_FLOW.md
│   │   └── ...
│   ├── workflow.py             # Supervisor orchestration
│   ├── memory.py               # Memory management
│   ├── logging.py              # Structured logging
│   └── inspector.py            # Log analysis
│
├── solution/                   # Deliverables
│   ├── tests/                  # Comprehensive test suite
│   │   ├── test_agents.py      # Agent unit tests
│   │   ├── test_workflow.py    # Integration tests
│   │   ├── test_e2e_scenarios.py  # End-to-end tests
│   │   ├── test_rag.py         # RAG system tests
│   │   ├── test_logging.py     # Logging tests
│   │   ├── conftest.py         # Pytest fixtures
│   │   └── fixtures/           # Test data
│   │       └── sample_tickets.py
│   └── docs/                   # Documentation
│       ├── AGENTS.md           # Agent specifications
│       ├── TOOLS.md            # Tool documentation
│       ├── WORKFLOW.md         # Workflow guide
│       └── SETUP.md            # Installation guide
│
├── data/                       # Databases and knowledge
│   ├── external/               # CultPass database
│   │   ├── cultpass.db
│   │   └── cultpass_articles.jsonl
│   ├── core/                   # UDA-Hub database
│   │   └── udahub.db
│   └── models/                 # Database models
│       ├── cultpass.py
│       └── udahub.py
│
├── logs/                       # Application logs
│   ├── udahub.log
│   └── udahub.json
│
├── 01_external_db_setup.ipynb  # Database initialization
├── 02_core_db_setup.ipynb
├── 03_agentic_app.ipynb        # Main application
├── requirements.txt            # Python dependencies
├── .env                        # Environment configuration (not in repo)
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

## Usage

### Example 1: Simple Login Issue (RAG Resolution)

```python
from langchain_core.messages import HumanMessage
from agentic.workflow import orchestrator

state = {
    "messages": [
        HumanMessage(content="I forgot my password. How do I reset it?")
    ],
    "ticket_metadata": {
        "ticket_id": "TKT-001",
        "account_id": "cultpass",
        "user_email": "user@example.com"
    }
}

config = {"configurable": {"thread_id": "TKT-001"}}
result = orchestrator.invoke(state, config)

print(result["messages"][-1].content)
# Output: "To reset your password: 1) Go to login page..."
```

**Flow**: Classifier → Resolver (RAG) → Response

### Example 2: Booking Query (Tool Execution)

```python
state = {
    "messages": [
        HumanMessage(content="Show me my upcoming class reservations")
    ],
    "ticket_metadata": {
        "ticket_id": "TKT-002",
        "account_id": "cultpass",
        "user_email": "yogi@example.com"
    }
}

result = orchestrator.invoke(state, config)

# Flow: Classifier → Tool Agent (database query) → Response
# Output: "You have 2 upcoming reservations: Yoga on Nov 10..."
```

### Example 3: Complex Issue (Escalation)

```python
state = {
    "messages": [
        HumanMessage(content="I was charged twice! I need a refund now!")
    ],
    "ticket_metadata": {
        "ticket_id": "TKT-003",
        "account_id": "cultpass",
        "user_email": "angry@example.com"
    }
}

result = orchestrator.invoke(state, config)

# Flow: Classifier → Escalation Agent
# Output: "I'm connecting you with our billing team..."
print(result["escalation"]["priority"])  # "high"
```

## Testing

### Run All Tests

```bash
pytest solution/tests/ -v
```

### Run Specific Test Suites

```bash
# Unit tests for agents
pytest solution/tests/test_agents.py -v

# Integration tests
pytest solution/tests/test_workflow.py -v

# End-to-end scenarios
pytest solution/tests/test_e2e_scenarios.py -v

# RAG system tests
pytest solution/tests/test_rag.py -v

# Logging tests
pytest solution/tests/test_logging.py -v
```

### Test Coverage

```bash
pytest solution/tests/ --cov=agentic --cov-report=html
# Open htmlcov/index.html in browser
```

### Test Breakdown

**Unit Tests** (`test_agents.py`):
- ✅ Classifier accuracy for different issue types
- ✅ Resolver RAG retrieval and confidence scoring
- ✅ Tool Agent database operations
- ✅ Escalation logic and triggers
- ✅ Agent integration and data flow

**Integration Tests** (`test_workflow.py`):
- ✅ Supervisor routing decisions
- ✅ Multi-agent coordination
- ✅ State management and transitions
- ✅ Memory persistence (session and cross-session)
- ✅ Error handling and fallbacks

**E2E Tests** (`test_e2e_scenarios.py`):
- ✅ Complete user journeys (login, booking, billing)
- ✅ Multi-turn conversations
- ✅ Complex multi-issue tickets
- ✅ User-requested escalations
- ✅ Memory and personalization

**RAG Tests** (`test_rag.py`):
- ✅ Document retrieval accuracy
- ✅ Confidence scoring
- ✅ Knowledge base quality
- ✅ Performance benchmarks

**Logging Tests** (`test_logging.py`):
- ✅ Event capture completeness
- ✅ Log format and structure
- ✅ Metrics calculation
- ✅ Error handling

## Documentation

Comprehensive documentation in `solution/docs/`:

- **[SETUP.md](solution/docs/SETUP.md)** - Complete installation and configuration guide
- **[AGENTS.md](solution/docs/AGENTS.md)** - Detailed agent specifications and usage
- **[TOOLS.md](solution/docs/TOOLS.md)** - RAG and database tool documentation
- **[WORKFLOW.md](solution/docs/WORKFLOW.md)** - Workflow architecture and routing logic
- **[VOCAREUM.md](solution/docs/VOCAREUM.md)** - Vocareum GenAI Gateway integration guide

Additional design docs in `agentic/design/`:
- ARCHITECTURE.md, AGENT_SPECIFICATIONS.md, DATA_FLOW.md, RAG_IMPLEMENTATION.md, MEMORY_STRATEGY.md, DIAGRAMS.md

## Rubric Coverage

This implementation covers all project rubric requirements:

1. ✅ **Classification** - Multi-category ticket classification with confidence scoring
2. ✅ **Routing** - Supervisor-based intelligent routing to specialized agents
3. ✅ **RAG** - Knowledge base retrieval with embeddings and similarity search
4. ✅ **Tools** - Database read/write tools for CultPass operations
5. ✅ **Memory** - Session memory (MemorySaver) and cross-session memory (CustomerMemoryStore)
6. ✅ **Escalation** - Automatic and user-requested escalation with context preservation
7. ✅ **Logging** - Structured JSON logging for all events and decisions
8. ✅ **Testing** - Comprehensive unit, integration, and E2E test coverage
9. ✅ **Documentation** - Complete setup, usage, and API documentation

## Performance Metrics

**Target Performance**:
- Classification accuracy: >90%
- Auto-resolution rate: >70%
- RAG retrieval time: <500ms
- Database query time: <200ms
- End-to-end response: <5 seconds
- Escalation rate: <30%

Monitor these metrics using the logging inspector:

```python
from agentic.inspector import analyze_logs

metrics = analyze_logs(log_dir="logs")
print(metrics)
```

## Built With

* [LangChain](https://python.langchain.com/) - Agent framework
* [LangGraph](https://langchain-ai.github.io/langgraph/) - Workflow orchestration
* [OpenAI](https://openai.com/) - Language models and embeddings
* [SQLAlchemy](https://www.sqlalchemy.org/) - Database ORM
* [Pytest](https://pytest.org/) - Testing framework

## Environment Variables

**Required Configuration**:

Create a `.env` file in the project root (never commit this file):

```bash
# OpenAI API Configuration
OPENAI_API_KEY=sk-your-actual-key-here

# Optional: Vocareum GenAI Gateway Configuration
# If using Vocareum, uncomment and set the gateway endpoint:
# OPENAI_API_BASE=https://gateway.vocareum.com/v1

# Optional: Model Configuration  
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Optional: Database Paths
CULTPASS_DB_PATH=data/external/cultpass.db
UDAHUB_DB_PATH=data/core/udahub.db

# Optional: Logging Configuration
LOG_LEVEL=INFO
LOG_DIR=logs
```

**Security Best Practices**:

1. **Never commit `.env` files** - Use `.env.example` as a template
2. **Rotate API keys regularly** - Especially if accidentally exposed
3. **Use environment-specific keys** - Different keys for dev/test/prod
4. **Limit API key permissions** - Only grant necessary scopes
5. **Monitor API usage** - Set up alerts for unusual activity

**Loading Environment Variables**:

The application automatically loads `.env` using `python-dotenv`:

```python
from dotenv import load_dotenv
import os

load_dotenv()  # Load from .env file
api_key = os.getenv("OPENAI_API_KEY")
```

**Running Tests**:

Tests mock all API calls, so no real API key is needed:

```bash
# Tests work even without OPENAI_API_KEY set
pytest solution/tests/
```

The `conftest.py` automatically patches `ChatOpenAI` and `OpenAIEmbeddings` to prevent network requests during testing.

**Accidental Key Exposure**:

If you accidentally commit an API key:

1. **Immediately rotate the key** in your OpenAI dashboard
2. **Remove from git history**: `git filter-repo --path .env --invert-paths`
3. **Verify removal**: Check GitHub/GitLab that file doesn't exist in history

## Security & Privacy

- ✅ No `.env` files in repository (use `.env.example` template)
- ✅ No large `.db` files committed (databases generated locally)
- ✅ API keys loaded from environment variables only
- ✅ User data access controls in tools
- ✅ Audit logging for all operations
- ✅ **Sensitive data redaction in logs** (emails, API keys, tokens automatically sanitized)
- ✅ **PII protection** (phone numbers, credit cards masked in logs)
- ✅ **Email hashing** (emails hashed but domain preserved for debugging)

## Known Limitations

- SQLite used for simplicity (use PostgreSQL/MySQL for production)
- In-memory vector store (consider Pinecone/Weaviate for scale)
- Single-threaded processing (add async for concurrency)
- Limited knowledge base (expand based on real tickets)

## Future Enhancements

- Real-time chat interface (WebSocket)
- Advanced analytics dashboard
- Multi-language support
- Voice/call integration
- Automated testing in CI/CD
- Performance monitoring and alerts
- A/B testing for different prompts

## Contributing

See development setup in `solution/docs/SETUP.md`.

Code formatting:
```bash
black agentic/ solution/
flake8 agentic/ solution/
```

## License

See LICENSE file for details.
