# Setup Guide

Complete installation and configuration guide for UDA-Hub Multi-Agent Customer Support System.

---

## Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **pip**: Latest version
- **SQLite**: 3.x (included with Python)
- **Operating System**: macOS, Linux, or Windows

### API Keys Required
- **OpenAI API Key**: For GPT-4o-mini and embeddings
  - Get key from: https://platform.openai.com/api-keys
  - Ensure billing is set up and you have credits

### Knowledge Required
- Basic Python programming
- Understanding of virtual environments
- Familiarity with Jupyter notebooks (optional)
- Basic command line usage

---

## Installation Steps

### 1. Clone or Download Project

```bash
cd /path/to/your/projects
# If using git (recommended)
git clone <repository-url> UDA-Hub
cd UDA-Hub

# Or download and extract ZIP file
```

### 2. Create Virtual Environment

**macOS/Linux**:
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows**:
```bash
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` in your terminal prompt.

### 3. Upgrade pip

```bash
pip install --upgrade pip
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies installed**:
- `langchain>=0.3.27` - Agent framework
- `langchain-openai>=0.3.28` - OpenAI integration
- `langgraph>=0.5.4` - Workflow orchestration
- `langgraph-supervisor>=0.0.28` - Supervisor pattern
- `sqlalchemy>=2.0.41` - Database ORM
- `python-dotenv>=1.1.1` - Environment variables
- `ipykernel>=6.30.0` - Jupyter notebook support

**Installation time**: ~2-5 minutes

### 5. Configure Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.example .env  # If example exists
# Or create new file
touch .env
```

Edit `.env` and add:

```env
# OpenAI Configuration
OPENAI_API_KEY=sk-your-actual-api-key-here

# Database Paths (SQLite - relative to project root)
EXTERNAL_DB_PATH=data/external/cultpass.db
CORE_DB_PATH=data/core/udahub.db

# Logging Configuration
LOG_LEVEL=INFO
LOG_DIR=logs

# Model Configuration
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
TEMPERATURE=0.7

# RAG Configuration
RAG_TOP_K=3
RAG_MIN_SIMILARITY=0.6

# Optional: LangSmith Tracing (for debugging)
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_API_KEY=your-langsmith-key
```

**Security Note**: Never commit `.env` file to version control!

### 6. Initialize Databases

**Step A: External Database (CultPass)**

Open and run notebook:
```bash
jupyter notebook 01_external_db_setup.ipynb
```

Or run cells programmatically:
```bash
python -c "
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
with open('01_external_db_setup.ipynb') as f:
    nb = nbformat.read(f, as_version=4)
ep = ExecutePreprocessor(timeout=600)
ep.preprocess(nb, {'metadata': {'path': '.'}})
"
```

This creates:
- `data/external/cultpass.db`
- Tables: User, Subscription, Experience, Reservation
- Sample data for testing

**Step B: Core Database (UDA-Hub)**

```bash
jupyter notebook 02_core_db_setup.ipynb
```

This creates:
- `data/core/udahub.db`
- Tables: Account, User, Ticket, TicketMetadata, TicketMessage, Knowledge
- Sample accounts and knowledge base

**Verification**:
```bash
ls -lh data/external/cultpass.db
ls -lh data/core/udahub.db
# Both files should exist and be >0 bytes
```

---

## Configuration

### Knowledge Base Setup

The knowledge base is stored in `data/external/cultpass_articles.jsonl`.

**Minimum Requirements**:
- At least 10 articles covering main categories
- Categories: login, booking, billing, subscription, account

**Add/Edit Articles**:
1. Open `data/external/cultpass_articles.jsonl`
2. Add JSON objects, one per line:

```json
{"title": "Article Title", "content": "Full content...", "category": "login", "tags": ["tag1", "tag2"], "last_updated": "2025-11-05"}
```

3. Re-initialize RAG system (automatically happens on startup)

### Logging Configuration

Logs are written to `logs/` directory (created automatically).

**Log Files**:
- `logs/udahub.log` - Application logs
- `logs/udahub.json` - Structured JSON logs for analysis

**Log Levels**:
- `DEBUG`: Detailed information for debugging
- `INFO`: General information (default)
- `WARNING`: Warning messages
- `ERROR`: Error messages

Change log level in `.env`:
```env
LOG_LEVEL=DEBUG  # For development
LOG_LEVEL=INFO   # For production
```

### Memory Configuration

**Session Memory** (MemorySaver):
- Automatically configured
- Stores conversation state
- Thread ID = Ticket ID

**Cross-Session Memory** (CustomerMemoryStore):
- Configured in `agentic/memory.py`
- Stores user history in UDA-Hub database
- Retrieves previous tickets by user_id

No additional configuration needed.

---

## Running the Application

### Option 1: Jupyter Notebook (Recommended for Testing)

```bash
jupyter notebook 03_agentic_app.ipynb
```

Run all cells to:
1. Initialize orchestrator
2. Run example scenarios
3. Analyze logs
4. View metrics

### Option 2: Python Script

Create `run.py`:
```python
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from agentic.workflow import orchestrator

load_dotenv()

# Create ticket
state = {
    "messages": [
        HumanMessage(content="I can't login to my account")
    ],
    "ticket_metadata": {
        "ticket_id": "TKT-001",
        "account_id": "cultpass",
        "user_email": "user@example.com",
        "channel": "email"
    }
}

# Process ticket
config = {"configurable": {"thread_id": "TKT-001"}}
result = orchestrator.invoke(state, config)

# Print response
print("Response:", result["messages"][-1].content)
```

Run:
```bash
python run.py
```

### Option 3: Interactive CLI

Create `chat.py`:
```python
from dotenv import load_dotenv
from utils import chat_interface

load_dotenv()
chat_interface()  # Start interactive chat
```

Run:
```bash
python chat.py
```

---

## Verification

### Test Installation

Run this test script:

```python
# test_installation.py
import sys

def test_imports():
    """Test all required imports."""
    try:
        import langchain
        import langchain_openai
        import langgraph
        import sqlalchemy
        from dotenv import load_dotenv
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_env_vars():
    """Test environment variables."""
    from dotenv import load_dotenv
    import os
    
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set")
        return False
    print("✅ Environment variables configured")
    return True

def test_databases():
    """Test database files exist."""
    import os
    
    external_db = "data/external/cultpass.db"
    core_db = "data/core/udahub.db"
    
    if not os.path.exists(external_db):
        print(f"❌ {external_db} not found")
        return False
    if not os.path.exists(core_db):
        print(f"❌ {core_db} not found")
        return False
    
    print("✅ Databases initialized")
    return True

def test_rag_system():
    """Test RAG system initialization."""
    try:
        from agentic.tools import initialize_rag_system
        retriever = initialize_rag_system()
        print("✅ RAG system initialized")
        return True
    except Exception as e:
        print(f"❌ RAG initialization failed: {e}")
        return False

def test_agents():
    """Test agent creation."""
    try:
        from agentic.agents import (
            create_classifier_agent,
            create_resolver_agent,
            create_tool_agent,
            create_escalation_agent
        )
        
        classifier = create_classifier_agent()
        resolver = create_resolver_agent()
        tool_agent = create_tool_agent()
        escalation = create_escalation_agent()
        
        print("✅ All agents created successfully")
        return True
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        return False

def test_workflow():
    """Test workflow compilation."""
    try:
        from agentic.workflow import orchestrator
        print("✅ Workflow orchestrator compiled")
        return True
    except Exception as e:
        print(f"❌ Workflow compilation failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing UDA-Hub Installation...\n")
    
    tests = [
        ("Imports", test_imports),
        ("Environment", test_env_vars),
        ("Databases", test_databases),
        ("RAG System", test_rag_system),
        ("Agents", test_agents),
        ("Workflow", test_workflow)
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\nTesting {name}...")
        results.append(test_func())
    
    print("\n" + "="*50)
    if all(results):
        print("✅ All tests passed! Setup complete.")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Check errors above.")
        sys.exit(1)
```

Run:
```bash
python test_installation.py
```

### Expected Output
```
Testing UDA-Hub Installation...

Testing Imports...
✅ All imports successful

Testing Environment...
✅ Environment variables configured

Testing Databases...
✅ Databases initialized

Testing RAG System...
✅ RAG system initialized

Testing Agents...
✅ All agents created successfully

Testing Workflow...
✅ Workflow orchestrator compiled

==================================================
✅ All tests passed! Setup complete.
```

---

## Troubleshooting

### Issue: ModuleNotFoundError

**Problem**: `ModuleNotFoundError: No module named 'langchain'`

**Solution**:
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: OpenAI API Key Error

**Problem**: `openai.error.AuthenticationError: Invalid API key`

**Solution**:
1. Verify key in `.env` file
2. Check key is valid at https://platform.openai.com/api-keys
3. Ensure billing is set up
4. Reload environment:
   ```python
   from dotenv import load_dotenv
   load_dotenv(override=True)
   ```

### Issue: Database Not Found

**Problem**: `sqlite3.OperationalError: unable to open database file`

**Solution**:
```bash
# Create directories
mkdir -p data/external data/core

# Re-run database setup notebooks
jupyter notebook 01_external_db_setup.ipynb
jupyter notebook 02_core_db_setup.ipynb
```

### Issue: Jupyter Kernel Not Found

**Problem**: Jupyter can't find Python kernel

**Solution**:
```bash
# Install kernel in virtual environment
python -m ipykernel install --user --name=venv --display-name="Python (UDA-Hub)"

# Select kernel in Jupyter: Kernel > Change kernel > Python (UDA-Hub)
```

### Issue: Rate Limit Error

**Problem**: `openai.error.RateLimitError: Rate limit exceeded`

**Solution**:
1. Check OpenAI account usage and limits
2. Add retry logic or reduce request frequency
3. Consider upgrading OpenAI plan

### Issue: Memory/Performance Issues

**Problem**: Slow response or high memory usage

**Solution**:
1. Reduce RAG top_k in `.env`:
   ```env
   RAG_TOP_K=2
   ```
2. Use lighter model:
   ```env
   LLM_MODEL=gpt-3.5-turbo
   ```
3. Limit conversation history length

---

## Development Setup

### Install Development Dependencies

```bash
pip install pytest pytest-cov black flake8 mypy
```

### Run Tests

```bash
# All tests
pytest solution/tests/

# Specific test file
pytest solution/tests/test_agents.py

# With coverage
pytest solution/tests/ --cov=agentic --cov-report=html
```

### Code Formatting

```bash
# Format code
black agentic/ solution/

# Check style
flake8 agentic/ solution/
```

---

## Next Steps

After successful setup:

1. **Run Example Scenarios**: Execute `03_agentic_app.ipynb`
2. **Review Logs**: Check `logs/` for structured logging
3. **Explore Documentation**: Read `solution/docs/` guides
4. **Run Tests**: Execute `pytest solution/tests/`
5. **Customize Knowledge Base**: Add articles to `cultpass_articles.jsonl`
6. **Monitor Performance**: Use logging inspector

---

## Support

For issues or questions:
1. Check this guide's Troubleshooting section
2. Review error logs in `logs/`
3. Consult project documentation in `solution/docs/`
4. Check LangChain/LangGraph documentation

---

## Production Deployment

For production deployment:

1. **Use Production Database**: Replace SQLite with PostgreSQL/MySQL
2. **Set Up Environment**: Use production OpenAI keys with rate limits
3. **Configure Logging**: Use production log aggregation
4. **Add Monitoring**: Set up alerts for errors and performance
5. **Scale Infrastructure**: Use container orchestration (Docker, Kubernetes)
6. **Secure Secrets**: Use secret management service (AWS Secrets Manager, etc.)
7. **Add Rate Limiting**: Protect against abuse
8. **Set Up Backups**: Regular database backups

See production deployment guide for details (if available).
