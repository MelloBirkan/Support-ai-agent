# RAG Implementation

This document describes the complete RAG (Retrieval Augmented Generation) system implementation for knowledge base retrieval in UDA-Hub.

---

## Overview

The RAG system enables the Resolver Agent to:
1. Retrieve relevant knowledge base articles based on user queries
2. Generate accurate responses using retrieved context
3. Follow suggested phrasing from articles
4. Calculate confidence scores for answers

This document covers the complete RAG architecture from document preprocessing to answer generation.

---

## RAG Architecture

### High-Level Flow

```
Knowledge Articles (JSONL)
  │
  ▼
Load from Database (Knowledge table)
  │
  ▼
Convert to Documents
  │
  ▼
Text Chunking (optional, for long articles)
  │
  ▼
Embedding Generation (OpenAI)
  │
  ▼
Vector Store (InMemoryVectorStore)
  │
  ▼
Create Retriever
  │
  ▼
Create Retriever Tool (LangChain)
  │
  ▼
Bind to Resolver Agent
  │
  ▼
User Query → Semantic Search
  │
  ▼
Relevant Articles Retrieved
  │
  ▼
Answer Generation with Context
  │
  ▼
Confidence Scoring
```

---

## Component Details

### 1. Knowledge Base Structure

**Source**: `data/external/cultpass_articles.jsonl`

**Article Format**:

```json
{
  "title": "How to Reserve a Spot for an Event",
  "content": "If a user asks how to reserve an event:\n\n- Guide them to the CultPass app\n- Instruct them to browse the experience catalog and tap 'Reserve'\n...\n\n**Suggested phrasing:**\n\"You can reserve an experience by opening the CultPass app...\"",
  "tags": "reservation, events, booking, attendance"
}
```

**Current Articles** (4 articles):
1. How to Reserve a Spot for an Event
2. What's Included in My Subscription
3. How to Handle Login Issues
4. How to Pause Your Subscription

**Target Articles** (14+ categories):

1. **Technical Issues**:
   - Login problems ✓ (existing)
   - App crashes
   - Performance issues
   - Compatibility problems
   - QR code not working

2. **Billing**:
   - Payment methods
   - Refund requests
   - Invoice generation
   - Subscription charges
   - Failed payments

3. **Account Management**:
   - Profile updates
   - Security settings
   - Password reset
   - Account deletion
   - Email change

4. **Booking/Reservations**:
   - How to reserve ✓ (existing)
   - Cancellation policy
   - Rescheduling events
   - Waitlist management
   - No-show policy

5. **Subscription**:
   - What's included ✓ (existing)
   - Tier differences
   - Upgrade/downgrade
   - Pause subscription ✓ (existing)
   - Quota management

**Storage**: Articles loaded into `Knowledge` table in `data/core/udahub.db`

**Knowledge Table Schema** (from `data/models/udahub.py`):
```python
class Knowledge(Base):
    __tablename__ = "knowledge"

    article_id = Column(String, primary_key=True)
    account_id = Column(String, ForeignKey("accounts.account_id"))
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    tags = Column(String, nullable=True)  # Comma-separated
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, onupdate=datetime.now)
```

---

### 2. Document Preprocessing

**Loading Articles from Database**:

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from data.models.udahub import Knowledge
from langchain_core.documents import Document

def load_knowledge_articles(account_id: str = "cultpass") -> list[Document]:
    """Load knowledge articles from database and convert to documents."""

    # Create database session
    engine = create_engine("sqlite:///data/core/udahub.db")
    Session = sessionmaker(bind=engine)
    session = Session()

    # Query articles for account
    articles = session.query(Knowledge).filter(
        Knowledge.account_id == account_id
    ).all()

    # Convert to LangChain documents
    documents = []
    for article in articles:
        doc = Document(
            page_content=f"Title: {article.title}\n\nContent: {article.content}",
            metadata={
                "article_id": article.article_id,
                "title": article.title,
                "tags": article.tags,
                "account_id": article.account_id
            }
        )
        documents.append(doc)

    session.close()
    return documents
```

**Text Chunking** (optional, for long articles):

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(documents: list[Document]) -> list[Document]:
    """Chunk long documents for better retrieval."""

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunked_docs = []
    for doc in documents:
        # Only chunk if article is very long (>1000 chars)
        if len(doc.page_content) > 1000:
            chunks = text_splitter.split_documents([doc])
            chunked_docs.extend(chunks)
        else:
            # Keep article whole
            chunked_docs.append(doc)

    return chunked_docs
```

**Recommendation**: For CultPass articles (which include suggested phrasing), **avoid chunking** to preserve context. Keep articles whole.

---

### 3. Embedding Generation

**Technology**: OpenAI Embeddings (text-embedding-3-small)

**Why OpenAI Embeddings**:
- High quality semantic understanding
- Fast inference
- Good performance on support content
- Already in requirements.txt (langchain-openai)

**Embedding Dimensions**: 1536 (for text-embedding-3-small)

**Implementation**:

```python
from langchain_openai import OpenAIEmbeddings

# Initialize embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    # API key from environment variable OPENAI_API_KEY
)
```

**Embedding Generation**:

```python
# Embed a single query
query = "How do I reset my password?"
query_embedding = embeddings.embed_query(query)
# Returns: list[float] of length 1536

# Embed multiple documents (batch)
documents_text = [doc.page_content for doc in documents]
doc_embeddings = embeddings.embed_documents(documents_text)
# Returns: list[list[float]]
```

---

### 4. Vector Store

**Technology**: InMemoryVectorStore (from langchain_core.vectorstores)

**Why InMemoryVectorStore**:
- Simple setup, no external dependencies
- Fast for small to medium knowledge bases (<1000 articles)
- Sufficient for CultPass (14+ articles)
- Can be replaced with persistent store later (Chroma, FAISS, Pinecone)

**Implementation**:

```python
from langchain_core.vectorstores import InMemoryVectorStore

# Create vector store from documents
vectorstore = InMemoryVectorStore.from_documents(
    documents=documents,
    embedding=embeddings
)
```

**Scaling Considerations**:
- **< 100 articles**: InMemoryVectorStore is perfect
- **100-1000 articles**: InMemoryVectorStore still good, consider caching
- **> 1000 articles**: Use persistent vector store (Chroma, FAISS, Pinecone)
- **Production**: Use persistent store for reliability

**Alternative Vector Stores**:

```python
# Chroma (persistent)
from langchain_chroma import Chroma

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# FAISS (persistent)
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(
    documents=documents,
    embedding=embeddings
)
vectorstore.save_local("faiss_index")
```

---

### 5. Retriever Creation

**Basic Retriever**:

```python
# Create retriever from vector store
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}  # Return top 3 most relevant articles
)
```

**Advanced Retriever** (with score threshold):

```python
# Only return articles with similarity > 0.7
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 3,
        "score_threshold": 0.7
    }
)
```

**Retriever Configuration Options**:

| Parameter | Description | Recommended Value |
|-----------|-------------|-------------------|
| `k` | Number of documents to return | 3 |
| `score_threshold` | Minimum similarity score | 0.7 |
| `search_type` | Search algorithm | "similarity_score_threshold" |

**Recommendation**: Use `similarity_score_threshold` to filter out irrelevant articles.

---

### 6. Retriever Tool Creation

**Purpose**: Wrap retriever as a LangChain tool for agent use

**Implementation**:

```python
from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever=retriever,
    name="knowledge_retriever",
    description=(
        "Search the CultPass knowledge base for articles about common issues. "
        "Use this tool when you need to find information about: "
        "login problems, reservations, subscriptions, billing, account management, "
        "or technical issues. Input should be a search query describing the user's issue."
    )
)
```

**Tool Behavior**:
- **Input**: Natural language query (e.g., "how to reset password")
- **Output**: List of relevant articles with content and metadata
- **Returns**: Top k articles (default k=3)

**Tool Description Best Practices**:
- List specific use cases
- Indicate what information it can provide
- Specify expected input format

---

### 7. Integration with Resolver Agent

**Binding Tool to Agent**:

```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

# Initialize model
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Create resolver agent with retriever tool
resolver_agent = create_react_agent(
    model=model,
    tools=[retriever_tool],  # Bind retriever tool
    state_schema=UDAHubState,
    prompt=SystemMessage(content=RESOLVER_PROMPT)
)
```

**Resolver Prompt** (example):

```python
RESOLVER_PROMPT = """
You are a helpful customer support agent for CultPass.

Your goal is to resolve user issues using the knowledge base.

When a user asks a question:
1. Use the knowledge_retriever tool to search for relevant articles
2. Read the retrieved articles carefully
3. Generate a response using the **Suggested phrasing** from the articles
4. If no relevant articles are found, acknowledge that you cannot help and recommend escalation
5. Always be polite and professional

IMPORTANT GUIDELINES:
- Follow article guidelines strictly (e.g., "Do NOT offer refunds unless approved")
- Use the exact suggested phrasing when available
- If you're not confident in your answer (confidence < 0.7), recommend escalation
- Cite article sources when possible

REFUND POLICY:
- Do NOT offer refunds unless article explicitly allows it
- Direct refund requests to human agents

RESPONSE FORMAT:
- Be concise and actionable
- Provide step-by-step instructions when available
- Include relevant links or references
"""
```

---

## RAG Workflow

### Step-by-Step Process

**Step 1: User Query Arrives**

```
User: "I can't log into my account"
```

**Step 2: Resolver Agent Invoked**

```python
# Supervisor routes to Resolver Agent
state = {
    "messages": [HumanMessage(content="I can't log into my account")],
    "classification": {"issue_type": "technical", "urgency": "high"},
    "ticket_metadata": {...}
}

result = resolver_agent.invoke(state)
```

**Step 3: Tool Invocation Decision**

```
Resolver Agent (LLM) analyzes query and decides to use knowledge_retriever tool

Generated tool call:
{
    "tool": "knowledge_retriever",
    "args": {
        "query": "login issues password reset authentication"
    }
}
```

**Step 4: Semantic Search**

```python
# Retriever tool executes
query = "login issues password reset authentication"

# 1. Embed query
query_embedding = embeddings.embed_query(query)

# 2. Search vector store
results = vectorstore.similarity_search(
    query=query,
    k=3
)

# 3. Return top 3 articles with similarity scores
```

**Step 5: Retrieved Articles**

```
Article 1 (similarity: 0.92):
  Title: "How to Handle Login Issues?"
  Content: "Most login issues are resolved with password reset.
            Try tapping 'Forgot Password' on the login screen.
            Make sure you're using the email associated with your account.

            **Suggested phrasing:**
            'Try tapping \"Forgot Password\" on the login screen...'"
  Tags: "login, password, authentication"

Article 2 (similarity: 0.78):
  Title: "Account Security Settings"
  Content: "...security features..."
  Tags: "account, security"

Article 3 (similarity: 0.65):
  Title: "Two-Factor Authentication"
  Content: "...2FA setup..."
  Tags: "security, authentication"
```

**Step 6: Answer Generation**

```python
# Resolver Agent (LLM) receives retrieved articles
# Reads articles
# Extracts suggested phrasing from Article 1
# Generates response

response = """
Try tapping 'Forgot Password' on the login screen. Make sure you're using
the email associated with your account. If the email doesn't arrive,
check your spam folder or try again in a few minutes.

If you continue to have issues, please let me know!
"""
```

**Step 7: Confidence Scoring**

```python
confidence = calculate_confidence(
    similarity_scores=[0.92, 0.78, 0.65],
    answer_length=len(response),
    has_suggested_phrasing=True,
    classification_match=True  # "technical" matches article tags
)

# Result: confidence = 0.88 (high)
```

**Step 8: Return to Supervisor**

```python
return {
    "resolved": True,
    "confidence": 0.88,
    "answer": response,
    "articles_used": ["article_4"],
    "escalation_reason": None
}
```

---

## Confidence Scoring Logic

### Factors

**1. Retrieval Similarity Score** (40% weight)

```python
avg_similarity = sum(similarity_scores) / len(similarity_scores)
similarity_factor = 0.4 * avg_similarity
```

**Interpretation**:
- 0.9-1.0: Exact or near-exact match
- 0.7-0.89: Strong semantic match
- 0.5-0.69: Moderate match
- < 0.5: Weak match

**2. Answer Completeness** (30% weight)

```python
if answer_length > 50 and has_actionable_steps:
    completeness_factor = 0.3 * 1.0
elif answer_length > 20:
    completeness_factor = 0.3 * 0.7
else:
    completeness_factor = 0.3 * 0.3
```

**Criteria**:
- Full answer with steps: 1.0
- Partial answer: 0.7
- Minimal answer: 0.3

**3. Article Quality** (20% weight)

```python
if "Suggested phrasing" in article_content:
    quality_factor = 0.2 * 1.0
elif article_tags:
    quality_factor = 0.2 * 0.7
else:
    quality_factor = 0.2 * 0.5
```

**Criteria**:
- Has suggested phrasing: 1.0
- Has tags: 0.7
- Basic article: 0.5

**4. Context Match** (10% weight)

```python
if classification_issue_type in article_tags:
    context_factor = 0.1 * 1.0
else:
    context_factor = 0.1 * 0.5
```

**Criteria**:
- Classification matches article tags: 1.0
- No match: 0.5

### Final Confidence Calculation

```python
def calculate_confidence(
    similarity_scores: list[float],
    answer_length: int,
    has_suggested_phrasing: bool,
    classification_match: bool
) -> float:
    """Calculate confidence score for resolution."""

    # Similarity score (40%)
    avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
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

    # Final confidence
    confidence = (
        similarity_factor +
        completeness_factor +
        quality_factor +
        context_factor
    )

    return round(confidence, 2)
```

### Escalation Thresholds

| Confidence | Action | Description |
|-----------|--------|-------------|
| ≥ 0.8 | Resolve immediately | High confidence, direct answer |
| 0.7-0.79 | Resolve with disclaimer | Medium confidence, answer + "If this doesn't help..." |
| 0.5-0.69 | Escalate with summary | Low confidence, provide context |
| < 0.5 | Escalate immediately | Very low confidence, no good answer |

---

## Document Grading (Optional Enhancement)

### Purpose
Validate that retrieved documents are actually relevant before generating answer.

### Implementation

**Grading Model**:

```python
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

class GradeDocuments(BaseModel):
    """Grade documents for relevance."""
    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, 'no' if not"
    )

GRADE_PROMPT = """
You are a grader assessing relevance of a retrieved document to a user question.

Document: {context}
Question: {question}

If the document contains information related to the question, grade it as relevant.
Give a binary score 'yes' or 'no'.
"""

def grade_documents(question: str, context: str) -> str:
    """Grade document relevance."""
    grader_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = GRADE_PROMPT.format(question=question, context=context)

    response = grader_model.with_structured_output(GradeDocuments).invoke(
        [{"role": "user", "content": prompt}]
    )

    return response.binary_score
```

### Usage in Workflow

```python
# In resolver agent
retrieved_docs = retriever_tool.invoke(query)

# Grade each document
relevant_docs = []
for doc in retrieved_docs:
    grade = grade_documents(query, doc.page_content)
    if grade == "yes":
        relevant_docs.append(doc)

# Use only relevant documents
if relevant_docs:
    # Generate answer with relevant docs
    pass
else:
    # No relevant docs, escalate
    pass
```

**Recommendation**: Implement document grading if accuracy is critical. For initial version, confidence scoring may be sufficient.

---

## RAG Initialization

### Setup Function

**Location**: `agentic/tools/rag_setup.py` (to be created)

**Implementation**:

```python
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.tools.retriever import create_retriever_tool
from data.models.udahub import Knowledge
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from langchain_core.documents import Document

def initialize_rag_system(account_id: str = "cultpass"):
    """
    Initialize RAG system for knowledge retrieval.

    Steps:
    1. Load articles from database
    2. Create embeddings
    3. Build vector store
    4. Create retriever
    5. Create retriever tool

    Args:
        account_id: Account to load articles for (default: "cultpass")

    Returns:
        retriever_tool: LangChain tool for knowledge retrieval
    """

    # 1. Load articles from database
    engine = create_engine("sqlite:///data/core/udahub.db")
    Session = sessionmaker(bind=engine)
    session = Session()

    articles = session.query(Knowledge).filter(
        Knowledge.account_id == account_id
    ).all()

    # Convert to documents
    documents = [
        Document(
            page_content=f"Title: {article.title}\n\nContent: {article.content}",
            metadata={
                "article_id": article.article_id,
                "title": article.title,
                "tags": article.tags,
                "account_id": article.account_id
            }
        )
        for article in articles
    ]

    session.close()

    # 2. Create embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 3. Build vector store
    vectorstore = InMemoryVectorStore.from_documents(
        documents=documents,
        embedding=embeddings
    )

    # 4. Create retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 3,
            "score_threshold": 0.7
        }
    )

    # 5. Create retriever tool
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

    return retriever_tool
```

**Usage in Workflow**:

```python
# In workflow.py
from agentic.tools.rag_setup import initialize_rag_system

# Initialize once at startup
retriever_tool = initialize_rag_system(account_id="cultpass")

# Pass to Resolver Agent
resolver_agent = create_react_agent(
    model=model,
    tools=[retriever_tool],
    state_schema=UDAHubState,
    prompt=SystemMessage(content=RESOLVER_PROMPT)
)
```

---

## Testing RAG System

### Unit Tests

**Test 1: Retrieval Accuracy**

```python
def test_retrieval_accuracy():
    """Test that retriever returns relevant articles."""
    query = "how to reset password"
    results = retriever.invoke(query)

    # Should return at least one result
    assert len(results) > 0

    # Should contain password-related content
    assert "password" in results[0].page_content.lower()
```

**Test 2: Similarity Threshold**

```python
def test_similarity_threshold():
    """Test that irrelevant queries return no results."""
    query = "how to fly a spaceship"  # Completely unrelated
    results = retriever.invoke(query)

    # Should return empty or very low similarity
    assert len(results) == 0 or results[0].metadata.get("score", 0) < 0.5
```

**Test 3: Tool Integration**

```python
def test_retriever_tool():
    """Test that retriever tool works correctly."""
    result = retriever_tool.invoke({"query": "login issues"})

    # Should return string with article content
    assert isinstance(result, str)
    assert "login" in result.lower()
```

### Integration Tests

**Test End-to-End RAG Flow**:

```python
def test_rag_end_to_end():
    """Test complete RAG flow from query to answer."""

    # Setup
    config = {"configurable": {"thread_id": "test_rag"}}

    # Invoke with login question
    result = orchestrator.invoke(
        {"messages": [HumanMessage(content="I can't login")]},
        config=config
    )

    # Check that retriever tool was called
    state = orchestrator.get_state(config)
    messages = state.values["messages"]

    # Find tool calls
    tool_calls = [
        msg for msg in messages
        if hasattr(msg, "tool_calls") and msg.tool_calls
    ]

    # Verify retriever was used
    assert any(
        "knowledge_retriever" in str(tc)
        for msg in tool_calls
        for tc in msg.tool_calls
    )

    # Check that answer was generated
    ai_messages = [msg for msg in messages if msg.type == "ai"]
    assert len(ai_messages) > 0

    # Check answer quality
    final_answer = ai_messages[-1].content
    assert "password" in final_answer.lower()
```

---

## Performance Optimization

### Caching Embeddings

**Problem**: Generating embeddings is slow and costly

**Solution**: Cache vector store to disk

```python
import pickle

# Save vector store
with open("vectorstore.pkl", "wb") as f:
    pickle.dump(vectorstore, f)

# Load vector store
with open("vectorstore.pkl", "rb") as f:
    vectorstore = pickle.load(f)
```

### Batch Processing

**Problem**: Generating embeddings one-by-one is slow

**Solution**: Batch embed all documents at once

```python
# Generate embeddings in batch (faster)
documents_text = [doc.page_content for doc in documents]
embeddings_list = embeddings.embed_documents(documents_text)

# Create vector store with pre-computed embeddings
# (requires manual vector store setup)
```

### Lazy Loading

**Problem**: RAG initialization slows down startup

**Solution**: Initialize RAG system only when first needed

```python
retriever_tool = None

def get_retriever_tool():
    """Lazy load retriever tool."""
    global retriever_tool
    if retriever_tool is None:
        retriever_tool = initialize_rag_system()
    return retriever_tool
```

---

## Future Enhancements

### 1. Hybrid Search
Combine semantic search with keyword search for better accuracy.

```python
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import BM25Retriever

# BM25 for keyword search
bm25_retriever = BM25Retriever.from_documents(documents)

# Ensemble retriever (combines semantic + keyword)
ensemble_retriever = EnsembleRetriever(
    retrievers=[retriever, bm25_retriever],
    weights=[0.7, 0.3]  # 70% semantic, 30% keyword
)
```

### 2. Re-ranking
Use cross-encoder to re-rank retrieved documents for better precision.

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker

reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")

compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=retriever
)
```

### 3. Query Expansion
Expand user query with synonyms for better retrieval.

```python
def expand_query(query: str) -> str:
    """Expand query with synonyms."""
    # Use LLM to generate expanded query
    expanded = llm.invoke(
        f"Expand this query with synonyms: {query}"
    )
    return expanded
```

### 4. Feedback Loop
Learn from user feedback to improve retrieval.

```python
def record_feedback(query: str, article_id: str, helpful: bool):
    """Record user feedback for continuous improvement."""
    # Store feedback in database
    # Use to fine-tune embeddings or adjust rankings
    pass
```

### 5. Multi-lingual Support
Support articles in multiple languages.

```python
# Use multi-lingual embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Translate query if needed
def translate_query(query: str, target_lang: str) -> str:
    # Use translation API
    pass
```

### 6. Dynamic Updates
Hot-reload articles without restarting system.

```python
def update_vector_store(new_article: Knowledge):
    """Add new article to vector store without rebuilding."""
    # Create document
    doc = Document(
        page_content=f"Title: {new_article.title}\n\n{new_article.content}",
        metadata={"article_id": new_article.article_id}
    )

    # Add to vector store
    vectorstore.add_documents([doc])
```

---

## Related Documentation

- **System Overview**: See `ARCHITECTURE.md`
- **Agent Details**: See `AGENT_SPECIFICATIONS.md` (Resolver Agent)
- **Data Flow**: See `DATA_FLOW.md`
- **Memory**: See `MEMORY_STRATEGY.md`
- **Diagrams**: See `DIAGRAMS.md`
