# Tools Documentation

## Overview

UDA-Hub integrates two main types of tools:
1. **RAG Retriever Tool**: Knowledge base search and retrieval
2. **Database Tools**: CultPass data operations

---

## RAG Retriever Tool

**Purpose**: Search knowledge base for relevant articles to answer user questions.

**Technology**:
- OpenAI Embeddings (text-embedding-3-small)
- InMemoryVectorStore for vector storage
- Semantic similarity search

**Knowledge Base Structure**:
Location: `data/external/cultpass_articles.jsonl`

Each article contains:
```json
{
  "title": "Article Title",
  "content": "Full article content...",
  "category": "login|booking|billing|subscription|account",
  "tags": ["tag1", "tag2"],
  "last_updated": "2025-11-01"
}
```

**Usage**:
```python
from agentic.tools import initialize_rag_system

retriever = initialize_rag_system()
results = retriever.invoke("How do I reset my password?")

# Results format
[
    {
        "page_content": "To reset your password: ...",
        "metadata": {
            "title": "Password Reset Guide",
            "category": "login",
            "similarity_score": 0.92
        }
    }
]
```

**Configuration**:
- Top-k results: 3 (configurable)
- Minimum similarity threshold: 0.6
- Embedding dimension: 1536

**Best Practices**:
- Keep articles focused on single topics
- Use clear, actionable language
- Include step-by-step instructions
- Tag articles comprehensively
- Update regularly based on user queries

---

## CultPass Database Tools

### Read Tools

#### 1. get_user_by_email
**Purpose**: Retrieve user information by email address

**Parameters**:
- `email` (str): User's email address

**Returns**:
```python
{
    "user_id": 1,
    "email": "user@example.com",
    "name": "User Name",
    "is_blocked": False,
    "created_at": "2025-01-15T00:00:00Z"
}
```

**Example**:
```python
from agentic.tools.cultpass_read_tools import get_user_by_email

user = get_user_by_email("user@example.com")
```

**Use Cases**:
- Verify user exists
- Check account status
- Get user ID for other operations

---

#### 2. get_user_subscriptions
**Purpose**: Get user's subscription details

**Parameters**:
- `user_id` (int): User's ID

**Returns**:
```python
{
    "subscription_id": 1,
    "user_id": 1,
    "tier": "premium",  # basic|premium|elite
    "status": "active",  # active|expired|cancelled
    "credits_remaining": 8,
    "credits_total": 12,
    "start_date": "2025-11-01",
    "renewal_date": "2025-12-01"
}
```

**Example**:
```python
from agentic.tools.cultpass_read_tools import get_user_subscriptions

subscription = get_user_subscriptions(user_id=1)
```

**Use Cases**:
- Check subscription status
- Verify credit availability
- Confirm tier and features

---

#### 3. get_user_reservations
**Purpose**: List user's reservations

**Parameters**:
- `user_id` (int): User's ID
- `status` (str, optional): Filter by status (upcoming|past|cancelled|all)

**Returns**:
```python
[
    {
        "reservation_id": 101,
        "user_id": 1,
        "experience_id": 5,
        "experience_name": "Vinyasa Yoga",
        "date": "2025-11-10",
        "time": "18:00",
        "center": "Indiranagar",
        "status": "confirmed",  # confirmed|cancelled|completed
        "credits_used": 1
    }
]
```

**Example**:
```python
from agentic.tools.cultpass_read_tools import get_user_reservations

# Get all upcoming reservations
reservations = get_user_reservations(user_id=1, status="upcoming")
```

**Use Cases**:
- Show upcoming bookings
- Check booking history
- Find specific reservation for cancellation

---

#### 4. get_experience_details
**Purpose**: Get details about a class/experience

**Parameters**:
- `experience_id` (int): Experience ID

**Returns**:
```python
{
    "experience_id": 5,
    "name": "Vinyasa Yoga",
    "category": "yoga",
    "is_premium": False,
    "available_slots": 15,
    "duration_minutes": 60,
    "centers": ["Indiranagar", "Koramangala", "HSR Layout"]
}
```

**Example**:
```python
from agentic.tools.cultpass_read_tools import get_experience_details

experience = get_experience_details(experience_id=5)
```

**Use Cases**:
- Show class information
- Check availability
- Verify if premium subscription needed

---

### Write Tools

#### 1. cancel_reservation
**Purpose**: Cancel a user's reservation

**Parameters**:
- `reservation_id` (int): Reservation to cancel
- `user_id` (int): User requesting cancellation (for verification)

**Returns**:
```python
{
    "success": True,
    "reservation_id": 101,
    "credits_refunded": 1,
    "cancellation_time": "2025-11-05T12:00:00Z",
    "message": "Reservation cancelled successfully"
}
```

**Business Rules**:
- Cancellations >4 hours before class: Full credit refund
- Cancellations <4 hours before class: No refund
- Cannot cancel completed classes

**Example**:
```python
from agentic.tools.cultpass_write_tools import cancel_reservation

result = cancel_reservation(reservation_id=101, user_id=1)
```

**Error Handling**:
- Reservation not found → Error
- Wrong user → Permission denied
- Too late to cancel → No refund warning

---

#### 2. update_user_preferences
**Purpose**: Update user notification and app preferences

**Parameters**:
- `user_id` (int): User ID
- `preferences` (dict): Settings to update

**Preferences**:
```python
{
    "email_notifications": True,
    "sms_notifications": False,
    "push_notifications": True,
    "reminder_hours_before": 2,
    "preferred_centers": ["Indiranagar", "Koramangala"]
}
```

**Returns**:
```python
{
    "success": True,
    "updated_fields": ["email_notifications", "reminder_hours_before"],
    "message": "Preferences updated successfully"
}
```

**Example**:
```python
from agentic.tools.cultpass_write_tools import update_user_preferences

result = update_user_preferences(
    user_id=1,
    preferences={"email_notifications": True}
)
```

---

## Tool Integration with Agents

### Resolver Agent + RAG Tool

```python
# Resolver uses RAG to answer knowledge-based questions
state = {
    "messages": [HumanMessage(content="How do I reset my password?")],
    "classification": {"issue_type": "login"}
}

# RAG retriever automatically invoked by Resolver
# Returns knowledge base articles
# Resolver generates answer from articles
```

### Tool Agent + Database Tools

```python
# Tool Agent uses database tools for data operations
state = {
    "messages": [HumanMessage(content="Show my reservations")],
    "classification": {"issue_type": "booking_inquiry"},
    "ticket_metadata": {"user_email": "user@example.com"}
}

# Tool Agent:
# 1. Identifies need for get_user_reservations
# 2. Gets user_id from email
# 3. Executes get_user_reservations(user_id)
# 4. Formats results for user
```

---

## Tool Security

### Data Access Controls
- User can only access their own data
- Email verification required for user lookup
- No direct SQL execution (ORM only)
- Audit logging for all write operations

### Error Handling
- Never expose internal errors to users
- Log detailed errors for debugging
- Provide user-friendly error messages
- Graceful degradation on tool failures

### Rate Limiting
- Limit tool executions per user per minute
- Prevent abuse of database operations
- Queue long-running operations

---

## Tool Performance

### Optimization
- Database query optimization with indexes
- Connection pooling for database
- Caching for frequently accessed data
- Async operations where possible

### Monitoring
- Track tool execution time
- Monitor success/failure rates
- Alert on performance degradation
- Log all tool invocations

**Target Metrics**:
- RAG retrieval: <500ms
- Database queries: <200ms
- Tool success rate: >95%
- Average confidence: >0.75

---

## Adding New Tools

### Process
1. Define tool function in appropriate module
2. Add tool schema and description
3. Register with Tool Agent
4. Write unit tests
5. Update documentation
6. Add to knowledge base

### Example
```python
# agentic/tools/cultpass_write_tools.py

def refund_credits(user_id: int, credits: int, reason: str) -> dict:
    """
    Refund credits to user account.
    
    Args:
        user_id: User receiving refund
        credits: Number of credits to refund
        reason: Reason for refund
    
    Returns:
        dict: Refund confirmation with new balance
    """
    # Implementation
    pass
```

---

## Testing Tools

See `solution/tests/test_agents.py` for tool testing examples:
- Mock database responses
- Test error handling
- Verify authorization
- Check data validation

**Example Test**:
```python
def test_cancel_reservation_success():
    with patch('agentic.tools.cultpass_write_tools.cancel_reservation') as mock:
        mock.return_value = {"success": True, "credits_refunded": 1}
        
        result = cancel_reservation(reservation_id=101, user_id=1)
        assert result["success"] is True
```

---

## Troubleshooting

### RAG Tool Issues
- **No results**: Expand knowledge base, lower similarity threshold
- **Poor quality**: Improve article content, add examples
- **Wrong results**: Better tagging, refine queries

### Database Tool Issues
- **Connection errors**: Check database availability, connection string
- **Permission denied**: Verify user authentication
- **Data not found**: Validate IDs, check data existence

---

For implementation details, see:
- `agentic/tools/rag_setup.py` - RAG implementation
- `agentic/tools/cultpass_read_tools.py` - Read operations
- `agentic/tools/cultpass_write_tools.py` - Write operations
- `solution/tests/test_rag.py` - RAG tests
