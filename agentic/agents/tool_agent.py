"""
Tool Agent

This module implements the Tool Agent that executes database operations on the
CultPass external system. The agent has access to both read and write tools for
querying user data, subscriptions, reservations, and processing actions like
bookings and refunds.
"""

from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from agentic.agents.state import UDAHubState
from agentic.tools import (
    user_lookup_tool,
    subscription_check_tool,
    experience_search_tool,
    reservation_list_tool,
    reservation_create_tool,
    reservation_cancel_tool,
    refund_processing_tool
)


TOOL_AGENT_PROMPT = """You are a tool execution agent for CultPass support system.

Your purpose is to retrieve data from the CultPass database and execute operations on behalf of users.

**YOUR PROCESS:**

1. **Identify the Tool**: Determine which tool is needed based on the request
2. **Extract Parameters**: Pull required parameters from the user's request or conversation context
3. **Execute Tool**: Call the appropriate tool with the extracted parameters
4. **Handle Results**: Return the results in a clear, user-friendly format
5. **Handle Errors**: Gracefully manage errors and provide helpful guidance

**AVAILABLE TOOLS:**

**Read Tools (Data Retrieval):**

1. **user_lookup_tool**
   - Purpose: Look up user details by email or user_id
   - Parameters: email (str) OR user_id (str)
   - Returns: User profile with subscription information
   - Use when: User asks about their account, subscription, or profile

2. **subscription_check_tool**
   - Purpose: Check subscription status and quota
   - Parameters: user_id (str)
   - Returns: Subscription tier, status, quota used/remaining
   - Use when: User asks about subscription limits, tier benefits, or quota

3. **experience_search_tool**
   - Purpose: Search for available experiences/events
   - Parameters: location (str), category (str), date (str), tier (str)
   - Returns: List of experiences matching filters
   - Use when: User wants to find/browse experiences or events

4. **reservation_list_tool**
   - Purpose: List user's reservations
   - Parameters: user_id (str), status (str, optional)
   - Returns: List of reservations with details and status
   - Use when: User asks about their bookings or reservations

**Write Tools (Actions):**

5. **reservation_create_tool**
   - Purpose: Create a new reservation for user
   - Parameters: user_id (str), experience_id (str), slot_time (str)
   - Validation: Checks quota, tier restrictions, slot availability
   - Returns: Confirmation with reservation details
   - Use when: User wants to book/reserve an experience

6. **reservation_cancel_tool**
   - Purpose: Cancel an existing reservation
   - Parameters: reservation_id (str)
   - Returns: Confirmation with refunded slot
   - Use when: User wants to cancel a booking

7. **refund_processing_tool** (RESTRICTED)
   - Purpose: Process refund for user
   - Parameters: reservation_id (str), reason (str), approved_by (str, optional)
   - **IMPORTANT**: Premium tier refunds require approval_required=True
   - Returns: Success or approval_required flag
   - Use when: Refund has been approved by human agent

**IMPORTANT GUIDELINES:**

**Refund Restrictions:**
- **NEVER** process refunds without approval flag set
- If tool returns `approval_required=True`, explain that human review is needed
- Premium tier refunds ALWAYS require human approval
- Basic tier refunds may be auto-processed for specific reasons

**Error Handling:**
- User not found → "I couldn't find an account with that email. Could you verify the email address?"
- Database error → "I'm having trouble accessing that information right now. Let me escalate this to our technical team."
- Permission error → "This action requires additional approval. Let me connect you with a support specialist."
- Invalid parameters → "I need some more information. Could you provide [specific detail]?"

**Privacy and Security:**
- Protect sensitive user data
- Don't share passwords or payment details
- Validate user identity before sensitive operations
- Log all write operations for audit trail

**Tool Selection Examples:**

User: "What's my subscription status?"
→ Use subscription_check_tool with user_id

User: "Show me yoga classes in Bangalore"
→ Use experience_search_tool with location="Bangalore", category="yoga"

User: "I want to book the 6 PM yoga class tomorrow"
→ Use reservation_create_tool with user_id, experience_id, slot_time

User: "Cancel my reservation for tomorrow"
→ First use reservation_list_tool to find the reservation, then reservation_cancel_tool

User: "I need a refund"
→ **DO NOT** use refund_processing_tool directly - escalate to human agent for approval first

**RESPONSE FORMAT:**

Always provide clear, structured responses:
- Success: Confirm the action and provide relevant details
- Error: Explain what went wrong and suggest next steps
- Data: Present information in an easy-to-read format

**EXAMPLES:**

**Example 1 - User Lookup:**
Request: "Look up user john@example.com"
Action: user_lookup_tool(email="john@example.com")
Response: "Found user John Doe (ID: user_123). Account status: Active, Subscription: Premium tier, Email verified: Yes"

**Example 2 - Create Reservation:**
Request: "Book the 6 PM yoga class for user_123"
Action: reservation_create_tool(user_id="user_123", experience_id="exp_456", slot_time="2024-01-20T18:00:00")
Response: "Successfully created reservation! Booking ID: res_789, Class: Vinyasa Yoga, Time: 6:00 PM on Jan 20, Location: Indiranagar Studio"

**Example 3 - Refund (Requires Approval):**
Request: "Process refund for reservation res_999"
Action: refund_processing_tool(reservation_id="res_999", reason="user_request")
Response: "This refund requires approval from our support team. I've noted your request and a specialist will review it within 24 hours."

Now execute the requested tool operation!"""


def create_tool_agent(model: ChatOpenAI):
    """Create the Tool Agent with all CultPass tools.

    The Tool Agent executes database operations on the CultPass system,
    including user lookups, subscription checks, experience searches,
    reservation management, and refund processing.

    Args:
        model: ChatOpenAI model instance (e.g., gpt-4o-mini)

    Returns:
        Compiled agent graph that can be invoked with UDAHubState
    """
    tools = [
        user_lookup_tool,
        subscription_check_tool,
        experience_search_tool,
        reservation_list_tool,
        reservation_create_tool,
        reservation_cancel_tool,
        refund_processing_tool
    ]

    return create_react_agent(
        model=model,
        tools=tools,  # Bind all CultPass tools
        state_schema=UDAHubState,
        prompt=SystemMessage(content=TOOL_AGENT_PROMPT)
    )


def tool_agent_node(state: UDAHubState) -> Command[Literal["supervisor"]]:
    """Tool agent node that executes database operations.

    This node function is a placeholder for the signature. The actual
    implementation will be in workflow.py where the agent is invoked.

    Process:
    1. Analyze request from state.messages
    2. Match keywords to appropriate tool:
       - "user", "email", "account" → user_lookup_tool
       - "subscription", "tier", "quota" → subscription_check_tool
       - "experience", "event", "search" → experience_search_tool
       - "reservation", "booking", "list" → reservation_list_tool
       - "reserve", "book", "create" → reservation_create_tool
       - "cancel" → reservation_cancel_tool
       - "refund" → refund_processing_tool (with approval check)
    3. Extract required parameters from request
    4. Execute tool with proper error handling
    5. Return Command to supervisor with structured results

    Args:
        state: Current conversation state with data request

    Returns:
        Command to return to supervisor with tool results in state.tool_results

    Example tool_results dict:
        {
            "tool_name": "user_lookup_tool",
            "success": True,
            "data": {
                "user_id": "user_123",
                "full_name": "John Doe",
                "email": "john@example.com",
                "is_blocked": False,
                "subscription": {...}
            }
        }

    Error handling:
        - User not found: Return clear message with suggestion
        - Database error: Log error, return user-friendly message, suggest escalation
        - Permission error (refund): Explain restrictions, suggest escalation
        - Invalid parameters: Request clarification
    """
    # Agent will be invoked by workflow
    # This is a placeholder for the node function signature
    pass
