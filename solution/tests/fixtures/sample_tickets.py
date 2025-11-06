"""
Sample Tickets for Testing

Provides realistic ticket examples covering all issue types and scenarios.
"""

from langchain_core.messages import HumanMessage, AIMessage
from datetime import datetime


class SampleTickets:
    """Collection of sample tickets for testing."""
    
    LOGIN_PASSWORD_RESET = {
        "messages": [HumanMessage(content="I can't login to my account. I forgot my password and need to reset it.")],
        "ticket_metadata": {"ticket_id": "SAMPLE-LOGIN-001", "account_id": "cultpass", "user_email": "user1@example.com"},
        "expected_classification": {"issue_type": "login", "confidence": "high", "urgency": "medium"}
    }
    
    BOOKING_VIEW_RESERVATIONS = {
        "messages": [HumanMessage(content="Can you show me all my upcoming class reservations for this week?")],
        "ticket_metadata": {"ticket_id": "SAMPLE-BOOK-001", "account_id": "cultpass", "user_email": "yogi@example.com"},
        "expected_classification": {"issue_type": "booking_inquiry", "confidence": "high", "urgency": "low"}
    }
    
    BILLING_DOUBLE_CHARGE = {
        "messages": [HumanMessage(content="I was charged twice for my premium subscription this month! I need a refund!")],
        "ticket_metadata": {"ticket_id": "SAMPLE-BILL-002", "account_id": "cultpass", "user_email": "angry@example.com"},
        "expected_classification": {"issue_type": "billing_dispute", "confidence": "high", "urgency": "high"},
        "expected_escalation": True
    }

SAMPLE_KB_ARTICLES = [
    {"title": "How to Reset Your Password", "category": "login", "tags": ["password", "login", "reset"]},
    {"title": "How to Book a Class", "category": "booking", "tags": ["booking", "reservation", "class"]},
    {"title": "Subscription Plans Overview", "category": "subscription", "tags": ["subscription", "plans"]}
]
