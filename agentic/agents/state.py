"""
UDA-Hub State Schema

This module defines the shared state schema for all agents in the UDA-Hub
multi-agent system. The UDAHubState class extends MessagesState with additional
fields for ticket processing, classification, resolution, tool results, and escalation.
"""

from typing import Annotated
from typing_extensions import TypedDict
import operator
from langchain_core.messages import AnyMessage
from langgraph.graph import MessagesState


class UDAHubState(MessagesState):
    """State schema for UDA-Hub multi-agent system.

    Extends MessagesState with additional fields for ticket processing,
    classification, resolution, tool results, and escalation.

    Attributes:
        messages: List of chat messages accumulated throughout the conversation.
            Uses operator.add to append new messages rather than replacing.

        ticket_metadata: Dictionary containing:
            - ticket_id (str): Unique ticket identifier
            - account_id (str): Account identifier (e.g., "cultpass")
            - user_id (str): UDA-Hub user ID
            - channel (str): Source channel ("email", "chat", "api")
            - urgency (str | None): Urgency level ("low", "medium", "high", "critical")

        classification: Dictionary from Classifier Agent containing:
            - issue_type (str): One of ["technical", "billing", "account", "booking", "general"]
            - urgency (str): One of ["low", "medium", "high", "critical"]
            - complexity (str): One of ["simple", "moderate", "complex"]
            - tags (list[str]): Relevant tags for search
            - confidence (float): Classification confidence (0-1)

        resolution: Dictionary from Resolver Agent containing:
            - resolved (bool): Whether ticket was resolved
            - confidence (float): Resolution confidence (0-1)
            - answer (str | None): Generated response
            - articles_used (list[str]): Article IDs used
            - escalation_reason (str | None): Why escalation is needed

        tool_results: Dictionary from Tool Agent containing:
            - Tool execution results with structured data
            - Format depends on which tool was executed

        escalation: Dictionary from Escalation Agent containing:
            - summary (str): Concise issue summary
            - attempted_steps (list[str]): Resolution attempts
            - priority (str): Escalation priority ("P1", "P2", "P3", "P4")
            - recommended_action (str): Next steps for human agent
            - context (dict): Relevant user/ticket information
    """

    # Messages accumulated using operator.add (inherited from MessagesState)
    messages: Annotated[list[AnyMessage], operator.add]

    # Ticket metadata (required)
    ticket_metadata: dict

    # Classification results from Classifier Agent (optional)
    classification: dict | None

    # Resolution results from Resolver Agent (optional)
    resolution: dict | None

    # Tool execution results from Tool Agent (optional)
    tool_results: dict | None

    # Escalation summary from Escalation Agent (optional)
    escalation: dict | None
    
    # Routing history to track consecutive agent visits (optional)
    routing_history: list[str] | None
    
    # Remaining steps for agent execution (required by LangGraph)
    # Used to limit the number of steps the react agent can take
    # Calculated as recursion_limit - total_steps_taken
    remaining_steps: int
