"""
Memory Management Module for UDA-Hub

Provides long-term memory capabilities including customer interaction history,
preferences, and resolved issues across sessions.
"""

from typing import Optional
from datetime import datetime, timedelta
from contextlib import contextmanager
from sqlalchemy import create_engine, desc
from sqlalchemy.orm import sessionmaker, Session
from data.models.udahub import (
    Ticket,
    TicketMetadata,
    TicketMessage,
    User,
    UserPreference,
    TicketResolution,
    ConversationSummary
)


class CustomerMemoryStore:
    """Manages long-term customer memory using persistent database storage."""
    
    def __init__(self, db_path: str = "data/core/udahub.db"):
        """Initialize memory store with database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.engine = create_engine(f"sqlite:///{db_path}")
        self.Session = sessionmaker(bind=self.engine)
    
    @contextmanager
    def _session(self):
        """Context manager for database sessions."""
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def get_user_ticket_history(
        self,
        user_id: str,
        limit: int = 10,
        include_current: bool = False
    ) -> list[dict]:
        """Get user's previous ticket history.
        
        Args:
            user_id: User identifier
            limit: Maximum number of tickets to return
            include_current: Include in-progress tickets
            
        Returns:
            List of ticket history dictionaries
        """
        with self._session() as session:
            query = session.query(Ticket, TicketMetadata).join(
                TicketMetadata, Ticket.ticket_id == TicketMetadata.ticket_id
            ).filter(Ticket.user_id == user_id)
            
            if not include_current:
                query = query.filter(TicketMetadata.status.in_(["resolved", "escalated"]))
            
            results = query.order_by(desc(Ticket.created_at)).limit(limit).all()
            
            history = []
            for ticket, metadata in results:
                history.append({
                    "ticket_id": ticket.ticket_id,
                    "created_at": ticket.created_at,
                    "channel": ticket.channel,
                    "status": metadata.status,
                    "issue_type": metadata.main_issue_type,
                    "tags": metadata.tags
                })
            
            return history
    
    def get_resolved_issues(
        self,
        user_id: str,
        limit: int = 10,
        days_back: Optional[int] = None
    ) -> list[dict]:
        """Get user's previously resolved issues with resolutions.
        
        Args:
            user_id: User identifier
            limit: Maximum number of issues to return
            days_back: Only include issues from last N days
            
        Returns:
            List of resolved issue dictionaries
        """
        with self._session() as session:
            query = session.query(
                Ticket, TicketMetadata, TicketResolution
            ).join(
                TicketMetadata, Ticket.ticket_id == TicketMetadata.ticket_id
            ).join(
                TicketResolution, Ticket.ticket_id == TicketResolution.ticket_id
            ).filter(
                Ticket.user_id == user_id,
                TicketMetadata.status == "resolved"
            )
            
            if days_back:
                cutoff = datetime.now() - timedelta(days=days_back)
                query = query.filter(Ticket.created_at >= cutoff)
            
            results = query.order_by(desc(Ticket.created_at)).limit(limit).all()
            
            resolved = []
            for ticket, metadata, resolution in results:
                resolved.append({
                    "ticket_id": ticket.ticket_id,
                    "created_at": ticket.created_at,
                    "issue_type": metadata.main_issue_type,
                    "tags": metadata.tags,
                    "resolution_method": resolution.resolution_method,
                    "confidence": resolution.confidence,
                    "articles_used": resolution.articles_used
                })
            
            return resolved
    
    def get_user_preferences(self, user_id: str) -> dict:
        """Get user preferences.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary of user preferences
        """
        with self._session() as session:
            preferences = session.query(UserPreference).filter_by(
                user_id=user_id
            ).all()
            
            prefs_dict = {}
            for pref in preferences:
                prefs_dict[pref.preference_key] = pref.preference_value
            
            return prefs_dict
    
    def set_user_preference(
        self,
        user_id: str,
        key: str,
        value: str
    ) -> None:
        """Set or update user preference.
        
        Args:
            user_id: User identifier
            key: Preference key
            value: Preference value
        """
        with self._session() as session:
            existing = session.query(UserPreference).filter_by(
                user_id=user_id,
                preference_key=key
            ).first()
            
            if existing:
                existing.preference_value = value
                existing.updated_at = datetime.now()
            else:
                from uuid import uuid4
                new_pref = UserPreference(
                    preference_id=str(uuid4()),
                    user_id=user_id,
                    preference_key=key,
                    preference_value=value
                )
                session.add(new_pref)
    
    def store_ticket_resolution(
        self,
        ticket_id: str,
        resolution_method: str,
        confidence: float,
        articles_used: Optional[str] = None,
        tool_results: Optional[str] = None
    ) -> None:
        """Store ticket resolution details for long-term memory.
        
        Args:
            ticket_id: Ticket identifier
            resolution_method: How the issue was resolved
            confidence: Resolution confidence score
            articles_used: Comma-separated article IDs
            tool_results: JSON string of tool execution results
        """
        with self._session() as session:
            from uuid import uuid4
            resolution = TicketResolution(
                resolution_id=str(uuid4()),
                ticket_id=ticket_id,
                resolution_method=resolution_method,
                confidence=confidence,
                articles_used=articles_used,
                tool_results=tool_results
            )
            session.add(resolution)
    
    def store_conversation_summary(
        self,
        ticket_id: str,
        summary: str,
        key_points: str,
        message_count: int
    ) -> None:
        """Store conversation summary for quick retrieval.
        
        Args:
            ticket_id: Ticket identifier
            summary: Conversation summary
            key_points: Key points extracted
            message_count: Number of messages in conversation
        """
        with self._session() as session:
            from uuid import uuid4
            summary_obj = ConversationSummary(
                summary_id=str(uuid4()),
                ticket_id=ticket_id,
                summary=summary,
                key_points=key_points,
                message_count=message_count
            )
            session.add(summary_obj)
    
    def get_conversation_summary(self, ticket_id: str) -> Optional[dict]:
        """Get conversation summary for a ticket.
        
        Args:
            ticket_id: Ticket identifier
            
        Returns:
            Summary dictionary or None
        """
        with self._session() as session:
            summary = session.query(ConversationSummary).filter_by(
                ticket_id=ticket_id
            ).first()
            
            if summary:
                return {
                    "ticket_id": summary.ticket_id,
                    "summary": summary.summary,
                    "key_points": summary.key_points,
                    "message_count": summary.message_count,
                    "created_at": summary.created_at
                }
            return None
    
    def find_similar_issues(
        self,
        user_id: str,
        issue_type: str,
        tags: list[str],
        limit: int = 5
    ) -> list[dict]:
        """Find similar past issues based on type and tags.
        
        Args:
            user_id: User identifier
            issue_type: Current issue type
            tags: Issue tags
            limit: Maximum results
            
        Returns:
            List of similar resolved issues
        """
        with self._session() as session:
            query = session.query(
                Ticket, TicketMetadata, TicketResolution
            ).join(
                TicketMetadata, Ticket.ticket_id == TicketMetadata.ticket_id
            ).join(
                TicketResolution, Ticket.ticket_id == TicketResolution.ticket_id
            ).filter(
                Ticket.user_id == user_id,
                TicketMetadata.status == "resolved",
                TicketMetadata.main_issue_type == issue_type
            )
            
            results = query.order_by(desc(Ticket.created_at)).limit(limit * 2).all()
            
            similar = []
            for ticket, metadata, resolution in results:
                if not metadata.tags:
                    continue
                
                ticket_tags = set(metadata.tags.split(','))
                tag_overlap = len(ticket_tags.intersection(set(tags)))
                
                if tag_overlap > 0:
                    similar.append({
                        "ticket_id": ticket.ticket_id,
                        "issue_type": metadata.main_issue_type,
                        "tags": metadata.tags,
                        "resolution_method": resolution.resolution_method,
                        "confidence": resolution.confidence,
                        "tag_overlap": tag_overlap
                    })
            
            similar.sort(key=lambda x: x["tag_overlap"], reverse=True)
            return similar[:limit]
    
    def get_recurring_issues(
        self,
        user_id: str,
        days_back: int = 90
    ) -> dict:
        """Identify recurring issues for a user.
        
        Args:
            user_id: User identifier
            days_back: Look back period in days
            
        Returns:
            Dictionary with issue type counts
        """
        with self._session() as session:
            cutoff = datetime.now() - timedelta(days=days_back)
            
            tickets = session.query(
                TicketMetadata.main_issue_type, 
                Ticket.ticket_id
            ).join(
                Ticket, TicketMetadata.ticket_id == Ticket.ticket_id
            ).filter(
                Ticket.user_id == user_id,
                Ticket.created_at >= cutoff
            ).all()
            
            issue_counts = {}
            for issue_type, _ in tickets:
                if issue_type:
                    issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
            
            return issue_counts


def get_memory_context_for_agent(
    user_id: str,
    current_issue_type: Optional[str] = None,
    current_tags: Optional[list[str]] = None,
    memory_store: Optional[CustomerMemoryStore] = None
) -> str:
    """Build memory context string for agent prompts.
    
    Args:
        user_id: User identifier
        current_issue_type: Current ticket issue type
        current_tags: Current ticket tags
        memory_store: Memory store instance (creates new if None)
        
    Returns:
        Formatted memory context string
    """
    if memory_store is None:
        memory_store = CustomerMemoryStore()
    
    context_parts = []
    
    # Get ticket history
    history = memory_store.get_user_ticket_history(user_id, limit=5)
    if history:
        context_parts.append(f"User has {len(history)} previous tickets:")
        for ticket in history[:3]:
            context_parts.append(
                f"  - {ticket['issue_type']} ({ticket['status']}) on {ticket['created_at']}"
            )
    
    # Get preferences
    preferences = memory_store.get_user_preferences(user_id)
    if preferences:
        context_parts.append(f"User preferences: {preferences}")
    
    # Get similar issues if available
    if current_issue_type and current_tags:
        similar = memory_store.find_similar_issues(
            user_id, current_issue_type, current_tags, limit=3
        )
        if similar:
            context_parts.append(f"Found {len(similar)} similar past issues:")
            for issue in similar:
                context_parts.append(
                    f"  - Resolved via {issue['resolution_method']} "
                    f"(confidence: {issue['confidence']:.2f})"
                )
    
    # Check for recurring issues
    recurring = memory_store.get_recurring_issues(user_id)
    if recurring:
        top_issues = sorted(recurring.items(), key=lambda x: x[1], reverse=True)[:3]
        if top_issues:
            context_parts.append("Recurring issue patterns:")
            for issue_type, count in top_issues:
                if count > 1:
                    context_parts.append(f"  - {issue_type}: {count} times")
    
    if context_parts:
        return "\n".join(["", "=== CUSTOMER MEMORY CONTEXT ==="] + context_parts + ["=" * 30, ""])
    return ""


__all__ = [
    "CustomerMemoryStore",
    "get_memory_context_for_agent"
]
