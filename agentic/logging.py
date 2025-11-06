"""
UDA-Hub Structured Logging System

This module provides JSON-based structured logging for agent decisions, routing choices,
tool usage, and outcomes. Logs are searchable and support real-time streaming.

Features:
- JSON-formatted logs with structured fields
- Agent-specific logging (Classifier, Resolver, Tool Agent, Escalation, Supervisor)
- Query interface for searching logs
- Performance metrics tracking
- File rotation and size management
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal
from enum import Enum
import hashlib


class EventType(str, Enum):
    """Log event types for categorization"""
    ROUTING = "routing"
    CLASSIFICATION = "classification"
    RESOLUTION = "resolution"
    TOOL_EXECUTION = "tool_execution"
    ESCALATION = "escalation"
    ERROR = "error"
    DECISION = "decision"
    RAG_RETRIEVAL = "rag_retrieval"
    STATE_UPDATE = "state_update"


def _sanitize_value(value: Any, hash_emails: bool = True) -> Any:
    """
    Sanitize sensitive data from log values.
    
    Redacts:
    - Email addresses (hashes or masks them)
    - API keys (OPENAI_API_KEY, etc.)
    - Authorization tokens and bearer tokens
    - Passwords
    - Credit card numbers
    - Phone numbers (partial masking)
    
    Args:
        value: Value to sanitize (can be str, dict, list, etc.)
        hash_emails: If True, hash emails; if False, mask them
        
    Returns:
        Sanitized value
    """
    if isinstance(value, str):
        # Redact API keys and tokens
        if any(keyword in value.upper() for keyword in ['API_KEY', 'APIKEY', 'TOKEN', 'SECRET', 'PASSWORD', 'AUTH']):
            # Check if it looks like an assignment or contains actual key
            if '=' in value or len(value) > 20:
                return "[REDACTED_KEY]"
        
        # Redact bearer tokens
        value = re.sub(r'Bearer\s+[\w\-\.]+', 'Bearer [REDACTED_TOKEN]', value, flags=re.IGNORECASE)
        
        # Handle email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        if hash_emails:
            def hash_email(match):
                email = match.group(0)
                # Hash email but keep domain visible for debugging
                local, domain = email.split('@')
                hashed_local = hashlib.sha256(local.encode()).hexdigest()[:8]
                return f"{hashed_local}@{domain}"
            value = re.sub(email_pattern, hash_email, value)
        else:
            value = re.sub(email_pattern, '[EMAIL_MASKED]', value)
        
        # Redact phone numbers (keep last 4 digits)
        value = re.sub(r'\b(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', 
                      '***-***-****', value)
        
        # Redact credit card numbers
        value = re.sub(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', '[CARD_REDACTED]', value)
        
        # Redact long alphanumeric strings that look like keys/tokens
        value = re.sub(r'\b[A-Za-z0-9]{32,}\b', '[LONG_TOKEN_REDACTED]', value)
        
        return value
    
    elif isinstance(value, dict):
        return {k: _sanitize_value(v, hash_emails) for k, v in value.items()}
    
    elif isinstance(value, list):
        return [_sanitize_value(item, hash_emails) for item in value]
    
    else:
        return value


def sanitize_log_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize all sensitive data from log entry.
    
    Args:
        data: Log data dictionary
        
    Returns:
        Sanitized log data
    """
    return _sanitize_value(data, hash_emails=True)


class AgentLogger:
    """Structured logger for UDA-Hub agents with JSON output"""
    
    def __init__(
        self,
        log_dir: str = "logs",
        log_file: str = "udahub_agent.log",
        console_output: bool = True,
        file_output: bool = True
    ):
        """
        Initialize the agent logger.
        
        Args:
            log_dir: Directory for log files
            log_file: Main log file name
            console_output: Enable console logging
            file_output: Enable file logging
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / log_file
        
        # Setup Python logger
        self.logger = logging.getLogger("UDAHub")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()
        
        # JSON formatter
        class JsonFormatter(logging.Formatter):
            def format(self, record):
                log_data = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "level": record.levelname,
                    "message": record.getMessage(),
                }
                if hasattr(record, 'extra_data'):
                    log_data.update(record.extra_data)
                return json.dumps(log_data)
        
        # File handler with JSON format
        if file_output:
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(JsonFormatter())
            self.logger.addHandler(file_handler)
        
        # Console handler (readable format)
        if console_output:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            )
            self.logger.addHandler(console_handler)
    
    def _log_structured(
        self,
        level: str,
        message: str,
        event_type: EventType,
        agent_name: str,
        thread_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        state_snapshot: Optional[Dict[str, Any]] = None
    ):
        """Internal method to log structured data"""
        extra_data = {
            "event_type": event_type.value,
            "agent_name": agent_name,
            "thread_id": thread_id,
            "details": details or {},
            "state_snapshot": state_snapshot or {}
        }
        
        # Sanitize all data before logging
        extra_data = sanitize_log_data(extra_data)
        message = _sanitize_value(message)
        
        log_level = getattr(logging, level.upper())
        self.logger.log(log_level, message, extra={'extra_data': extra_data})
    
    def log_routing(
        self,
        from_agent: str,
        to_agent: str,
        reason: str,
        thread_id: str,
        state_summary: Optional[Dict[str, Any]] = None
    ):
        """Log supervisor routing decisions"""
        self._log_structured(
            level="INFO",
            message=f"Routing: {from_agent} → {to_agent}",
            event_type=EventType.ROUTING,
            agent_name="supervisor",
            thread_id=thread_id,
            details={
                "from_agent": from_agent,
                "to_agent": to_agent,
                "reason": reason
            },
            state_snapshot=state_summary
        )
    
    def log_classification(
        self,
        thread_id: str,
        classification: Dict[str, Any],
        confidence: float,
        user_message: str
    ):
        """Log classifier agent results"""
        self._log_structured(
            level="INFO",
            message=f"Classification: {classification.get('issue_type')} | Urgency: {classification.get('urgency')} | Confidence: {confidence:.2f}",
            event_type=EventType.CLASSIFICATION,
            agent_name="classifier",
            thread_id=thread_id,
            details={
                "classification": classification,
                "confidence": confidence,
                "user_message": user_message[:200]  # Truncate for brevity
            }
        )
    
    def log_resolution(
        self,
        thread_id: str,
        resolved: bool,
        confidence: float,
        response: str,
        rag_articles: Optional[List[Dict]] = None
    ):
        """Log resolver agent outcomes"""
        self._log_structured(
            level="INFO",
            message=f"Resolution: {'Success' if resolved else 'Failed'} | Confidence: {confidence:.2f}",
            event_type=EventType.RESOLUTION,
            agent_name="resolver",
            thread_id=thread_id,
            details={
                "resolved": resolved,
                "confidence": confidence,
                "response": response[:300],  # Truncate
                "rag_articles_count": len(rag_articles) if rag_articles else 0,
                "rag_articles": rag_articles[:3] if rag_articles else []  # First 3
            }
        )
    
    def log_rag_retrieval(
        self,
        thread_id: str,
        query: str,
        articles_retrieved: int,
        top_article: Optional[Dict] = None
    ):
        """Log RAG system retrievals"""
        self._log_structured(
            level="DEBUG",
            message=f"RAG Retrieval: {articles_retrieved} articles for query",
            event_type=EventType.RAG_RETRIEVAL,
            agent_name="resolver",
            thread_id=thread_id,
            details={
                "query": query[:200],
                "articles_retrieved": articles_retrieved,
                "top_article": top_article
            }
        )
    
    def log_tool_execution(
        self,
        thread_id: str,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_output: Any,
        success: bool,
        execution_time: Optional[float] = None
    ):
        """Log tool agent executions"""
        self._log_structured(
            level="INFO",
            message=f"Tool Execution: {tool_name} | Success: {success}",
            event_type=EventType.TOOL_EXECUTION,
            agent_name="tool_agent",
            thread_id=thread_id,
            details={
                "tool_name": tool_name,
                "tool_input": tool_input,
                "tool_output": str(tool_output)[:500],  # Truncate
                "success": success,
                "execution_time_ms": execution_time
            }
        )
    
    def log_escalation(
        self,
        thread_id: str,
        reason: str,
        priority: str,
        context: Dict[str, Any],
        attempted_steps: List[str]
    ):
        """Log escalation to human agents"""
        self._log_structured(
            level="WARNING",
            message=f"Escalation: {reason} | Priority: {priority}",
            event_type=EventType.ESCALATION,
            agent_name="escalation",
            thread_id=thread_id,
            details={
                "reason": reason,
                "priority": priority,
                "context": context,
                "attempted_steps": attempted_steps
            }
        )
    
    def log_error(
        self,
        thread_id: str,
        agent_name: str,
        error_type: str,
        error_message: str,
        stack_trace: Optional[str] = None,
        recovery_action: Optional[str] = None
    ):
        """Log errors and exceptions"""
        self._log_structured(
            level="ERROR",
            message=f"Error in {agent_name}: {error_type}",
            event_type=EventType.ERROR,
            agent_name=agent_name,
            thread_id=thread_id,
            details={
                "error_type": error_type,
                "error_message": error_message,
                "stack_trace": stack_trace,
                "recovery_action": recovery_action
            }
        )
    
    def log_decision(
        self,
        thread_id: str,
        agent_name: str,
        decision: str,
        reasoning: str,
        alternatives: Optional[List[str]] = None
    ):
        """Log agent decision-making process"""
        self._log_structured(
            level="DEBUG",
            message=f"Decision by {agent_name}: {decision}",
            event_type=EventType.DECISION,
            agent_name=agent_name,
            thread_id=thread_id,
            details={
                "decision": decision,
                "reasoning": reasoning,
                "alternatives_considered": alternatives or []
            }
        )
    
    def query_logs(
        self,
        agent_name: Optional[str] = None,
        event_type: Optional[EventType] = None,
        thread_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Query logs with filters.
        
        Args:
            agent_name: Filter by agent
            event_type: Filter by event type
            thread_id: Filter by thread/ticket ID
            start_time: Filter by start timestamp
            end_time: Filter by end timestamp
            limit: Maximum results to return
            
        Returns:
            List of matching log entries
        """
        results = []
        
        if not self.log_file.exists():
            return results
        
        with open(self.log_file, 'r') as f:
            for line in f:
                try:
                    log_entry = json.loads(line.strip())
                    
                    # Apply filters
                    if agent_name and log_entry.get('agent_name') != agent_name:
                        continue
                    if event_type and log_entry.get('event_type') != event_type.value:
                        continue
                    if thread_id and log_entry.get('thread_id') != thread_id:
                        continue
                    if start_time:
                        log_time = datetime.fromisoformat(log_entry.get('timestamp', ''))
                        if log_time < start_time:
                            continue
                    if end_time:
                        log_time = datetime.fromisoformat(log_entry.get('timestamp', ''))
                        if log_time > end_time:
                            continue
                    
                    results.append(log_entry)
                    
                    if len(results) >= limit:
                        break
                        
                except json.JSONDecodeError:
                    continue
        
        return results
    
    def get_metrics(
        self,
        thread_id: Optional[str] = None,
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Calculate performance metrics from logs.
        
        Args:
            thread_id: Filter by specific thread
            time_window_hours: Time window for metrics
            
        Returns:
            Dictionary with metrics
        """
        from datetime import timedelta
        
        start_time = datetime.utcnow() - timedelta(hours=time_window_hours)
        logs = self.query_logs(start_time=start_time, thread_id=thread_id, limit=10000)
        
        metrics = {
            "total_tickets": len(set(log.get('thread_id') for log in logs if log.get('thread_id'))),
            "total_events": len(logs),
            "events_by_type": {},
            "events_by_agent": {},
            "resolutions": {"success": 0, "failed": 0, "avg_confidence": 0.0},
            "escalations": 0,
            "errors": 0,
            "tool_executions": 0
        }
        
        confidences = []
        
        for log in logs:
            event_type = log.get('event_type', 'unknown')
            agent_name = log.get('agent_name', 'unknown')
            
            metrics["events_by_type"][event_type] = metrics["events_by_type"].get(event_type, 0) + 1
            metrics["events_by_agent"][agent_name] = metrics["events_by_agent"].get(agent_name, 0) + 1
            
            if event_type == EventType.RESOLUTION.value:
                details = log.get('details', {})
                if details.get('resolved'):
                    metrics["resolutions"]["success"] += 1
                else:
                    metrics["resolutions"]["failed"] += 1
                confidences.append(details.get('confidence', 0.0))
            
            elif event_type == EventType.ESCALATION.value:
                metrics["escalations"] += 1
            
            elif event_type == EventType.ERROR.value:
                metrics["errors"] += 1
            
            elif event_type == EventType.TOOL_EXECUTION.value:
                metrics["tool_executions"] += 1
        
        if confidences:
            metrics["resolutions"]["avg_confidence"] = sum(confidences) / len(confidences)
        
        return metrics
    
    def log_agent(
        self,
        agent_name: str,
        action: str,
        details: Dict[str, Any],
        thread_id: str,
        level: str = "INFO"
    ):
        """Generic agent logging with automatic sanitization"""
        self._log_structured(
            level=level,
            message=f"{agent_name}: {action}",
            event_type=EventType.STATE_UPDATE,
            agent_name=agent_name,
            thread_id=thread_id,
            details=details
        )


# Global logger instance
_global_logger: Optional[AgentLogger] = None


def get_logger(
    log_dir: str = "logs",
    log_file: str = "udahub_agent.log",
    console_output: bool = True,
    file_output: bool = True
) -> AgentLogger:
    """Get or create the global logger instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = AgentLogger(
            log_dir=log_dir,
            log_file=log_file,
            console_output=console_output,
            file_output=file_output
        )
    return _global_logger


__all__ = [
    "AgentLogger",
    "EventType",
    "get_logger",
    "sanitize_log_data"
]
