# reset_udahub.py
import os
from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
from langchain_core.messages import (
    SystemMessage,
    HumanMessage, 
)
from langgraph.graph.state import CompiledStateGraph

# Import all models to ensure they're registered
from data.models.udahub import (
    Base,
    Account,
    User,
    Ticket,
    TicketMetadata,
    TicketMessage,
    Knowledge,
    UserPreference,
    TicketResolution,
    ConversationSummary
)


def reset_db(db_path: str, echo: bool = True):
    """Drops the existing udahub.db file and recreates all tables including new memory tables."""

    # Remove the file if it exists
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"✅ Removed existing {db_path}")

    # Create a new engine and recreate tables
    engine = create_engine(f"sqlite:///{db_path}", echo=echo)
    Base.metadata.create_all(engine)
    print(f"✅ Recreated {db_path} with fresh schema")
    print(f"   Tables created: {', '.join(Base.metadata.tables.keys())}")


@contextmanager
def get_session(engine: Engine):
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()


def model_to_dict(instance):
    """Convert a SQLAlchemy model instance to a dictionary."""
    return {
        column.name: getattr(instance, column.name)
        for column in instance.__table__.columns
    }

def chat_interface(
    agent: CompiledStateGraph,
    ticket_id: str,
    show_logs: bool = False,
    show_stats: bool = True
):
    """
    Enhanced chat interface with better UX and optional logging display.
    
    Args:
        agent: Compiled LangGraph agent
        ticket_id: Unique ticket/thread identifier
        show_logs: Display agent routing and decisions in real-time
        show_stats: Show session statistics at the end
        
    Commands:
        - quit, exit, q: End conversation
        - /help: Show available commands
        - /history: View conversation history
        - /stats: Show current session statistics
        - /logs: Toggle log display
        - /export: Export conversation to file
    """
    from datetime import datetime
    from agentic.logging import get_logger
    
    # Initialize
    logger = get_logger()
    start_time = datetime.now()
    message_count = 0
    
    # Print welcome message
    print("\n" + "="*70)
    print("  UDA-Hub Support Assistant")
    print("="*70)
    print(f"  Session ID: {ticket_id}")
    print(f"  Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("  Type '/help' for commands or 'quit' to exit")
    print("="*70 + "\n")
    
    conversation_history = []
    
    while True:
        try:
            # Get user input with better prompt
            user_input = input("\n💬 You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ["quit", "exit", "q"]:
                print("\n" + "="*70)
                print("  Session Summary")
                print("="*70)
                duration = (datetime.now() - start_time).total_seconds()
                print(f"  Duration: {duration:.1f}s")
                print(f"  Messages exchanged: {message_count}")
                
                if show_stats:
                    metrics = logger.get_metrics(thread_id=ticket_id)
                    print(f"  Total events logged: {metrics['total_events']}")
                    print(f"  Escalations: {metrics['escalations']}")
                    print(f"  Errors: {metrics['errors']}")
                
                print("="*70)
                print("\n👋 Thank you for using UDA-Hub Support. Goodbye!\n")
                break
            
            elif user_input == "/help":
                print("\n📚 Available Commands:")
                print("  /help     - Show this help message")
                print("  /history  - View conversation history")
                print("  /stats    - Show session statistics")
                print("  /logs     - Toggle real-time log display")
                print("  /export   - Export conversation to file")
                print("  quit/q    - End conversation")
                continue
            
            elif user_input == "/history":
                print("\n📜 Conversation History:")
                print("-" * 70)
                for i, (role, msg, time) in enumerate(conversation_history, 1):
                    print(f"{i}. [{time}] {role}: {msg[:100]}{'...' if len(msg) > 100 else ''}")
                print("-" * 70)
                continue
            
            elif user_input == "/stats":
                metrics = logger.get_metrics(thread_id=ticket_id)
                print("\n📊 Session Statistics:")
                print("-" * 70)
                print(f"Total Events: {metrics['total_events']}")
                print(f"Events by Type: {metrics['events_by_type']}")
                print(f"Events by Agent: {metrics['events_by_agent']}")
                print(f"Resolutions: {metrics['resolutions']}")
                print(f"Escalations: {metrics['escalations']}")
                print(f"Errors: {metrics['errors']}")
                print("-" * 70)
                continue
            
            elif user_input == "/logs":
                show_logs = not show_logs
                print(f"\n🔄 Real-time logs: {'ON' if show_logs else 'OFF'}")
                continue
            
            elif user_input == "/export":
                export_file = f"conversation_{ticket_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                with open(export_file, 'w') as f:
                    f.write(f"UDA-Hub Conversation Export\n")
                    f.write(f"Session ID: {ticket_id}\n")
                    f.write(f"Exported: {datetime.now().isoformat()}\n")
                    f.write("="*70 + "\n\n")
                    for role, msg, time in conversation_history:
                        f.write(f"[{time}] {role}:\n{msg}\n\n")
                print(f"\n💾 Conversation exported to: {export_file}")
                continue
            
            # Process user message
            message_count += 1
            timestamp = datetime.now().strftime('%H:%M:%S')
            conversation_history.append(("You", user_input, timestamp))
            
            if show_logs:
                print(f"\n🔄 Processing your request...")
            
            # Invoke agent
            trigger = {
                "messages": [HumanMessage(content=user_input)],
                "ticket_metadata": {"ticket_id": ticket_id}
            }
            config = {
                "configurable": {
                    "thread_id": ticket_id,
                }
            }
            
            result = agent.invoke(input=trigger, config=config)
            
            # Get assistant response
            assistant_msg = result["messages"][-1].content
            timestamp = datetime.now().strftime('%H:%M:%S')
            conversation_history.append(("Assistant", assistant_msg, timestamp))
            
            # Display response with formatting
            print(f"\n🤖 Assistant: {assistant_msg}")
            
            # Show routing info if logs enabled
            if show_logs and result.get("classification"):
                classification = result["classification"]
                print(f"\n📋 Classification: {classification.get('issue_type')} | Urgency: {classification.get('urgency')}")
                if result.get("resolution"):
                    resolution = result["resolution"]
                    print(f"✅ Resolution confidence: {resolution.get('confidence', 0):.2f}")
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted. Type 'quit' to exit properly.")
            continue
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            logger.log_error(
                thread_id=ticket_id,
                agent_name="chat_interface",
                error_type=type(e).__name__,
                error_message=str(e),
                recovery_action="Continuing conversation"
            )
            continue


        