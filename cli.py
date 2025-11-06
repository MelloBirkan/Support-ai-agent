#!/usr/bin/env python3
"""
UDA-Hub CLI - Interactive Command-Line Interface

Interactive CLI for testing the UDA-Hub multi-agent customer support system.
Provides real-time chat interface with logging, statistics, and conversation management.

Usage:
    python cli.py [options]
    
Options:
    --session-id ID     Custom session ID (default: auto-generated)
    --show-logs         Enable real-time logging display
    --no-stats          Disable session statistics at end
    --account ACCOUNT   Account ID for RAG system (default: cultpass)
    --memory            Enable long-term memory features
    
Commands (during chat):
    /help               Show available commands
    /history            View conversation history
    /stats              Show session statistics
    /logs               Toggle real-time log display
    /export             Export conversation to file
    quit, exit, q       End conversation
    
Examples:
    python cli.py
    python cli.py --show-logs --session-id TEST-001
    python cli.py --memory --account cultpass
"""

import argparse
import sys
from datetime import datetime
from dotenv import load_dotenv

# Import orchestrator and utilities
from agentic.workflow import create_orchestrator
from utils import chat_interface
from agentic.logging import get_logger


def generate_session_id() -> str:
    """Generate a unique session ID with timestamp."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"CLI-{timestamp}"


def print_banner():
    """Print CLI startup banner."""
    print("\n" + "=" * 70)
    print("  UDA-Hub Multi-Agent Support System - CLI")
    print("=" * 70)
    print("  A supervisor-based multi-agent workflow for customer support")
    print("  Agents: Classifier → Resolver → Tool Agent → Escalation")
    print("=" * 70 + "\n")


def main():
    """Main CLI entry point."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="UDA-Hub Interactive CLI - Test the multi-agent support system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--session-id',
        type=str,
        default=None,
        help='Custom session ID (default: auto-generated with timestamp)'
    )
    parser.add_argument(
        '--show-logs',
        action='store_true',
        help='Enable real-time logging display during chat'
    )
    parser.add_argument(
        '--no-stats',
        action='store_true',
        help='Disable session statistics display at end'
    )
    parser.add_argument(
        '--account',
        type=str,
        default='cultpass',
        help='Account ID for RAG system (default: cultpass)'
    )
    parser.add_argument(
        '--memory',
        action='store_true',
        help='Enable long-term memory features (database persistence)'
    )
    parser.add_argument(
        '--db-path',
        type=str,
        default='data/core/udahub.db',
        help='Path to database for long-term memory (default: data/core/udahub.db)'
    )
    
    args = parser.parse_args()
    
    try:
        # Load environment variables
        load_dotenv()
        
        # Print banner
        print_banner()
        
        # Initialize logger
        logger = get_logger(log_dir="logs", console_output=args.show_logs)
        print("✅ Logger initialized")
        
        # Generate or use provided session ID
        session_id = args.session_id or generate_session_id()
        print(f"📋 Session ID: {session_id}")
        
        # Initialize orchestrator
        print(f"🔄 Initializing orchestrator...")
        print(f"   Account: {args.account}")
        print(f"   Long-term memory: {'enabled' if args.memory else 'disabled'}")
        
        orchestrator = create_orchestrator(
            account_id=args.account,
            enable_long_term_memory=args.memory,
            db_path=args.db_path
        )
        
        print("✅ Orchestrator ready")
        print(f"📊 Agents: Classifier, Resolver, Tool Agent, Escalation")
        print(f"🔄 Workflow: Supervisor-based routing with logging")
        
        # Start interactive chat
        print("\n" + "=" * 70)
        print("  Starting Interactive Chat")
        print("  Type '/help' for commands or 'quit' to exit")
        print("=" * 70)
        
        chat_interface(
            agent=orchestrator,
            ticket_id=session_id,
            show_logs=args.show_logs,
            show_stats=not args.no_stats
        )
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
