#!/usr/bin/env python3
"""
Conversational Interface Entry Point

Run the agentic pipeline in conversational mode.
"""

import argparse
import sys
import uuid
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agentic_code.stage0_agent import run_conversational_turn
from agentic_code.config import print_config


def main():
    """Main entry point for conversational mode."""
    parser = argparse.ArgumentParser(
        description="Conversational Agentic AI - Talk to your data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start interactive session
  python run_conversational.py

  # Ask a single question
  python run_conversational.py --query "What can you predict?"
        """
    )
    
    parser.add_argument(
        "--query",
        type=str,
        help="Single query to execute (non-interactive mode)"
    )
    
    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="Session ID for conversation memory (default: auto-generated)"
    )
    
    args = parser.parse_args()
    
    # Generate session ID if not provided
    session_id = args.session_id or str(uuid.uuid4())[:8]
    
    print("\n" + "=" * 60)
    print("🤖 CONVERSATIONAL DATA AGENT")
    print("=" * 60)
    print(f"Session ID: {session_id}")
    
    # Single query mode
    if args.query:
        print(f"\nUser: {args.query}")
        print("\nThinking...", end="", flush=True)
        response = run_conversational_turn(args.query, thread_id=session_id)
        print(f"\rAgent: {response}\n")
        return 0
        
    # Interactive mode
    print("\nType your questions below. Type 'exit' or 'quit' to end.")
    print("-" * 60)
    
    while True:
        try:
            query = input("\nUser: ").strip()
            
            if not query:
                continue
                
            if query.lower() in ["exit", "quit", "bye"]:
                print("\nGoodbye! 👋")
                break
            
            print("Thinking...", end="", flush=True)
            response = run_conversational_turn(query, thread_id=session_id)
            # Clear "Thinking..." line and print response
            print(f"\rAgent: {response}")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            
    return 0


if __name__ == "__main__":
    sys.exit(main())
