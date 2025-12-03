#!/usr/bin/env python3
"""
Main entry point for the unified agentic AI pipeline.

Run individual stages or the complete pipeline from the command line.
"""

import argparse
import sys
from pathlib import Path

# Add the parent directory to path so we can import agentic_code
sys.path.insert(0, str(Path(__file__).parent.parent))

from agentic_code.master_agent import run_full_pipeline, run_partial_pipeline
from agentic_code.stage1_agent import run_stage1
from agentic_code.stage2_agent import run_stage2
from agentic_code.stage3_agent import run_stage3
from agentic_code.stage4_agent import run_stage4
from agentic_code.stage5_agent import run_stage5
from agentic_code.config import print_config


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Unified Agentic AI Pipeline - Multi-stage data analytics system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run the full pipeline
  python run_pipeline.py

  # Run full pipeline with specific task
  python run_pipeline.py --task TSK-001

  # Run only Stage 1 (dataset summarization)
  python run_pipeline.py --stage 1

  # Run only Stage 2 (task proposal)
  python run_pipeline.py --stage 2

  # Run Stage 3 for a specific task
  python run_pipeline.py --stage 3 --task TSK-001

  # Run stages 1-3
  python run_pipeline.py --start 1 --end 3

  # Show configuration
  python run_pipeline.py --config
        """
    )
    
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Run a single stage (1-5)"
    )
    
    parser.add_argument(
        "--start",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=1,
        help="Start stage (for partial pipeline run)"
    )
    
    parser.add_argument(
        "--end",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=5,
        help="End stage (for partial pipeline run)"
    )
    
    parser.add_argument(
        "--task",
        type=str,
        help="Task ID to execute (e.g., TSK-001). Required for stages 3-5."
    )
    
    parser.add_argument(
        "--config",
        action="store_true",
        help="Print configuration and exit"
    )
    
    parser.add_argument(
        "--conversational",
        action="store_true",
        help="Run in conversational mode (interactive)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        default=True,
        help="Enable debug output (default: True)"
    )
    
    args = parser.parse_args()
    
    # Conversational mode
    if args.conversational:
        from agentic_code.run_conversational import main as run_conv
        # We need to hack sys.argv or call the function directly
        # Since run_conversational.main() parses args too, let's just call it
        # But we need to clear args first or pass them explicitly
        # Simpler: just import and run the interactive loop logic if possible, 
        # or subprocess. But importing is better.
        # Let's just call the main function from run_conversational
        # We might need to adjust sys.argv to avoid conflict if we passed other flags
        sys.argv = [sys.argv[0]] # Clear args for the next parser
        return run_conv()
    
    # Print config and exit if requested
    if args.config:
        print_config()
        return 0
    
    # Run single stage
    if args.stage:
        print(f"\n🎯 Running Stage {args.stage}")
        
        if args.stage == 1:
            run_stage1()
        elif args.stage == 2:
            run_stage2()
        elif args.stage == 3:
            if not args.task:
                print("ERROR: --task is required for Stage 3")
                return 1
            run_stage3(args.task, debug=args.debug)
        elif args.stage == 4:
            if not args.task:
                # Expect task in format PLAN-TSK-XXX
                print("ERROR: --task is required for Stage 4 (use plan ID like PLAN-TSK-001)")
                return 1
            # Stage 4 expects plan ID, so prepend PLAN- if not present
            plan_id = args.task if args.task.startswith("PLAN-") else f"PLAN-{args.task}"
            run_stage4(plan_id, debug=args.debug)
        elif args.stage == 5:
            if not args.task:
                print("ERROR: --task is required for Stage 5 (use plan ID like PLAN-TSK-001)")
                return 1
            plan_id = args.task if args.task.startswith("PLAN-") else f"PLAN-{args.task}"
            run_stage5(plan_id, debug=args.debug)
        
        return 0
    
    # Run partial or full pipeline
    if args.start != 1 or args.end != 5:
        # Partial pipeline
        print(f"\n🎯 Running pipeline stages {args.start}-{args.end}")
        if args.end >= 3 and not args.task:
            print("ERROR: --task is required when running stages 3 or higher")
            return 1
        run_partial_pipeline(args.start, args.end, args.task)
    else:
        # Full pipeline
        print("\n🎯 Running full pipeline (Stages 1-5)")
        if args.task:
            print(f"   Task ID: {args.task}")
        run_full_pipeline(selected_task_id=args.task)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
