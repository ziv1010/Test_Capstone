#!/usr/bin/env python3
"""
Conversational AI Pipeline - Main Entry Point

This script provides the main interface for the conversational AI pipeline.
It supports multiple modes of operation:

1. Conversational Mode (default): Interactive chat interface
2. Analyze Mode: Quick data analysis
3. Propose Mode: Generate task proposals
4. Run Mode: Execute pipeline for a specific task
5. Full Mode: Run the complete pipeline

Usage:
    # Interactive conversation
    python run_conversational.py

    # Analyze available data
    python run_conversational.py --mode analyze

    # Propose tasks
    python run_conversational.py --mode propose
    python run_conversational.py --mode propose --query "Can I forecast sales?"

    # Run a specific task
    python run_conversational.py --mode run --task TSK-001

    # Run full pipeline
    python run_conversational.py --mode full --task TSK-001

    # Run specific stages
    python run_conversational.py --mode run --task TSK-001 --stages "stage3,stage3b,stage4"

    # Show pipeline status
    python run_conversational.py --status

    # Show configuration
    python run_conversational.py --config
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from code.config import (
    PROJECT_ROOT, DATA_DIR, OUTPUT_ROOT,
    PRIMARY_LLM_CONFIG, SECONDARY_LLM_CONFIG,
    STAGE_MAX_ROUNDS, logger
)
from code.master_orchestrator import (
    ConversationalOrchestrator,
    quick_analyze_data,
    quick_propose_tasks,
    quick_run_task,
    run_full_pipeline,
    run_pipeline_stages,
    load_cached_state
)


def print_banner():
    """Print welcome banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║           CONVERSATIONAL AI FORECASTING PIPELINE                  ║
║                                                                    ║
║   A dataset-agnostic pipeline for time series forecasting          ║
║   with conversational interface and automated method selection     ║
╚══════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def print_config():
    """Print current configuration."""
    print("\n=== Configuration ===\n")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Output Directory: {OUTPUT_ROOT}")
    print(f"\nLLM Configuration:")
    print(f"  Primary Model: {PRIMARY_LLM_CONFIG.get('model')}")
    print(f"  Secondary Model: {SECONDARY_LLM_CONFIG.get('model')}")
    print(f"  Base URL: {PRIMARY_LLM_CONFIG.get('base_url')}")
    print(f"\nStage Max Rounds:")
    for stage, rounds in STAGE_MAX_ROUNDS.items():
        print(f"  {stage}: {rounds}")


def print_status():
    """Print current pipeline status."""
    from tools.conversation_tools import check_pipeline_status
    print("\n=== Pipeline Status ===\n")
    print(check_pipeline_status.invoke({}))


def run_conversation_mode():
    """Run interactive conversation mode."""
    print_banner()

    print("""
How to use this pipeline:

1. First, I'll analyze your data (Stage 1)
2. Then propose analytical tasks (Stage 2)
3. You can ask questions about the data or tasks
4. Request forecasting by saying "run task TSK-XXX"
5. View results and visualizations

Commands:
  'status' - Check pipeline status
  'summary' - Get data and task summary
  'help' - Show this help
  'quit' - Exit

Ask me anything about your data!
""")

    orchestrator = ConversationalOrchestrator()

    # Check if we have any data analyzed
    state, start_stage = load_cached_state()
    if start_stage == "stage1":
        print("\nNo data has been analyzed yet. Analyzing available datasets...")
        result = quick_analyze_data()
        print(f"Analyzed {len(result.get('datasets', []))} datasets.")

        # If we found data, propose tasks
        if result.get('datasets'):
            print("\nGenerating task proposals...")
            proposals = quick_propose_tasks()
            if proposals.get('proposals'):
                print(f"Generated {len(proposals['proposals'])} task proposals.")
                print("\nAvailable tasks:")
                for p in proposals['proposals']:
                    print(f"  - {p['id']}: {p['title']}")

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            # Handle special commands
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nGoodbye! Your session has been saved.")
                orchestrator.conversation.save_session()
                break

            if user_input.lower() == 'help':
                print("""
Commands:
  'status' - Check what stages have been completed
  'summary' - Get summary of data and available tasks
  'analyze' - Analyze the data (Stage 1 & 2)
  'run task TSK-XXX' - Execute forecasting pipeline for a task
  'show results' - View execution results
  'show plots' - View visualizations
  'quit' - Exit the pipeline

You can also ask natural language questions like:
  "What data is available?"
  "Can I predict sales?"
  "What tasks can I do?"
""")
                continue

            if user_input.lower() == 'status':
                print(f"\n{orchestrator.get_status()}")
                continue

            if user_input.lower() == 'summary':
                print(f"\n{orchestrator.get_summary()}")
                continue

            if user_input.lower() == 'analyze':
                print("\nAnalyzing data...")
                result = quick_analyze_data()
                print(f"Analyzed {len(result.get('datasets', []))} datasets.")
                proposals = quick_propose_tasks()
                print(f"Generated {len(proposals.get('proposals', []))} task proposals.")
                continue

            # Process through conversation agent
            result = orchestrator.process_user_input(user_input)
            print(f"\nAssistant: {result['response']}")

        except KeyboardInterrupt:
            print("\n\nSession interrupted. Saving...")
            orchestrator.conversation.save_session()
            break
        except Exception as e:
            print(f"\nError: {e}")
            logger.error(f"Conversation error: {e}")
            continue


def run_analyze_mode():
    """Run quick analysis mode."""
    print("\n=== Data Analysis Mode ===\n")
    print("Analyzing available datasets...")

    result = quick_analyze_data()

    print(f"\nAnalyzed {len(result.get('datasets', []))} datasets:\n")

    for ds in result.get('datasets', []):
        print(f"  {ds['filename']}:")
        print(f"    Rows: {ds['rows']}, Columns: {ds['cols']}")
        print(f"    Quality: {ds['quality_score']:.1%}")

    if result.get('datetime_columns_found'):
        print("\nDatetime columns (suitable for forecasting):")
        for dt in result['datetime_columns_found']:
            print(f"  - {dt['dataset']}: {dt['column']}")

    if result.get('target_candidates'):
        print("\nPotential target columns:")
        for tc in result['target_candidates'][:10]:
            print(f"  - {tc['dataset']}: {tc['column']}")

    if result.get('errors'):
        print("\nErrors:")
        for err in result['errors']:
            print(f"  - {err['dataset']}: {err['error']}")


def run_propose_mode(query: str = None):
    """Run task proposal mode."""
    print("\n=== Task Proposal Mode ===\n")

    if query:
        print(f"Generating proposals for: {query}\n")
    else:
        print("Generating general task proposals...\n")

    result = quick_propose_tasks(query)

    if not result.get('proposals'):
        print("No proposals generated. Make sure data has been analyzed first.")
        return

    print(f"Generated {len(result['proposals'])} proposals:\n")

    for p in result['proposals']:
        print(f"  {p['id']}: {p['title']}")
        print(f"    Category: {p['category']}")
        print(f"    Target: {p['target_column']}")
        print(f"    Feasibility: {p.get('feasibility_score', 'N/A')}")
        print(f"    Description: {p['problem_statement'][:100]}...")
        print()


def run_execute_mode(task_id: str, stages: str = None):
    """Run pipeline execution mode."""
    print(f"\n=== Pipeline Execution Mode ===\n")
    print(f"Task: {task_id}\n")

    if stages:
        stage_list = [s.strip() for s in stages.split(",")]
        print(f"Running stages: {stage_list}\n")

        state = run_pipeline_stages(stage_list, task_id)
    else:
        print("Running forecasting pipeline: 3 → 3B → 3.5A → 3.5B → 4 → 5\n")
        state = quick_run_task(task_id)

    print("\n=== Execution Summary ===\n")

    if isinstance(state, dict):
        print(f"Completed stages: {state.get('completed_stages', [])}")
        print(f"Failed stages: {state.get('failed_stages', [])}")
        if state.get('errors'):
            print(f"Errors: {state['errors']}")
        print(f"\nSuccess: {state.get('success', False)}")
    else:
        # PipelineState object
        from code.models import StageStatus
        completed = [s for s, st in state.stages.items() if st.status == StageStatus.COMPLETED]
        failed = [s for s, st in state.stages.items() if st.status == StageStatus.FAILED]

        print(f"Completed stages: {completed}")
        print(f"Failed stages: {failed}")
        if state.errors:
            print(f"Errors: {state.errors}")

        # Show metrics if available
        if state.stage4_output and state.stage4_output.metrics:
            print(f"\nMetrics: {state.stage4_output.metrics}")


def run_full_mode(task_id: str):
    """Run full pipeline mode."""
    print(f"\n=== Full Pipeline Mode ===\n")
    print(f"Running complete pipeline for task: {task_id}\n")

    state = run_full_pipeline(task_id)

    print("\n=== Pipeline Complete ===\n")

    # Show summary
    from code.models import StageStatus
    for stage_name, stage_state in state.stages.items():
        status_icon = "✓" if stage_state.status == StageStatus.COMPLETED else "✗"
        print(f"  [{status_icon}] {stage_name}: {stage_state.status.value}")

    if state.errors:
        print(f"\nErrors encountered: {state.errors}")

    # Show final metrics
    if state.stage4_output:
        print(f"\nExecution Status: {state.stage4_output.status.value}")
        if state.stage4_output.metrics:
            print(f"Metrics: {json.dumps(state.stage4_output.metrics, indent=2)}")

    if state.stage5_output:
        print(f"\nVisualizations: {len(state.stage5_output.visualizations)} plots created")
        if state.stage5_output.insights:
            print(f"Insights: {len(state.stage5_output.insights)} insights generated")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Conversational AI Forecasting Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Interactive mode
  %(prog)s --mode analyze            # Analyze data
  %(prog)s --mode propose            # Generate task proposals
  %(prog)s --mode run --task TSK-001 # Run forecasting pipeline
  %(prog)s --status                  # Show pipeline status
  %(prog)s --config                  # Show configuration
"""
    )

    parser.add_argument(
        "--mode",
        choices=["conversation", "analyze", "propose", "run", "full"],
        default="conversation",
        help="Execution mode (default: conversation)"
    )
    parser.add_argument(
        "--task",
        type=str,
        help="Task ID for run/full mode (e.g., TSK-001)"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="User query for propose mode"
    )
    parser.add_argument(
        "--stages",
        type=str,
        help="Comma-separated stages to run (e.g., 'stage3,stage3b,stage4')"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show pipeline status and exit"
    )
    parser.add_argument(
        "--config",
        action="store_true",
        help="Show configuration and exit"
    )

    args = parser.parse_args()

    # Handle special flags
    if args.config:
        print_config()
        return

    if args.status:
        print_status()
        return

    # Run appropriate mode
    if args.mode == "conversation":
        run_conversation_mode()

    elif args.mode == "analyze":
        run_analyze_mode()

    elif args.mode == "propose":
        run_propose_mode(args.query)

    elif args.mode == "run":
        if not args.task:
            print("Error: --task is required for run mode")
            print("Usage: python run_conversational.py --mode run --task TSK-001")
            sys.exit(1)
        run_execute_mode(args.task, args.stages)

    elif args.mode == "full":
        if not args.task:
            print("Error: --task is required for full mode")
            print("Usage: python run_conversational.py --mode full --task TSK-001")
            sys.exit(1)
        run_full_mode(args.task)


if __name__ == "__main__":
    main()
