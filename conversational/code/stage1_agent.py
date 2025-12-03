"""
Stage 1 Agent: Dataset Summarization

This agent profiles all available datasets and generates comprehensive summaries
that describe the data characteristics, quality, and potential for analysis.
"""

import json
from typing import Dict, Any, Optional, Annotated
from datetime import datetime
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from code.config import (
    DATA_DIR, SUMMARIES_DIR, SECONDARY_LLM_CONFIG,
    STAGE_MAX_ROUNDS, DataPassingManager, logger
)
from code.utils import profile_csv, list_data_files
from code.models import DatasetSummary, Stage1Output, PipelineState
from tools.stage1_tools import STAGE1_TOOLS


# ============================================================================
# STATE DEFINITION
# ============================================================================

class Stage1State(BaseModel):
    """State for Stage 1 agent."""
    messages: Annotated[list, add_messages] = []
    datasets_to_process: list = []
    datasets_processed: list = []
    summaries: list = []
    errors: list = []
    iteration: int = 0
    complete: bool = False


# ============================================================================
# SYSTEM PROMPT
# ============================================================================

STAGE1_SYSTEM_PROMPT = """You are a Data Profiling Agent responsible for analyzing datasets and generating comprehensive summaries.

## Your Role
You analyze CSV/Parquet/TSV files to understand their structure, data types, quality, and potential for analytical tasks like forecasting.

## Your Goals
1. Profile each dataset to understand its columns, types, and characteristics
2. Identify datetime columns (critical for time series/forecasting)
3. Identify numeric columns that could be prediction targets
4. Assess data quality (nulls, duplicates, data types)
5. Save comprehensive summaries for downstream stages

## Available Tools
- list_available_datasets: List all data files available
- profile_dataset: Profile a single dataset and get detailed statistics
- save_dataset_summary: Save a dataset summary to disk
- get_existing_summaries: Check what summaries already exist
- analyze_dataset_for_forecasting: Specifically analyze forecasting potential

## Workflow
1. First, list available datasets
2. Check existing summaries to avoid duplicate work
3. For each unprocessed dataset:
   a. Profile the dataset
   b. Analyze its forecasting potential
   c. Save the summary
4. Report completion when all datasets are processed

## Output Format
When profiling is complete, provide a summary of:
- Datasets processed
- Key findings (datetime columns, potential targets)
- Data quality assessment
- Recommendations for next steps

Be thorough but efficient. Focus on information that will help downstream stages make decisions about what analyses are possible.
"""


# ============================================================================
# AGENT LOGIC
# ============================================================================

def create_stage1_agent():
    """Create the Stage 1 agent graph."""

    # Initialize LLM with tools
    llm = ChatOpenAI(**SECONDARY_LLM_CONFIG)
    llm_with_tools = llm.bind_tools(STAGE1_TOOLS, parallel_tool_calls=False)

    def agent_node(state: Stage1State) -> Dict[str, Any]:
        """Main agent reasoning node."""
        messages = state.messages

        # Add system message if not present
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=STAGE1_SYSTEM_PROMPT)] + list(messages)

        # Check iteration limit
        if state.iteration >= STAGE_MAX_ROUNDS.get("stage1", 10):
            return {
                "messages": [AIMessage(content="Maximum iterations reached. Completing Stage 1.")],
                "complete": True
            }

        # Get LLM response
        response = llm_with_tools.invoke(messages)

        return {
            "messages": [response],
            "iteration": state.iteration + 1
        }

    def should_continue(state: Stage1State) -> str:
        """Determine if we should continue or end."""
        if state.complete:
            return "end"

        # Check last message for tool calls
        if state.messages:
            last_message = state.messages[-1]
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "tools"

        return "end"

    # Build graph
    builder = StateGraph(Stage1State)

    # Add nodes
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(STAGE1_TOOLS))

    # Set entry point
    builder.set_entry_point("agent")

    # Add edges
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    builder.add_edge("tools", "agent")

    # Compile with checkpointer
    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


def run_stage1(pipeline_state: PipelineState = None) -> Stage1Output:
    """
    Run Stage 1: Dataset Summarization.

    This can be run in two ways:
    1. Autonomous profiling of all datasets
    2. Quick mode for conversational context

    Returns:
        Stage1Output with all dataset summaries
    """
    logger.info("Starting Stage 1: Dataset Summarization")

    # Create agent
    graph = create_stage1_agent()

    # Initial message
    initial_message = HumanMessage(content="""
Please analyze all available datasets:

1. First, list the available datasets
2. Check if any summaries already exist
3. Profile each new/unprocessed dataset
4. Analyze each dataset's forecasting potential
5. Save summaries for all datasets

When complete, provide a summary of findings.
""")

    # Run agent
    config = {"configurable": {"thread_id": "stage1_main"}}
    initial_state = Stage1State(messages=[initial_message])

    try:
        final_state = graph.invoke(initial_state, config)

        # Collect summaries from disk
        summaries = []
        summary_files = list(SUMMARIES_DIR.glob("*.summary.json"))

        for sf in summary_files:
            try:
                data = DataPassingManager.load_artifact(sf)
                summary_data = data.get('data', data) if isinstance(data, dict) else data
                summaries.append(DatasetSummary(**summary_data))
            except Exception as e:
                logger.warning(f"Could not load summary {sf}: {e}")

        output = Stage1Output(
            summaries=summaries,
            total_files_processed=len(summaries),
            errors=[]
        )

        logger.info(f"Stage 1 complete: {len(summaries)} datasets summarized")
        return output

    except Exception as e:
        logger.error(f"Stage 1 failed: {e}")
        return Stage1Output(
            summaries=[],
            total_files_processed=0,
            errors=[str(e)]
        )


def run_stage1_quick() -> Dict[str, Any]:
    """
    Quick Stage 1 run for conversational mode.

    Directly profiles datasets without full agent loop.
    Returns summary dict for immediate use.
    """
    logger.info("Running Stage 1 (quick mode)")

    results = {
        "datasets": [],
        "datetime_columns_found": [],
        "target_candidates": [],
        "errors": []
    }

    # Get list of data files
    files = list_data_files(DATA_DIR)

    for filename in files:
        try:
            filepath = DATA_DIR / filename

            # Profile the dataset
            summary = profile_csv(filepath)

            # Save summary
            summary_dict = summary.model_dump()
            output_name = f"{filepath.stem}.summary.json"

            DataPassingManager.save_artifact(
                data=summary_dict,
                output_dir=SUMMARIES_DIR,
                filename=output_name,
                metadata={"stage": "stage1", "type": "dataset_summary"}
            )

            results["datasets"].append({
                "filename": filename,
                "rows": summary.n_rows,
                "cols": summary.n_cols,
                "quality_score": summary.data_quality_score
            })

            if summary.datetime_columns:
                results["datetime_columns_found"].extend([
                    {"dataset": filename, "column": c} for c in summary.datetime_columns
                ])

            if summary.has_target_candidates:
                results["target_candidates"].extend([
                    {"dataset": filename, "column": c} for c in summary.has_target_candidates
                ])

        except Exception as e:
            results["errors"].append({"dataset": filename, "error": str(e)})
            logger.error(f"Error profiling {filename}: {e}")

    logger.info(f"Stage 1 quick mode complete: {len(results['datasets'])} datasets")
    return results


# ============================================================================
# PIPELINE NODE FUNCTION
# ============================================================================

def stage1_node(state: PipelineState) -> PipelineState:
    """
    Stage 1 node for the master pipeline graph.

    Args:
        state: Current pipeline state

    Returns:
        Updated pipeline state with Stage 1 output
    """
    state.mark_stage_started("stage1")

    try:
        output = run_stage1(state)
        state.stage1_output = output
        state.mark_stage_completed("stage1", output)
    except Exception as e:
        state.mark_stage_failed("stage1", str(e))

    return state


if __name__ == "__main__":
    # Test run
    result = run_stage1_quick()
    print(json.dumps(result, indent=2, default=str))
