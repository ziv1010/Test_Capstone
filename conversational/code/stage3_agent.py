"""
Stage 3 Agent: Execution Planning

This agent creates detailed execution plans for selected tasks,
specifying exactly how to load, transform, and process the data.
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
    STAGE2_OUT_DIR, STAGE3_OUT_DIR, SECONDARY_LLM_CONFIG,
    STAGE_MAX_ROUNDS, MIN_NON_NULL_FRACTION, DataPassingManager, logger
)
from code.models import Stage3Plan, PipelineState
from tools.stage3_tools import STAGE3_TOOLS


# ============================================================================
# STATE DEFINITION
# ============================================================================

class Stage3State(BaseModel):
    """State for Stage 3 agent."""
    messages: Annotated[list, add_messages] = []
    task_id: str = ""
    task_loaded: bool = False
    data_inspected: list = []
    plan_created: bool = False
    iteration: int = 0
    complete: bool = False


# ============================================================================
# SYSTEM PROMPT
# ============================================================================

STAGE3_SYSTEM_PROMPT = """You are an Execution Planning Agent responsible for creating detailed execution plans for analytical tasks.

## Your Role
Given a task proposal, you create a comprehensive execution plan that specifies exactly:
- Which files to load and how
- What columns to use
- How to join datasets (if multiple)
- What features to engineer
- How to validate the data
- What the expected outputs are

## Your Goals
1. Load and understand the selected task proposal
2. Inspect all required data files
3. Validate that columns meet quality requirements (≥65% non-null)
4. Design join strategy if multiple datasets involved
5. Specify feature engineering steps
6. Define validation/test split strategy
7. Save a comprehensive execution plan

## Available Tools
- load_task_proposal: Load a specific task by ID
- list_all_proposals: List available tasks
- list_data_files_stage3: List available data files
- inspect_data_file_stage3: Inspect a file's structure
- validate_columns_for_task: Check column quality
- analyze_join_feasibility: Analyze if joins will work
- python_sandbox_stage3: Execute Python for analysis
- save_stage3_plan: Save the execution plan
- get_execution_plan_template: Get the plan structure

## Plan Requirements
Your plan MUST include:
- plan_id: "PLAN-{task_id}" format
- selected_task_id: The task ID
- goal: Clear description of objective
- task_category: forecasting/regression/classification/etc
- file_instructions: How to load each file
  - filename, filepath
  - columns_to_use (only columns needed)
  - filters (any row filters)
  - parse_dates (datetime columns)
- join_steps: If multiple files, how to join them
- feature_engineering: Features to create
  - name, description, source_columns
  - implementation_code (actual Python code)
- target_column: What to predict
- date_column: For temporal tasks
- validation_strategy: temporal or random
- expected_model_types: What models to try
- evaluation_metrics: How to measure success

## Quality Validation Rules
- All columns must have ≥65% non-null values
- If a column fails, either:
  - Specify how to handle missing values
  - Exclude the column
- Document any data quality issues found

## Implementation Code Guidelines
Keep implementation_code CONCISE:
- Use simple pandas operations
- Single line when possible: "df['lag_1'] = df['target'].shift(1)"
- No comments in the code
- No error handling (that's for execution stage)

## Workflow
1. Load the task proposal
2. Inspect each required data file
3. Validate column quality
4. If joins needed, analyze join feasibility
5. Design feature engineering (especially lag features for forecasting)
6. Create and save the execution plan

IMPORTANT: The plan must be complete and actionable. Downstream stages will execute exactly what you specify.
"""


# ============================================================================
# AGENT LOGIC
# ============================================================================

def create_stage3_agent():
    """Create the Stage 3 agent graph."""

    llm = ChatOpenAI(**SECONDARY_LLM_CONFIG)
    llm_with_tools = llm.bind_tools(STAGE3_TOOLS, parallel_tool_calls=False)

    def agent_node(state: Stage3State) -> Dict[str, Any]:
        """Main agent reasoning node."""
        messages = state.messages

        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=STAGE3_SYSTEM_PROMPT)] + list(messages)

        if state.iteration >= STAGE_MAX_ROUNDS.get("stage3", 30):
            return {
                "messages": [AIMessage(content="Maximum iterations reached. Saving current plan.")],
                "complete": True
            }

        response = llm_with_tools.invoke(messages)

        return {
            "messages": [response],
            "iteration": state.iteration + 1
        }

    def should_continue(state: Stage3State) -> str:
        """Determine if we should continue or end."""
        if state.complete:
            return "end"

        if state.messages:
            last_message = state.messages[-1]
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "tools"

        return "end"

    builder = StateGraph(Stage3State)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(STAGE3_TOOLS))
    builder.set_entry_point("agent")
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END}
    )
    builder.add_edge("tools", "agent")

    return builder.compile(checkpointer=MemorySaver())


def run_stage3(task_id: str, pipeline_state: PipelineState = None) -> Stage3Plan:
    """
    Run Stage 3: Execution Planning.

    Creates a detailed plan for executing the specified task.
    """
    logger.info(f"Starting Stage 3: Execution Planning for {task_id}")

    graph = create_stage3_agent()

    initial_message = HumanMessage(content=f"""
Create an execution plan for task: {task_id}

Follow these steps:
1. Load the task proposal for {task_id}
2. List and inspect all required data files
3. Validate column quality (≥65% non-null required)
4. If joins needed, analyze feasibility
5. Design feature engineering (especially for forecasting: lags, rolling means, etc.)
6. Create comprehensive execution plan
7. Save the plan using save_stage3_plan

The plan ID should be: PLAN-{task_id}

Be thorough - downstream stages depend on this plan being complete and accurate.
""")

    config = {"configurable": {"thread_id": f"stage3_{task_id}"}}
    initial_state = Stage3State(messages=[initial_message], task_id=task_id)

    try:
        final_state = graph.invoke(initial_state, config)

        # Load plan from disk
        plan_path = STAGE3_OUT_DIR / f"PLAN-{task_id}.json"
        if plan_path.exists():
            data = DataPassingManager.load_artifact(plan_path)
            plan = Stage3Plan(**data)
            logger.info(f"Stage 3 complete: Plan saved to {plan_path}")
            return plan
        else:
            logger.error("Plan not saved to disk")
            raise RuntimeError("Execution plan was not saved")

    except Exception as e:
        logger.error(f"Stage 3 failed: {e}")
        raise


# ============================================================================
# PIPELINE NODE FUNCTION
# ============================================================================

def stage3_node(state: PipelineState) -> PipelineState:
    """
    Stage 3 node for the master pipeline graph.
    """
    state.mark_stage_started("stage3")

    task_id = state.selected_task_id
    if not task_id:
        state.mark_stage_failed("stage3", "No task ID specified")
        return state

    try:
        output = run_stage3(task_id, state)
        state.stage3_output = output
        state.mark_stage_completed("stage3", output)
    except Exception as e:
        state.mark_stage_failed("stage3", str(e))

    return state


if __name__ == "__main__":
    import sys
    task_id = sys.argv[1] if len(sys.argv) > 1 else "TSK-001"
    plan = run_stage3(task_id)
    print(f"Created plan: {plan.plan_id}")
    print(f"Goal: {plan.goal}")
