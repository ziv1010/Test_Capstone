"""
Stage 3B Agent: Data Preparation

This agent prepares data according to the execution plan,
handling joins, feature engineering, and missing value treatment.
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
    STAGE3_OUT_DIR, STAGE3B_OUT_DIR, SECONDARY_LLM_CONFIG,
    STAGE_MAX_ROUNDS, DataPassingManager, logger
)
from code.models import PreparedDataOutput, PipelineState
from tools.stage3b_tools import STAGE3B_TOOLS, reset_react_state


# ============================================================================
# STATE DEFINITION
# ============================================================================

class Stage3BState(BaseModel):
    """State for Stage 3B agent."""
    messages: Annotated[list, add_messages] = []
    plan_id: str = ""
    plan_loaded: bool = False
    data_prepared: bool = False
    iteration: int = 0
    complete: bool = False


# ============================================================================
# SYSTEM PROMPT
# ============================================================================

STAGE3B_SYSTEM_PROMPT = """You are a Data Preparation Agent responsible for transforming raw data into model-ready format.

## Your Role
Execute the Stage 3 execution plan to:
1. Load required data files
2. Apply specified filters
3. Perform joins if multiple datasets
4. Engineer features as specified
5. Handle ALL missing values
6. Save the prepared dataset

## CRITICAL REQUIREMENT: NO NULLS
The prepared data MUST have ZERO null values. Before saving:
- Check for nulls in every column
- Impute numeric columns (median or mean)
- Fill categorical columns with 'UNKNOWN' or mode
- Drop rows only as last resort
- VERIFY: df.isnull().sum().sum() == 0

## ReAct Framework (MANDATORY)
For EVERY action, you MUST:
1. THINK: Call record_thought() with your reasoning
2. ACT: Execute the action using appropriate tool
3. OBSERVE: Call record_observation() with what you learned

This prevents loops and ensures transparent reasoning.

## Available Tools
- load_execution_plan: Load the Stage 3 plan
- record_thought: Document your reasoning before acting
- record_observation: Document learnings after acting
- run_data_prep_code: Execute data preparation Python code
- check_data_quality: Check data for nulls and issues
- save_prepared_data: Save the final prepared dataset
- verify_prepared_data: Verify the saved data is correct
- get_react_summary: Review your reasoning trail

## Workflow
1. Load the execution plan
2. For each file in file_instructions:
   - THINK: What am I loading and why?
   - ACT: Load the file with specified columns
   - OBSERVE: What shape/quality is the data?
3. If joins needed:
   - THINK: What's the join strategy?
   - ACT: Perform joins in order specified
   - OBSERVE: Did rows match? Any data loss?
4. For each feature in feature_engineering:
   - THINK: What feature am I creating?
   - ACT: Execute the implementation_code
   - OBSERVE: Did it work? Any nulls created?
5. Handle missing values:
   - THINK: What's my imputation strategy?
   - ACT: Apply imputation to each column
   - OBSERVE: Verify zero nulls
6. Save the prepared data

## Python Code Guidelines
When using run_data_prep_code:
```python
import pandas as pd
import numpy as np
from pathlib import Path

# Load data
DATA_DIR = Path('/path/to/data')
df = pd.read_csv(DATA_DIR / 'file.csv')

# Process...

# ALWAYS print results
print(f"Shape: {df.shape}")
print(f"Nulls: {df.isnull().sum().sum()}")
```

## Error Recovery
If something fails:
1. Record the observation with what went wrong
2. Think about an alternative approach
3. Try again with the fix
4. If stuck after 3 attempts, document the issue and move on

REMEMBER: The final dataset must have ZERO nulls. This is non-negotiable.
"""


# ============================================================================
# AGENT LOGIC
# ============================================================================

def create_stage3b_agent():
    """Create the Stage 3B agent graph."""

    llm = ChatOpenAI(**SECONDARY_LLM_CONFIG)
    llm_with_tools = llm.bind_tools(STAGE3B_TOOLS, parallel_tool_calls=False)

    def agent_node(state: Stage3BState) -> Dict[str, Any]:
        """Main agent reasoning node."""
        messages = state.messages

        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=STAGE3B_SYSTEM_PROMPT)] + list(messages)

        if state.iteration >= STAGE_MAX_ROUNDS.get("stage3b", 100):
            return {
                "messages": [AIMessage(content="Maximum iterations reached. Completing with current state.")],
                "complete": True
            }

        response = llm_with_tools.invoke(messages)

        return {
            "messages": [response],
            "iteration": state.iteration + 1
        }

    def should_continue(state: Stage3BState) -> str:
        """Determine if we should continue or end."""
        if state.complete:
            return "end"

        if state.messages:
            last_message = state.messages[-1]
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "tools"

        return "end"

    builder = StateGraph(Stage3BState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(STAGE3B_TOOLS))
    builder.set_entry_point("agent")
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END}
    )
    builder.add_edge("tools", "agent")

    return builder.compile(checkpointer=MemorySaver())


def run_stage3b(plan_id: str, pipeline_state: PipelineState = None) -> PreparedDataOutput:
    """
    Run Stage 3B: Data Preparation.

    Transforms raw data into model-ready format according to plan.
    """
    logger.info(f"Starting Stage 3B: Data Preparation for {plan_id}")

    # Reset ReAct state for clean tracking
    reset_react_state()

    graph = create_stage3b_agent()

    initial_message = HumanMessage(content=f"""
Prepare data according to execution plan: {plan_id}

Follow the ReAct framework strictly:

1. First, load the execution plan for {plan_id}
2. THINK about your approach, then execute step by step
3. Load each file as specified in file_instructions
4. Perform any joins as specified
5. Create all features in feature_engineering
6. Handle ALL missing values - NO NULLS ALLOWED
7. Verify zero nulls before saving
8. Save the prepared dataset

Use record_thought() BEFORE each major action.
Use record_observation() AFTER each action.

The output should be saved as: prepared_{plan_id}.parquet
""")

    config = {"configurable": {"thread_id": f"stage3b_{plan_id}"}}
    initial_state = Stage3BState(messages=[initial_message], plan_id=plan_id)

    try:
        final_state = graph.invoke(initial_state, config)

        # Load metadata from disk
        meta_path = STAGE3B_OUT_DIR / f"prepared_{plan_id}_meta.json"
        if meta_path.exists():
            data = DataPassingManager.load_artifact(meta_path)
            output = PreparedDataOutput(**data)
            logger.info(f"Stage 3B complete: Data saved to {data.get('prepared_file_path')}")
            return output
        else:
            # Create minimal output
            parquet_path = STAGE3B_OUT_DIR / f"prepared_{plan_id}.parquet"
            if parquet_path.exists():
                import pandas as pd
                df = pd.read_parquet(parquet_path)
                output = PreparedDataOutput(
                    plan_id=plan_id,
                    prepared_file_path=str(parquet_path),
                    original_row_count=len(df),
                    final_row_count=len(df),
                    columns_created=list(df.columns),
                    data_quality_report={
                        "total_rows": len(df),
                        "total_columns": len(df.columns),
                        "null_counts": df.isnull().sum().to_dict()
                    },
                    has_no_nulls=df.isnull().sum().sum() == 0
                )
                return output
            else:
                raise RuntimeError("Prepared data was not saved")

    except Exception as e:
        logger.error(f"Stage 3B failed: {e}")
        raise


# ============================================================================
# PIPELINE NODE FUNCTION
# ============================================================================

def stage3b_node(state: PipelineState) -> PipelineState:
    """
    Stage 3B node for the master pipeline graph.
    """
    state.mark_stage_started("stage3b")

    plan_id = f"PLAN-{state.selected_task_id}" if state.selected_task_id else None
    if not plan_id:
        state.mark_stage_failed("stage3b", "No plan ID available")
        return state

    try:
        output = run_stage3b(plan_id, state)
        state.stage3b_output = output
        state.mark_stage_completed("stage3b", output)
    except Exception as e:
        state.mark_stage_failed("stage3b", str(e))

    return state


if __name__ == "__main__":
    import sys
    plan_id = sys.argv[1] if len(sys.argv) > 1 else "PLAN-TSK-001"
    output = run_stage3b(plan_id)
    print(f"Prepared data: {output.prepared_file_path}")
    print(f"Rows: {output.final_row_count}")
    print(f"No nulls: {output.has_no_nulls}")
