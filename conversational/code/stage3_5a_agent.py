"""
Stage 3.5A Agent: Method Proposal

This agent proposes exactly 3 forecasting methods to benchmark,
ranging from simple baselines to machine learning approaches.
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
    STAGE3_OUT_DIR, STAGE3B_OUT_DIR, STAGE3_5A_OUT_DIR, SECONDARY_LLM_CONFIG,
    STAGE_MAX_ROUNDS, DataPassingManager, logger, DEBUG, RECURSION_LIMIT
)
from code.models import MethodProposalOutput, PipelineState
from tools.stage3_5a_tools import STAGE3_5A_TOOLS, reset_react_state


# ============================================================================
# STATE DEFINITION
# ============================================================================

class Stage35AState(BaseModel):
    """State for Stage 3.5A agent."""
    messages: Annotated[list, add_messages] = []
    plan_id: str = ""
    data_analyzed: bool = False
    methods_proposed: bool = False
    iteration: int = 0
    complete: bool = False


# ============================================================================
# SYSTEM PROMPT
# ============================================================================

STAGE35A_SYSTEM_PROMPT = """You are a Method Proposal Agent responsible for proposing forecasting methods.

## CRITICAL: Prevent Column Hallucination
❌ DO NOT assume column names exist (e.g., 'Year', 'date', 'time')
✅ ALWAYS call get_actual_columns() FIRST to see real columns
✅ Use ONLY columns that actually exist in the prepared data

## Your Role
Analyze the data and task, then propose EXACTLY 3 forecasting methods:
1. A simple BASELINE method (moving average, naive, etc.)
2. A STATISTICAL method (ARIMA, exponential smoothing, etc.)
3. A MACHINE LEARNING method (random forest, gradient boosting, etc.)

## Your Goals
1. **FIRST**: Call get_actual_columns() to see what columns exist
2. Understand the data structure and time series characteristics
3. Analyze temporal patterns (trend, seasonality, frequency)
4. Propose 3 diverse, appropriate methods
5. Write complete, executable implementation code using ONLY existing columns
6. Define data split strategy

## Available Tools
- get_actual_columns: **CALL THIS FIRST** to prevent column hallucination
- load_plan_and_data: Load execution plan and prepared data info
- analyze_time_series: Analyze time series characteristics
- record_thought_3_5a: Document reasoning (ReAct)
- record_observation_3_5a: Document observations (ReAct)
- python_sandbox_stage3_5a: Execute Python for analysis
- get_method_templates: Get method implementation templates
- save_method_proposal: Save final proposals
- finish_method_proposal: Signal completion (Call this LAST)

## Method Requirements
Each method MUST include:
- method_id: M1, M2, or M3
- name: Clear method name
- category: "baseline", "statistical", or "ml"
- description: What the method does
- implementation_code: COMPLETE, EXECUTABLE Python function using ONLY existing columns
- required_libraries: List of imports needed
- hyperparameters: Key parameters and their values
- expected_strengths: What this method does well
- expected_weaknesses: Known limitations

## Implementation Code Structure
Each method's code must be a complete function:
```python
def predict_method_name(train_df, test_df, target_col, date_col, **params):
    import pandas as pd
    import numpy as np
    # ... (imports and implementation)

    # Return predictions DataFrame with 'predicted' column
    return pd.DataFrame({'predicted': predictions}, index=test_df.index)
```

## Data Split Strategy
You must also specify:
- strategy_type: "temporal" (preferred for time series) or "random"
- date_column: Column used for splitting
- target_column: Column being predicted
- train_period: Training data range/size
- validation_period: Validation data range/size
- test_period: Test data range/size

## Workflow
1. Load plan and prepared data info
2. Analyze time series characteristics (trend, seasonality, frequency)
3. THINK about which methods are appropriate
4. Get method templates for reference
5. Create 3 methods with complete code
6. Define data split strategy
7. Save the method proposal
8. Call finish_method_proposal() to end the stage

## Method Selection Guidelines
- For SHORT time series (<100 points): Prefer simple methods
- For SEASONAL data: Include methods that handle seasonality
- For TRENDING data: Include methods that capture trends
- Always include a simple baseline for comparison

IMPORTANT: The implementation code must be complete and run without errors.
The benchmarking stage will execute this code exactly as written.
"""


# ============================================================================
# AGENT LOGIC
# ============================================================================

def create_stage3_5a_agent():
    """Create the Stage 3.5A agent graph."""

    llm = ChatOpenAI(**SECONDARY_LLM_CONFIG)
    llm_with_tools = llm.bind_tools(STAGE3_5A_TOOLS, parallel_tool_calls=False)

    def agent_node(state: Stage35AState) -> Dict[str, Any]:
        """Main agent reasoning node."""
        messages = state.messages

        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=STAGE35A_SYSTEM_PROMPT)] + list(messages)

        if state.iteration >= STAGE_MAX_ROUNDS.get("stage3_5a", 35):
            return {
                "messages": [AIMessage(content="Maximum iterations reached.")],
                "complete": True
            }

        response = llm_with_tools.invoke(messages)

        if DEBUG:
            logger.debug(f"Stage 3.5A Agent Response: {response.content}")
            if response.tool_calls:
                logger.debug(f"Tool Calls: {response.tool_calls}")

        return {
            "messages": [response],
            "iteration": state.iteration + 1
        }

    def should_continue(state: Stage35AState) -> str:
        """Determine if we should continue or end."""
        if state.complete:
            return "end"

        if state.messages:
            last_message = state.messages[-1]
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "tools"

        return "end"

    builder = StateGraph(Stage35AState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(STAGE3_5A_TOOLS))
    builder.set_entry_point("agent")
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END}
    )
    builder.add_edge("tools", "agent")

    return builder.compile(checkpointer=MemorySaver())


def run_stage3_5a(plan_id: str, pipeline_state: PipelineState = None) -> MethodProposalOutput:
    """
    Run Stage 3.5A: Method Proposal.

    Proposes 3 forecasting methods for benchmarking.
    """
    logger.info(f"Starting Stage 3.5A: Method Proposal for {plan_id}")

    reset_react_state()
    graph = create_stage3_5a_agent()

    initial_message = HumanMessage(content=f"""
Propose forecasting methods for plan: {plan_id}

Steps:
1. Load the plan and data information
2. Analyze time series characteristics
3. Get method templates for reference
4. Propose EXACTLY 3 methods:
   - M1: Simple baseline (e.g., moving average, naive)
   - M2: Statistical method (e.g., ARIMA, exponential smoothing)
   - M3: Machine learning (e.g., random forest with lag features)
5. Write COMPLETE implementation code for each
6. Define data split strategy (prefer temporal split)
7. Save the method proposal using save_method_proposal tool

Each method's code must:
- Be a complete, runnable function
- Accept train_df, test_df, target_col, date_col parameters
- Return a DataFrame with 'predicted' column

IMPORTANT: You MUST call save_method_proposal with a valid JSON object.
The JSON must have these fields:
- plan_id: "{plan_id}"
- methods_proposed: list of 3 method objects
- data_split_strategy: object with split info
- date_column: column name (or null)
- target_column: column name

Save output as: method_proposal_{plan_id}.json
""")

    config = {
        "configurable": {"thread_id": f"stage3_5a_{plan_id}"},
        "recursion_limit": RECURSION_LIMIT
    }
    initial_state = Stage35AState(messages=[initial_message], plan_id=plan_id)

    try:
        final_state = graph.invoke(initial_state, config)

        # Load proposal from disk
        proposal_path = STAGE3_5A_OUT_DIR / f"method_proposal_{plan_id}.json"
        if proposal_path.exists():
            data = DataPassingManager.load_artifact(proposal_path)
            output = MethodProposalOutput(**data)
            logger.info(f"Stage 3.5A complete: {len(output.methods_proposed)} methods proposed")
            return output
        else:
            # NO FALLBACK - raise an error so the pipeline can retry
            logger.error("Agent failed to save method proposal. Stage must be retried.")
            raise RuntimeError(
                "Stage 3.5A failed: Agent did not save method proposal. "
                "This may be due to max_tokens error or other agent failure. "
                "The stage should be retried."
            )

    except Exception as e:
        # Check if it's a max_tokens error
        error_msg = str(e).lower()
        if 'max_tokens' in error_msg or 'token' in error_msg:
            logger.error(f"Stage 3.5A failed with token error: {e}")
            raise RuntimeError(
                f"Stage 3.5A failed due to max_tokens error: {e}. "
                "This stage needs to be retried."
            )
        else:
            logger.error(f"Stage 3.5A failed: {e}")
            raise


# Removed _create_default_method_proposal function
# No more fallback logic - stages must complete successfully or fail properly


# ============================================================================
# PIPELINE NODE FUNCTION
# ============================================================================

def stage3_5a_node(state: PipelineState) -> PipelineState:
    """
    Stage 3.5A node for the master pipeline graph.
    Includes automatic retry logic for transient failures.
    """
    from code.config import MAX_RETRIES, RETRY_STAGES
    
    state.mark_stage_started("stage3_5a")

    plan_id = f"PLAN-{state.selected_task_id}" if state.selected_task_id else None
    if not plan_id:
        state.mark_stage_failed("stage3_5a", "No plan ID available")
        return state

    # Retry logic for resilient execution
    max_retries = MAX_RETRIES if "stage3_5a" in RETRY_STAGES else 1
    last_error = None
    
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Stage 3.5A attempt {attempt}/{max_retries}")
            output = run_stage3_5a(plan_id, state)
            state.stage3_5a_output = output
            state.mark_stage_completed("stage3_5a", output)
            logger.info(f"✅ Stage 3.5A succeeded on attempt {attempt}")
            return state
        except Exception as e:
            last_error = e
            error_msg = str(e).lower()
            
            # Check if it's a retryable error
            is_retryable = (
                'max_tokens' in error_msg or 
                'token' in error_msg or
                'did not save' in error_msg
            )
            
            if is_retryable and attempt < max_retries:
                logger.warning(
                    f"⚠️  Stage 3.5A attempt {attempt} failed with retryable error: {e}. "
                    f"Retrying... ({attempt}/{max_retries})"
                )
                # Clean up any partial outputs before retry
                proposal_path = STAGE3_5A_OUT_DIR / f"method_proposal_{plan_id}.json"
                if proposal_path.exists():
                    logger.info(f"Removing partial output: {proposal_path}")
                    proposal_path.unlink()
                continue
            else:
                # Non-retryable error or max retries reached
                if attempt >= max_retries:
                    logger.error(
                        f"❌ Stage 3.5A failed after {max_retries} attempts. "
                        f"Last error: {e}"
                    )
                state.mark_stage_failed("stage3_5a", str(last_error))
                return state
    
    # Should not reach here, but handle it
    state.mark_stage_failed("stage3_5a", str(last_error))
    return state


if __name__ == "__main__":
    import sys
    plan_id = sys.argv[1] if len(sys.argv) > 1 else "PLAN-TSK-001"
    output = run_stage3_5a(plan_id)
    print(f"Proposed {len(output.methods_proposed)} methods:")
    for m in output.methods_proposed:
        print(f"  - {m.method_id}: {m.name} ({m.category})")
