"""
Stage 3.5B Agent: Method Benchmarking

This agent benchmarks the proposed methods and selects the best performer.
Runs each method 3 times to ensure consistency and detect hallucinations.
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
    STAGE3B_OUT_DIR, STAGE3_5A_OUT_DIR, STAGE3_5B_OUT_DIR, SECONDARY_LLM_CONFIG,
    STAGE_MAX_ROUNDS, BENCHMARK_ITERATIONS, MAX_CV_THRESHOLD,
    DataPassingManager, logger
)
from code.models import TesterOutput, PipelineState
from tools.stage3_5b_tools import STAGE3_5B_TOOLS, reset_benchmark_state


# ============================================================================
# STATE DEFINITION
# ============================================================================

class Stage35BState(BaseModel):
    """State for Stage 3.5B agent."""
    messages: Annotated[list, add_messages] = []
    plan_id: str = ""
    methods_loaded: bool = False
    methods_tested: list = []
    best_method: str = ""
    iteration: int = 0
    complete: bool = False


# ============================================================================
# SYSTEM PROMPT
# ============================================================================

STAGE35B_SYSTEM_PROMPT = f"""You are a Method Benchmarking Agent responsible for testing and selecting the best forecasting method.

## Your Role
1. Load method proposals from Stage 3.5A
2. Run each method {BENCHMARK_ITERATIONS} times for consistency
3. Calculate metrics (MAE, RMSE, MAPE)
4. Validate results aren't hallucinated (check consistency)
5. Select the best method based on average performance
6. Save comprehensive benchmark results

## Consistency Validation (CRITICAL)
You MUST run each method {BENCHMARK_ITERATIONS} times to check consistency:
- Calculate coefficient of variation (CV) of metrics across runs
- If CV < {MAX_CV_THRESHOLD}: Results are VALID
- If CV >= {MAX_CV_THRESHOLD}: Results may be HALLUCINATED

This prevents fake/random results from being selected.

## Available Tools
- load_method_proposals: Load methods from Stage 3.5A
- load_checkpoint: Resume from previous run if exists
- save_checkpoint: Save progress after each method
- record_thought_3_5b: Document reasoning
- run_benchmark_code: Execute method and get metrics
- calculate_metrics: Compute MAE, RMSE, MAPE
- validate_consistency: Check if results are consistent
- select_best_method: Select winner based on results
- save_tester_output: Save final results

## Benchmark Code Structure
For each method, your benchmark code should:
```python
import pandas as pd
import numpy as np
from pathlib import Path

# Load prepared data
STAGE3B_OUT_DIR = Path('{STAGE3B_OUT_DIR}')
df = pd.read_parquet(STAGE3B_OUT_DIR / 'prepared_{{plan_id}}.parquet')

# Define the method function (from proposal)
def predict_method(train_df, test_df, target_col, date_col, **params):
    # ... implementation ...
    return pd.DataFrame({{'predicted': predictions}}, index=test_df.index)

# Split data (use the split strategy from proposal)
# ...

# Run prediction
predictions = predict_method(train_df, test_df, target_col, date_col)

# Calculate metrics
actual = test_df[target_col].values
predicted = predictions['predicted'].values

mae = np.mean(np.abs(actual - predicted))
rmse = np.sqrt(np.mean((actual - predicted) ** 2))
mape = np.mean(np.abs((actual - predicted) / actual)) * 100

# Print results as JSON
import json
print(json.dumps({{"mae": mae, "rmse": rmse, "mape": mape}}))
```

## Workflow
1. Load method proposals
2. Check for existing checkpoint (resume if exists)
3. For each method (M1, M2, M3):
   a. THINK about how to run this method
   b. Run {BENCHMARK_ITERATIONS} iterations
   c. Collect metrics from each run
   d. Validate consistency (check CV)
   e. Save checkpoint after completing method
4. Compare all methods
5. Select best method (lowest average MAE for valid methods)
6. Save tester output

## Error Handling
- If a method fails, record the error and move to next
- If all iterations fail, mark method as invalid
- At least one method must succeed for stage to complete

## Output Requirements
The tester output must include:
- methods_tested: List of results for each method
- selected_method_id: Winner (e.g., "M2")
- selected_method_name: Winner's name
- selection_rationale: Why this method was selected
- method_comparison_summary: Brief comparison of all methods

IMPORTANT: Use checkpoints! If interrupted, you can resume from the last completed method.
"""


# ============================================================================
# AGENT LOGIC
# ============================================================================

def create_stage3_5b_agent():
    """Create the Stage 3.5B agent graph."""

    llm = ChatOpenAI(**SECONDARY_LLM_CONFIG)
    llm_with_tools = llm.bind_tools(STAGE3_5B_TOOLS, parallel_tool_calls=False)

    def agent_node(state: Stage35BState) -> Dict[str, Any]:
        """Main agent reasoning node."""
        messages = state.messages

        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=STAGE35B_SYSTEM_PROMPT)] + list(messages)

        if state.iteration >= STAGE_MAX_ROUNDS.get("stage3_5b", 120):
            return {
                "messages": [AIMessage(content="Maximum iterations reached. Finalizing with current results.")],
                "complete": True
            }

        response = llm_with_tools.invoke(messages)

        return {
            "messages": [response],
            "iteration": state.iteration + 1
        }

    def should_continue(state: Stage35BState) -> str:
        """Determine if we should continue or end."""
        if state.complete:
            return "end"

        if state.messages:
            last_message = state.messages[-1]
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "tools"

        return "end"

    builder = StateGraph(Stage35BState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(STAGE3_5B_TOOLS))
    builder.set_entry_point("agent")
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END}
    )
    builder.add_edge("tools", "agent")

    return builder.compile(checkpointer=MemorySaver())


def run_stage3_5b(plan_id: str, pipeline_state: PipelineState = None) -> TesterOutput:
    """
    Run Stage 3.5B: Method Benchmarking.

    Tests all proposed methods and selects the best.
    """
    logger.info(f"Starting Stage 3.5B: Method Benchmarking for {plan_id}")

    reset_benchmark_state()
    graph = create_stage3_5b_agent()

    initial_message = HumanMessage(content=f"""
Benchmark forecasting methods for plan: {plan_id}

Steps:
1. Load method proposals from Stage 3.5A
2. Check for existing checkpoint (resume if available)
3. For each of the 3 methods (M1, M2, M3):
   a. Run {BENCHMARK_ITERATIONS} iterations
   b. Calculate metrics each time
   c. Validate consistency (CV < {MAX_CV_THRESHOLD})
   d. Save checkpoint after completing the method
4. Compare all valid methods
5. Select the best method (lowest average MAE)
6. Save tester output

The prepared data is at: {STAGE3B_OUT_DIR}/prepared_{plan_id}.parquet

Save output as: tester_{plan_id}.json

Remember: Run each method {BENCHMARK_ITERATIONS} times and check consistency!
""")

    config = {"configurable": {"thread_id": f"stage3_5b_{plan_id}"}}
    initial_state = Stage35BState(messages=[initial_message], plan_id=plan_id)

    try:
        final_state = graph.invoke(initial_state, config)

        # Load tester output from disk
        tester_path = STAGE3_5B_OUT_DIR / f"tester_{plan_id}.json"
        if tester_path.exists():
            data = DataPassingManager.load_artifact(tester_path)
            output = TesterOutput(**data)
            logger.info(f"Stage 3.5B complete: Selected {output.selected_method_id}")
            return output
        else:
            raise RuntimeError("Tester output was not saved")

    except Exception as e:
        logger.error(f"Stage 3.5B failed: {e}")
        raise


# ============================================================================
# PIPELINE NODE FUNCTION
# ============================================================================

def stage3_5b_node(state: PipelineState) -> PipelineState:
    """
    Stage 3.5B node for the master pipeline graph.
    """
    state.mark_stage_started("stage3_5b")

    plan_id = f"PLAN-{state.selected_task_id}" if state.selected_task_id else None
    if not plan_id:
        state.mark_stage_failed("stage3_5b", "No plan ID available")
        return state

    try:
        output = run_stage3_5b(plan_id, state)
        state.stage3_5b_output = output
        state.mark_stage_completed("stage3_5b", output)
    except Exception as e:
        state.mark_stage_failed("stage3_5b", str(e))

    return state


if __name__ == "__main__":
    import sys
    plan_id = sys.argv[1] if len(sys.argv) > 1 else "PLAN-TSK-001"
    output = run_stage3_5b(plan_id)
    print(f"Selected method: {output.selected_method_id} - {output.selected_method_name}")
    print(f"Rationale: {output.selection_rationale}")
