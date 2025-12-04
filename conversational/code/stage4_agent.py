"""
Stage 4 Agent: Execution

This agent executes the selected method and generates predictions.
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
    STAGE3_OUT_DIR, STAGE3B_OUT_DIR, STAGE3_5B_OUT_DIR,
    STAGE4_OUT_DIR, STAGE4_WORKSPACE, SECONDARY_LLM_CONFIG,
    STAGE_MAX_ROUNDS, DataPassingManager, logger
)
from code.models import ExecutionResult, ExecutionStatus, PipelineState
from tools.stage4_tools import STAGE4_TOOLS


# ============================================================================
# STATE DEFINITION
# ============================================================================

class Stage4State(BaseModel):
    """State for Stage 4 agent."""
    messages: Annotated[list, add_messages] = []
    plan_id: str = ""
    context_loaded: bool = False
    execution_complete: bool = False
    iteration: int = 0
    complete: bool = False


# ============================================================================
# SYSTEM PROMPT
# ============================================================================

STAGE4_SYSTEM_PROMPT = """You are an Execution Agent responsible for running the selected forecasting method.

## Your Role
1. Load the execution context (plan, data, selected method)
2. Execute the selected method from Stage 3.5B
3. Generate predictions for the test set
4. Calculate evaluation metrics
5. Save comprehensive results

## CRITICAL: METRIC CONSISTENCY WITH STAGE 3.5B
Your execution MUST produce metrics that match (or are very close to) the benchmark
metrics from Stage 3.5B. If your MAE/RMSE differ significantly, you are likely using
a different data split. CHECK THE DATA SPLIT STRATEGY CAREFULLY.

## Your Goals
- Execute the winning method from benchmarking
- Generate predictions with actual vs predicted values
- Calculate final metrics (MAE, RMSE, MAPE, R²) that MATCH benchmark
- Save results in a format suitable for visualization

## Available Tools
- load_execution_context: Get plan, data info, and selected method
- load_prepared_data: Load and inspect the prepared data
- get_selected_method_code: Get implementation code for winner (INCLUDES benchmark metrics)
- execute_python_code: Run Python for model execution
- save_predictions: Save predictions to parquet
- save_execution_result: Save execution metadata
- verify_execution: Verify outputs are correct
- list_stage4_results: List existing results

## Execution Workflow
1. Load execution context for the plan
2. **CRITICAL**: Call get_selected_method_code to get:
   - The winning method's implementation code
   - The EXACT data split strategy used in benchmarking
   - The benchmark metrics (your results should match these)
3. Load prepared data
4. **CRITICAL**: Split data according to the EXACT strategy from Stage 3.5B
5. Execute the method using the EXACT same code
6. Calculate metrics - they should match benchmark
7. Create results DataFrame with:
   - Date/index column
   - Actual values
   - Predicted values
   - Any relevant features
8. Save predictions and execution result
9. Verify the outputs

## DATA SPLIT STRATEGY (MUST FOLLOW EXACTLY)
The get_selected_method_code tool returns the data_split_strategy JSON.
You MUST use the EXACT same split to get matching metrics:

- strategy_type: "temporal" = row-based temporal split
- strategy_type: "temporal_column" = wide format (column-based, use all rows)
- train_size, validation_size, test_size: The exact percentages to use

## Results DataFrame Requirements
The saved predictions must include:
- date/index column (for time series plotting)
- 'actual' column (true values)
- 'predicted' column (model predictions)
- Keep original feature columns for context

## Execution Code Template
```python
import pandas as pd
import numpy as np
from pathlib import Path

# Load prepared data
STAGE3B_OUT_DIR = Path('{STAGE3B_OUT_DIR}')
df = pd.read_parquet(STAGE3B_OUT_DIR / 'prepared_{{plan_id}}.parquet')

# Get column info from execution context
target_col = 'your_target_column'  # From get_selected_method_code
date_col = 'your_date_column'       # From get_selected_method_code

# CRITICAL: Use EXACT split strategy from Stage 3.5B
# These values should come from the data_split_strategy in get_selected_method_code:
train_size = 0.7   # From data_split_strategy.train_size
val_size = 0.15    # From data_split_strategy.validation_size
test_size = 0.15   # From data_split_strategy.test_size

# Apply split based on strategy_type
train_end = int(len(df) * train_size)
val_end = int(len(df) * (train_size + val_size))

train_df = df.iloc[:train_end].copy()
test_df = df.iloc[val_end:].copy()  # Skip validation, use test only

# Define the selected method (COPY EXACTLY from get_selected_method_code)
def predict_selected_method(train_df, test_df, target_col, date_col, **params):
    # ... implementation from Stage 3.5B winner ...
    pass

# Run prediction
predictions = predict_selected_method(train_df, test_df, target_col, date_col)

# Create results DataFrame
results_df = test_df.copy()
results_df['predicted'] = predictions['predicted'].values
results_df['actual'] = results_df[target_col]

# Calculate metrics
actual = results_df['actual'].values
predicted = results_df['predicted'].values

mae = np.mean(np.abs(actual - predicted))
rmse = np.sqrt(np.mean((actual - predicted) ** 2))
mask = actual != 0
mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100 if mask.any() else 0.0
r2 = 1 - (np.sum((actual - predicted)**2) / np.sum((actual - np.mean(actual))**2)) if len(actual) > 1 else 0.0

# VERIFY: Compare with benchmark metrics
# Expected MAE from Stage 3.5B: X.XXXX
# Your MAE should be close to this value!
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.2f}%")
print(f"R²: {r2:.4f}")
```

## Error Handling
- If the selected method fails, try to fix and retry
- If still failing, fall back to a simple baseline
- Always produce some output even if suboptimal

IMPORTANT: The results_df must be saved as parquet for Stage 5 visualization.
"""


# ============================================================================
# AGENT LOGIC
# ============================================================================

def create_stage4_agent():
    """Create the Stage 4 agent graph."""

    llm = ChatOpenAI(**SECONDARY_LLM_CONFIG)
    llm_with_tools = llm.bind_tools(STAGE4_TOOLS, parallel_tool_calls=False)

    def agent_node(state: Stage4State) -> Dict[str, Any]:
        """Main agent reasoning node."""
        messages = state.messages

        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=STAGE4_SYSTEM_PROMPT)] + list(messages)

        if state.iteration >= STAGE_MAX_ROUNDS.get("stage4", 100):
            return {
                "messages": [AIMessage(content="Maximum iterations reached. Finalizing.")],
                "complete": True
            }

        response = llm_with_tools.invoke(messages)

        return {
            "messages": [response],
            "iteration": state.iteration + 1
        }

    def should_continue(state: Stage4State) -> str:
        """Determine if we should continue or end."""
        if state.complete:
            return "end"

        if state.messages:
            last_message = state.messages[-1]
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "tools"

        return "end"

    builder = StateGraph(Stage4State)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(STAGE4_TOOLS))
    builder.set_entry_point("agent")
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END}
    )
    builder.add_edge("tools", "agent")

    return builder.compile(checkpointer=MemorySaver())


def run_stage4(plan_id: str, pipeline_state: PipelineState = None) -> ExecutionResult:
    """
    Run Stage 4: Execution.

    Executes the selected method and generates predictions.
    """
    logger.info(f"Starting Stage 4: Execution for {plan_id}")

    graph = create_stage4_agent()

    initial_message = HumanMessage(content=f"""
Execute forecasting for plan: {plan_id}

Steps:
1. Load execution context (plan, data info, selected method)
2. Get the selected method's implementation code from Stage 3.5B
3. Load the prepared data
4. Split data according to plan's validation strategy
5. Execute the selected method
6. Calculate final metrics (MAE, RMSE, MAPE, R²)
7. Create results DataFrame with actual and predicted values
8. Save predictions parquet and execution result JSON using save_predictions tool
9. Verify outputs are correct

The results should include:
- Date/index column for time series plots
- 'actual' column with true values
- 'predicted' column with model predictions
- Original relevant columns for context

IMPORTANT: You MUST use save_predictions tool to save the results.

Save outputs:
- Predictions: {STAGE4_OUT_DIR}/results_{plan_id}.parquet
- Metadata: {STAGE4_OUT_DIR}/execution_result_{plan_id}.json
""")

    config = {"configurable": {"thread_id": f"stage4_{plan_id}"}}
    initial_state = Stage4State(messages=[initial_message], plan_id=plan_id)

    try:
        final_state = graph.invoke(initial_state, config)

        # Load execution result from disk
        result_path = STAGE4_OUT_DIR / f"execution_result_{plan_id}.json"
        if result_path.exists():
            data = DataPassingManager.load_artifact(result_path)
            output = ExecutionResult(**data)
            logger.info(f"Stage 4 complete: {output.status}")
            return output
        else:
            # Check if predictions exist
            predictions_path = STAGE4_OUT_DIR / f"results_{plan_id}.parquet"
            if predictions_path.exists():
                output = ExecutionResult(
                    plan_id=plan_id,
                    status=ExecutionStatus.SUCCESS,
                    outputs={"predictions": str(predictions_path)},
                    summary="Execution completed"
                )
                # Save execution result
                DataPassingManager.save_artifact(
                    data=output.model_dump(),
                    output_dir=STAGE4_OUT_DIR,
                    filename=f"execution_result_{plan_id}.json",
                    metadata={"stage": "stage4", "type": "execution_result"}
                )
                return output

            # Fallback: create default predictions
            logger.warning("Agent failed to create predictions, creating fallback")
            output = _create_fallback_execution(plan_id)
            return output

    except Exception as e:
        logger.error(f"Stage 4 failed: {e}")
        # Try fallback
        try:
            logger.warning("Creating fallback execution after exception")
            output = _create_fallback_execution(plan_id)
            return output
        except Exception as fallback_error:
            logger.error(f"Fallback also failed: {fallback_error}")
            return ExecutionResult(
                plan_id=plan_id,
                status=ExecutionStatus.FAILURE,
                summary=f"Execution failed: {e}",
                errors=[str(e)]
            )


def _create_fallback_execution(plan_id: str) -> ExecutionResult:
    """
    Create execution using winning method code from Stage 3.5B.
    Only falls back to naive prediction if method execution fails.

    CRITICAL: Uses the EXACT same data split strategy as Stage 3.5B
    to ensure metric consistency.
    """
    import pandas as pd
    import numpy as np
    import sys
    from io import StringIO

    try:
        # Load prepared data
        prepared_path = STAGE3B_OUT_DIR / f"prepared_{plan_id}.parquet"
        if not prepared_path.exists():
            raise FileNotFoundError(f"Prepared data not found: {prepared_path}")

        df = pd.read_parquet(prepared_path)

        # Get target column from plan
        plan_path = STAGE3_OUT_DIR / f"{plan_id}.json"
        plan = DataPassingManager.load_artifact(plan_path)
        target_col = plan.get('target_column')

        if not target_col or target_col not in df.columns:
            # Find a numeric column
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            target_col = numeric_cols[-1] if numeric_cols else df.columns[-1]

        # ================================================================
        # PRIORITY 1: Try to execute winning method from Stage 3.5B
        # ================================================================
        tester_path = STAGE3_5B_OUT_DIR / f"tester_{plan_id}.json"
        if tester_path.exists():
            try:
                tester = DataPassingManager.load_artifact(tester_path)
                winning_code = tester.get('winning_method_code')

                if winning_code:
                    logger.info(f"Attempting to execute winning method {tester.get('selected_method_id')}")

                    # ============================================================
                    # CRITICAL: Use the EXACT same split strategy as Stage 3.5B
                    # ============================================================
                    split_strategy = tester.get('data_split_strategy', {})
                    strategy_type = split_strategy.get('strategy_type', 'temporal')
                    train_size_pct = split_strategy.get('train_size', 0.7)
                    val_size_pct = split_strategy.get('validation_size', 0.15)
                    test_size_pct = split_strategy.get('test_size', 0.15)
                    date_col = tester.get('date_column') or split_strategy.get('date_column')

                    # Apply the exact same split as Stage 3.5B
                    if strategy_type == 'temporal_column':
                        # Wide format - use column-based split (all rows, different columns)
                        logger.info("Using temporal_column (wide format) split strategy")
                        train_df = df.copy()
                        test_df = df.copy()
                    elif date_col and date_col in df.columns:
                        # Temporal split based on date column
                        logger.info(f"Using temporal split on {date_col}")
                        df = df.sort_values(date_col)
                        train_end_idx = int(len(df) * train_size_pct)
                        val_end_idx = int(len(df) * (train_size_pct + val_size_pct))
                        train_df = df.iloc[:train_end_idx].copy()
                        test_df = df.iloc[val_end_idx:].copy()
                    else:
                        # Default row-based split
                        logger.info(f"Using row-based split: train={train_size_pct}, val={val_size_pct}, test={test_size_pct}")
                        train_end_idx = int(len(df) * train_size_pct)
                        val_end_idx = int(len(df) * (train_size_pct + val_size_pct))
                        train_df = df.iloc[:train_end_idx].copy()
                        test_df = df.iloc[val_end_idx:].copy()

                    # Override target_col from tester if available
                    if tester.get('target_column') and tester.get('target_column') in df.columns:
                        target_col = tester.get('target_column')

                    # Setup execution namespace
                    namespace = {
                        'pd': pd,
                        'np': np,
                        'train_df': train_df,
                        'test_df': test_df,
                        'target_col': target_col,
                        'date_col': date_col,
                        'df': df,  # Full dataframe for wide format methods
                    }

                    # Add ML imports
                    try:
                        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
                        from sklearn.linear_model import LinearRegression, Ridge, Lasso
                        namespace['RandomForestRegressor'] = RandomForestRegressor
                        namespace['GradientBoostingRegressor'] = GradientBoostingRegressor
                        namespace['LinearRegression'] = LinearRegression
                        namespace['Ridge'] = Ridge
                        namespace['Lasso'] = Lasso
                    except ImportError:
                        pass

                    # Execute the winning method code
                    exec(winning_code, namespace)

                    # Get the prediction function name
                    func_name = None
                    for name, obj in namespace.items():
                        if callable(obj) and name.startswith('predict_'):
                            func_name = name
                            break

                    if func_name:
                        predict_func = namespace[func_name]
                        predictions_df = predict_func(train_df, test_df, target_col, date_col)

                        # Create results DataFrame
                        results_df = test_df.copy()
                        results_df['actual'] = results_df[target_col]
                        if isinstance(predictions_df, pd.DataFrame) and 'predicted' in predictions_df.columns:
                            results_df['predicted'] = predictions_df['predicted'].values
                        else:
                            results_df['predicted'] = np.array(predictions_df).flatten()

                        # Calculate metrics
                        actual = results_df['actual'].values
                        predicted = results_df['predicted'].values

                        mae = np.mean(np.abs(actual - predicted))
                        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
                        mask = actual != 0
                        mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100 if mask.sum() > 0 else 0.0
                        r2 = 1 - (np.sum((actual - predicted)**2) / np.sum((actual - np.mean(actual))**2)) if len(actual) > 1 else 0.0

                        # Compare with benchmark metrics
                        benchmark_metrics = tester.get('benchmark_metrics', {})
                        if benchmark_metrics:
                            benchmark_mae = benchmark_metrics.get('mae')
                            if benchmark_mae:
                                diff_pct = abs(mae - benchmark_mae) / benchmark_mae * 100 if benchmark_mae > 0 else 0
                                if diff_pct > 10:
                                    logger.warning(f"Metric difference from benchmark: MAE {mae:.4f} vs {benchmark_mae:.4f} ({diff_pct:.1f}% diff)")
                                else:
                                    logger.info(f"Metrics match benchmark: MAE {mae:.4f} vs {benchmark_mae:.4f} ({diff_pct:.1f}% diff)")

                        # Save predictions
                        predictions_path = STAGE4_OUT_DIR / f"results_{plan_id}.parquet"
                        results_df.to_parquet(predictions_path, index=False)

                        # Create and save execution result
                        result = ExecutionResult(
                            plan_id=plan_id,
                            status=ExecutionStatus.SUCCESS,
                            outputs={"predictions": str(predictions_path)},
                            metrics={"mae": mae, "rmse": rmse, "mape": mape, "r2": r2},
                            summary=f"Executed {tester.get('selected_method_name', 'winning method')} (MAE: {mae:.4f})"
                        )

                        DataPassingManager.save_artifact(
                            data=result.model_dump(),
                            output_dir=STAGE4_OUT_DIR,
                            filename=f"execution_result_{plan_id}.json",
                            metadata={
                                "stage": "stage4",
                                "type": "execution_result",
                                "method": tester.get('selected_method_id'),
                                "benchmark_mae": benchmark_metrics.get('mae'),
                            }
                        )

                        logger.info(f"Winning method execution succeeded with MAE: {mae:.4f}")
                        return result
                    else:
                        logger.warning("Could not find prediction function in winning method code")

            except Exception as method_error:
                logger.warning(f"Winning method execution failed: {method_error}, falling back to naive prediction")
                import traceback
                logger.debug(traceback.format_exc())

        # ================================================================
        # PRIORITY 2: Fallback to naive prediction
        # ================================================================
        logger.warning("Using naive prediction fallback")

        # Use same split strategy for consistency
        split_strategy = {}
        if tester_path.exists():
            try:
                tester = DataPassingManager.load_artifact(tester_path)
                split_strategy = tester.get('data_split_strategy', {})
            except:
                pass

        train_size_pct = split_strategy.get('train_size', 0.7)
        val_size_pct = split_strategy.get('validation_size', 0.15)

        # Create train/test split
        train_end_idx = int(len(df) * train_size_pct)
        val_end_idx = int(len(df) * (train_size_pct + val_size_pct))
        train_df = df.iloc[:train_end_idx]
        test_df = df.iloc[val_end_idx:]

        # Simple naive prediction
        last_value = train_df[target_col].iloc[-1]

        results_df = test_df.copy()
        results_df['actual'] = results_df[target_col]
        results_df['predicted'] = last_value

        # Calculate metrics
        actual = results_df['actual'].values
        predicted = results_df['predicted'].values

        mae = np.mean(np.abs(actual - predicted))
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        mask = actual != 0
        mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100 if mask.sum() > 0 else 0.0
        r2 = 1 - (np.sum((actual - predicted)**2) / np.sum((actual - np.mean(actual))**2)) if len(actual) > 1 else 0.0

        # Save predictions
        predictions_path = STAGE4_OUT_DIR / f"results_{plan_id}.parquet"
        results_df.to_parquet(predictions_path, index=False)

        # Create and save execution result
        result = ExecutionResult(
            plan_id=plan_id,
            status=ExecutionStatus.SUCCESS,
            outputs={"predictions": str(predictions_path)},
            metrics={"mae": mae, "rmse": rmse, "mape": mape, "r2": r2},
            summary=f"Fallback execution with naive prediction (MAE: {mae:.4f})"
        )

        DataPassingManager.save_artifact(
            data=result.model_dump(),
            output_dir=STAGE4_OUT_DIR,
            filename=f"execution_result_{plan_id}.json",
            metadata={"stage": "stage4", "type": "execution_result", "fallback": True}
        )

        logger.info(f"Fallback execution saved to {predictions_path}")
        return result

    except Exception as e:
        logger.error(f"Fallback execution failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return ExecutionResult(
            plan_id=plan_id,
            status=ExecutionStatus.FAILURE,
            summary=f"Fallback execution failed: {e}",
            errors=[str(e)]
        )


# ============================================================================
# PIPELINE NODE FUNCTION
# ============================================================================

def stage4_node(state: PipelineState) -> PipelineState:
    """
    Stage 4 node for the master pipeline graph.
    """
    state.mark_stage_started("stage4")

    plan_id = f"PLAN-{state.selected_task_id}" if state.selected_task_id else None
    if not plan_id:
        state.mark_stage_failed("stage4", "No plan ID available")
        return state

    try:
        output = run_stage4(plan_id, state)
        state.stage4_output = output
        state.mark_stage_completed("stage4", output)
    except Exception as e:
        state.mark_stage_failed("stage4", str(e))

    return state


if __name__ == "__main__":
    import sys
    plan_id = sys.argv[1] if len(sys.argv) > 1 else "PLAN-TSK-001"
    output = run_stage4(plan_id)
    print(f"Execution status: {output.status}")
    print(f"Metrics: {output.metrics}")
