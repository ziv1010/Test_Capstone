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

## Your Goals
- Execute the winning method from benchmarking
- Generate predictions with actual vs predicted values
- Calculate final metrics (MAE, RMSE, MAPE, R²)
- Save results in a format suitable for visualization

## Available Tools
- load_execution_context: Get plan, data info, and selected method
- load_prepared_data: Load and inspect the prepared data
- get_selected_method_code: Get implementation code for winner
- execute_python_code: Run Python for model execution
- save_predictions: Save predictions to parquet
- save_execution_result: Save execution metadata
- verify_execution: Verify outputs are correct
- list_stage4_results: List existing results

## Execution Workflow
1. Load execution context for the plan
2. Get the selected method's implementation code
3. Load prepared data
4. Split data according to the plan's strategy
5. Execute the method
6. Calculate metrics
7. Create results DataFrame with:
   - Date/index column
   - Actual values
   - Predicted values
   - Any relevant features
8. Save predictions and execution result
9. Verify the outputs

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

# Parse dates if needed
date_col = 'your_date_column'
target_col = 'your_target_column'

df[date_col] = pd.to_datetime(df[date_col])
df = df.sort_values(date_col)

# Split data (temporal split for time series)
train_size = int(len(df) * 0.7)
val_size = int(len(df) * 0.15)

train_df = df.iloc[:train_size]
val_df = df.iloc[train_size:train_size+val_size]
test_df = df.iloc[train_size+val_size:]

# Define the selected method
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
mape = np.mean(np.abs((actual - predicted) / actual)) * 100
r2 = 1 - (np.sum((actual - predicted)**2) / np.sum((actual - np.mean(actual))**2))

print(f"MAE: {{mae:.4f}}")
print(f"RMSE: {{rmse:.4f}}")
print(f"MAPE: {{mape:.2f}}%")
print(f"R²: {{r2:.4f}}")
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
                    
                    # Create train/test split (same as Stage 3.5B)
                    train_size = int(len(df) * 0.8)
                    train_df = df.iloc[:train_size].copy()
                    test_df = df.iloc[train_size:].copy()
                    
                    # Setup execution namespace
                    namespace = {
                        'pd': pd,
                        'np': np,
                        'train_df': train_df,
                        'test_df': test_df,
                        'target_col': target_col,
                        'date_col': tester.get('date_column'),
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
                        predictions_df = predict_func(train_df, test_df, target_col)
                        
                        # Create results DataFrame
                        results_df = test_df.copy()
                        results_df['actual'] = results_df[target_col]
                        results_df['predicted'] = predictions_df['predicted'].values
                        
                        # Calculate metrics
                        actual = results_df['actual'].values
                        predicted = results_df['predicted'].values
                        
                        mae = np.mean(np.abs(actual - predicted))
                        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
                        mask = actual != 0
                        mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100 if mask.sum() > 0 else 0.0
                        
                        # Save predictions
                        predictions_path = STAGE4_OUT_DIR / f"results_{plan_id}.parquet"
                        results_df.to_parquet(predictions_path, index=False)
                        
                        # Create and save execution result
                        result = ExecutionResult(
                            plan_id=plan_id,
                            status=ExecutionStatus.SUCCESS,
                            outputs={"predictions": str(predictions_path)},
                            metrics={"mae": mae, "rmse": rmse, "mape": mape},
                            summary=f"Executed {tester.get('selected_method_name', 'winning method')} (MAE: {mae:.4f})"
                        )
                        
                        DataPassingManager.save_artifact(
                            data=result.model_dump(),
                            output_dir=STAGE4_OUT_DIR,
                            filename=f"execution_result_{plan_id}.json",
                            metadata={"stage": "stage4", "type": "execution_result", "method": tester.get('selected_method_id')}
                        )
                        
                        logger.info(f"Winning method execution succeeded with MAE: {mae:.4f}")
                        return result
                    else:
                        logger.warning("Could not find prediction function in winning method code")
                        
            except Exception as method_error:
                logger.warning(f"Winning method execution failed: {method_error}, falling back to naive prediction")

        # ================================================================
        # PRIORITY 2: Fallback to naive prediction
        # ================================================================
        logger.warning("Using naive prediction fallback")
        
        # Create train/test split
        train_size = int(len(df) * 0.8)
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]

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

        # Save predictions
        predictions_path = STAGE4_OUT_DIR / f"results_{plan_id}.parquet"
        results_df.to_parquet(predictions_path, index=False)

        # Create and save execution result
        result = ExecutionResult(
            plan_id=plan_id,
            status=ExecutionStatus.SUCCESS,
            outputs={"predictions": str(predictions_path)},
            metrics={"mae": mae, "rmse": rmse, "mape": mape},
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
