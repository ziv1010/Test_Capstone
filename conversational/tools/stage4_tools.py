"""
Stage 4 Tools: Execution

Tools for executing the selected method and generating predictions.
"""

import json
import time
from pathlib import Path
from typing import Optional
from langchain_core.tools import tool
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from code.config import (
    DATA_DIR, STAGE3_OUT_DIR, STAGE3B_OUT_DIR, STAGE3_5B_OUT_DIR,
    STAGE4_OUT_DIR, STAGE4_WORKSPACE, DataPassingManager, logger
)
from code.utils import load_dataframe, safe_json_dumps


@tool
def load_execution_context(plan_id: str = None) -> str:
    """
    Load all necessary context for execution.

    Loads: execution plan, prepared data info, and selected method.

    Args:
        plan_id: Plan ID. If not provided, loads most recent.

    Returns:
        Summary of execution context
    """
    try:
        # Find plan
        if plan_id:
            plan_path = STAGE3_OUT_DIR / f"{plan_id}.json"
        else:
            plans = list(STAGE3_OUT_DIR.glob("PLAN-*.json"))
            if not plans:
                return "No execution plans found."
            plan_path = max(plans, key=lambda p: p.stat().st_mtime)
            plan_id = plan_path.stem

        plan = DataPassingManager.load_artifact(plan_path)

        result = [
            f"=== Execution Context: {plan_id} ===",
            f"Goal: {plan.get('goal')}",
            f"Target: {plan.get('target_column')}",
            f"Category: {plan.get('task_category')}",
            "",
        ]

        # Check for prepared data
        prepared_path = STAGE3B_OUT_DIR / f"prepared_{plan_id}.parquet"
        if prepared_path.exists():
            df = pd.read_parquet(prepared_path)
            result.append(f"Prepared Data: {df.shape[0]} rows x {df.shape[1]} columns")
            result.append(f"Columns: {list(df.columns)}")
        else:
            result.append("Prepared Data: NOT FOUND - will use raw files")

        # Check for selected method
        tester_path = STAGE3_5B_OUT_DIR / f"tester_{plan_id}.json"
        if tester_path.exists():
            tester = DataPassingManager.load_artifact(tester_path)
            result.append(f"\nSelected Method: {tester.get('selected_method_id')} - {tester.get('selected_method_name')}")
            result.append(f"Selection Reason: {tester.get('selection_rationale')}")

            # Get implementation code
            for m in tester.get('methods_tested', []):
                if m.get('method_id') == tester.get('selected_method_id'):
                    result.append("\nImplementation available in tester output")
                    break
        else:
            result.append("\nSelected Method: NOT FOUND - will use default approach")

        result.append("\n\nPlan JSON:")
        result.append(json.dumps(plan, indent=2, default=str))

        return "\n".join(result)

    except Exception as e:
        return f"Error loading context: {e}"


@tool
def load_prepared_data(plan_id: str) -> str:
    """
    Load and summarize prepared data for execution.

    Args:
        plan_id: Plan ID

    Returns:
        Data summary and statistics
    """
    try:
        prepared_path = STAGE3B_OUT_DIR / f"prepared_{plan_id}.parquet"

        if not prepared_path.exists():
            return f"Prepared data not found: {prepared_path}"

        df = pd.read_parquet(prepared_path)

        result = [
            f"=== Prepared Data: {plan_id} ===",
            f"Shape: {df.shape[0]} rows x {df.shape[1]} columns",
            "",
            "Columns and Types:",
        ]

        for col in df.columns:
            dtype = df[col].dtype
            null_count = df[col].isna().sum()
            result.append(f"  {col}: {dtype} ({null_count} nulls)")

        result.append("\nNumeric Summary:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:10]:  # First 10
            result.append(f"  {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}")

        result.append(f"\nSample (first 5 rows):")
        result.append(df.head().to_string())

        return "\n".join(result)

    except Exception as e:
        return f"Error loading data: {e}"


@tool
def get_selected_method_code(plan_id: str) -> str:
    """
    Get the implementation code for the selected method.

    First tries tester output (Stage 3.5B) which has the verified winning code,
    then falls back to method proposal (Stage 3.5A) if not found.

    Args:
        plan_id: Plan ID

    Returns:
        Method details and implementation code
    """
    try:
        # ================================================================
        # PRIORITY 1: Load from tester output (has verified winning code)
        # ================================================================
        tester_path = STAGE3_5B_OUT_DIR / f"tester_{plan_id}.json"
        if tester_path.exists():
            tester = DataPassingManager.load_artifact(tester_path)
            
            # Check if winning method code is stored in tester output
            if tester.get('winning_method_code'):
                selected_id = tester.get('selected_method_id')
                selected_name = tester.get('selected_method_name', selected_id)
                
                result = [
                    f"=== Selected Method: {selected_id} (from Stage 3.5B tester output) ===",
                    f"Name: {selected_name}",
                    f"Target Column: {tester.get('target_column', 'N/A')}",
                    f"Date Column: {tester.get('date_column', 'N/A')}",
                    f"Libraries: {tester.get('winning_method_libraries', [])}",
                    "",
                    "Data Split Strategy:",
                    json.dumps(tester.get('data_split_strategy', {}), indent=2),
                    "",
                    "Implementation Code:",
                    "```python",
                    tester.get('winning_method_code'),
                    "```",
                    "",
                    "Hyperparameters:",
                    json.dumps(tester.get('winning_method_hyperparameters', {}), indent=2),
                    "",
                    "Benchmark Results:",
                    f"  Selection Rationale: {tester.get('selection_rationale', 'N/A')}",
                ]
                
                # Add benchmark metrics for the selected method
                for method in tester.get('methods_tested', []):
                    if method.get('method_id') == selected_id:
                        result.append(f"  Avg MAE: {method.get('avg_mae', 'N/A')}")
                        result.append(f"  Avg RMSE: {method.get('avg_rmse', 'N/A')}")
                        break
                
                logger.info(f"Retrieved winning method {selected_id} from tester output")
                return "\n".join(result)
        
        # ================================================================
        # PRIORITY 2: Fallback to method proposal (Stage 3.5A)
        # ================================================================
        proposal_path = STAGE3_5A_OUT_DIR / f"method_proposal_{plan_id}.json"
        if not proposal_path.exists():
            return "Method proposals not found."

        proposal = DataPassingManager.load_artifact(proposal_path)

        # Load tester output to get selected method ID
        if not tester_path.exists():
            return "Tester output not found - no method selected."

        tester = DataPassingManager.load_artifact(tester_path)
        selected_id = tester.get('selected_method_id')

        # Find the method in proposal
        for method in proposal.get('methods_proposed', []):
            if method.get('method_id') == selected_id:
                result = [
                    f"=== Selected Method: {selected_id} (from Stage 3.5A proposal) ===",
                    f"Name: {method.get('name')}",
                    f"Category: {method.get('category')}",
                    f"Description: {method.get('description')}",
                    f"Libraries: {method.get('required_libraries', [])}",
                    "",
                    "Implementation Code:",
                    "```python",
                    method.get('implementation_code', '# No code available'),
                    "```",
                    "",
                    "Hyperparameters:",
                    json.dumps(method.get('hyperparameters', {}), indent=2),
                ]
                return "\n".join(result)

        return f"Method {selected_id} not found in proposals."

    except Exception as e:
        return f"Error getting method code: {e}"


@tool
def execute_python_code(code: str, description: str = "") -> str:
    """
    Execute Python code for model training and prediction.

    Available: pd, np, sklearn, statsmodels, DATA_DIR, load_dataframe()
    Stage-specific dirs: STAGE3B_OUT_DIR, STAGE4_OUT_DIR, STAGE4_WORKSPACE

    Args:
        code: Python code to execute
        description: What the code does

    Returns:
        Execution output
    """
    import sys
    from io import StringIO

    namespace = {
        'pd': pd,
        'np': np,
        'json': json,
        'Path': Path,
        'time': time,
        'DATA_DIR': DATA_DIR,
        'STAGE3B_OUT_DIR': STAGE3B_OUT_DIR,
        'STAGE4_OUT_DIR': STAGE4_OUT_DIR,
        'STAGE4_WORKSPACE': STAGE4_WORKSPACE,
        'load_dataframe': load_dataframe,
    }

    # Import ML libraries
    try:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.linear_model import LinearRegression, Ridge, Lasso
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        from sklearn.model_selection import train_test_split

        namespace.update({
            'mean_absolute_error': mean_absolute_error,
            'mean_squared_error': mean_squared_error,
            'r2_score': r2_score,
            'RandomForestRegressor': RandomForestRegressor,
            'GradientBoostingRegressor': GradientBoostingRegressor,
            'LinearRegression': LinearRegression,
            'Ridge': Ridge,
            'Lasso': Lasso,
            'StandardScaler': StandardScaler,
            'MinMaxScaler': MinMaxScaler,
            'train_test_split': train_test_split,
        })
    except ImportError:
        pass

    try:
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        namespace['ARIMA'] = ARIMA
        namespace['ExponentialSmoothing'] = ExponentialSmoothing
    except ImportError:
        pass

    # Capture output
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = StringIO()
    sys.stderr = StringIO()

    start_time = time.time()
    success = True

    try:
        exec(code, namespace)
        output = sys.stdout.getvalue()
        stderr = sys.stderr.getvalue()
        if stderr:
            output += f"\n[STDERR]\n{stderr}"
    except Exception as e:
        success = False
        import traceback
        output = f"Error: {e}\n{traceback.format_exc()}"
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    exec_time = time.time() - start_time

    result = [
        f"Execution time: {exec_time:.2f}s",
        f"Status: {'SUCCESS' if success else 'FAILED'}",
        "",
        output
    ]

    return "\n".join(result)


@tool
def save_predictions(
    plan_id: str,
    predictions_code: str,
    metrics_json: str
) -> str:
    """
    Save predictions and results to output files.

    Args:
        plan_id: Plan ID
        predictions_code: Code that creates 'results_df' DataFrame with predictions
        metrics_json: JSON string with metrics

    Returns:
        Confirmation with saved paths
    """
    try:
        # Execute code to get results DataFrame
        namespace = {
            'pd': pd,
            'np': np,
            'Path': Path,
            'STAGE3B_OUT_DIR': STAGE3B_OUT_DIR,
            'STAGE4_OUT_DIR': STAGE4_OUT_DIR,
            'load_dataframe': load_dataframe,
        }

        exec(predictions_code, namespace)

        if 'results_df' not in namespace:
            return "Error: Code must create 'results_df' DataFrame"

        results_df = namespace['results_df']
        metrics = json.loads(metrics_json)

        # Save predictions parquet
        predictions_path = STAGE4_OUT_DIR / f"results_{plan_id}.parquet"
        results_df.to_parquet(predictions_path, index=False)

        # Save execution result
        result = {
            "plan_id": plan_id,
            "status": "success",
            "outputs": {
                "predictions": str(predictions_path)
            },
            "metrics": metrics,
            "summary": f"Generated predictions for {len(results_df)} samples",
            "data_shape": list(results_df.shape),
        }

        result_path = DataPassingManager.save_artifact(
            data=result,
            output_dir=STAGE4_OUT_DIR,
            filename=f"execution_result_{plan_id}.json",
            metadata={"stage": "stage4", "type": "execution_result"}
        )

        return f"Results saved:\n  Predictions: {predictions_path}\n  Metadata: {result_path}\n  Shape: {results_df.shape}"

    except json.JSONDecodeError as e:
        return f"Error: Invalid metrics JSON - {e}"
    except Exception as e:
        return f"Error saving predictions: {e}"


@tool
def save_execution_result(result_json: str) -> str:
    """
    Save execution result metadata.

    Args:
        result_json: JSON string with ExecutionResult structure

    Returns:
        Confirmation with saved path
    """
    try:
        result = json.loads(result_json)

        required = ['plan_id', 'status', 'summary']
        missing = [f for f in required if f not in result]
        if missing:
            return f"Error: Missing required fields: {missing}"

        plan_id = result['plan_id']
        filename = f"execution_result_{plan_id}.json"

        output_path = DataPassingManager.save_artifact(
            data=result,
            output_dir=STAGE4_OUT_DIR,
            filename=filename,
            metadata={"stage": "stage4", "type": "execution_result"}
        )

        return f"Execution result saved to: {output_path}"

    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON - {e}"
    except Exception as e:
        return f"Error saving result: {e}"


@tool
def list_stage4_results() -> str:
    """
    List existing Stage 4 execution results.

    Returns list of completed executions.
    """
    try:
        results = list(STAGE4_OUT_DIR.glob("execution_result_*.json"))

        if not results:
            return "No execution results found."

        output = ["=== Stage 4 Results ===\n"]

        for r in results:
            try:
                data = DataPassingManager.load_artifact(r)
                output.append(f"{r.name}:")
                output.append(f"  Status: {data.get('status')}")
                output.append(f"  Summary: {data.get('summary', 'N/A')[:100]}")
                if data.get('metrics'):
                    output.append(f"  Metrics: {data.get('metrics')}")
                output.append("")
            except:
                output.append(f"{r.name}: (error reading)")

        return "\n".join(output)

    except Exception as e:
        return f"Error listing results: {e}"


@tool
def verify_execution(plan_id: str) -> str:
    """
    Verify that execution completed successfully.

    Args:
        plan_id: Plan ID to verify

    Returns:
        Verification report
    """
    try:
        result_path = STAGE4_OUT_DIR / f"execution_result_{plan_id}.json"
        predictions_path = STAGE4_OUT_DIR / f"results_{plan_id}.parquet"

        result = [f"=== Execution Verification: {plan_id} ===\n"]

        # Check result file
        if result_path.exists():
            data = DataPassingManager.load_artifact(result_path)
            result.append(f"Result file: EXISTS")
            result.append(f"  Status: {data.get('status')}")
            result.append(f"  Metrics: {data.get('metrics', {})}")
        else:
            result.append("Result file: MISSING")
            return "\n".join(result) + "\n\nVERIFICATION: FAILED"

        # Check predictions file
        if predictions_path.exists():
            df = pd.read_parquet(predictions_path)
            result.append(f"\nPredictions file: EXISTS")
            result.append(f"  Shape: {df.shape}")
            result.append(f"  Columns: {list(df.columns)}")

            # Check for predicted column
            pred_cols = [c for c in df.columns if 'predict' in c.lower() or 'forecast' in c.lower()]
            if pred_cols:
                result.append(f"  Prediction columns: {pred_cols}")
            else:
                result.append("  WARNING: No prediction column found")
        else:
            result.append("\nPredictions file: MISSING")
            return "\n".join(result) + "\n\nVERIFICATION: FAILED"

        result.append("\nVERIFICATION: PASSED")
        return "\n".join(result)

    except Exception as e:
        return f"Error verifying execution: {e}"


# Export tools list
STAGE4_TOOLS = [
    load_execution_context,
    load_prepared_data,
    get_selected_method_code,
    execute_python_code,
    save_predictions,
    save_execution_result,
    list_stage4_results,
    verify_execution,
]
