"""
Stage 3.5B Tools: Method Benchmarking

Tools for benchmarking proposed methods and selecting the best one.
"""

import json
import time
from pathlib import Path
from typing import Optional, Dict, Any
from langchain_core.tools import tool
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from code.config import (
    DATA_DIR, STAGE3_OUT_DIR, STAGE3B_OUT_DIR, STAGE3_5A_OUT_DIR, STAGE3_5B_OUT_DIR,
    DataPassingManager, BENCHMARK_ITERATIONS, MAX_CV_THRESHOLD, logger
)
from code.utils import load_dataframe, safe_json_dumps


# State tracking
_benchmark_results = {}
_stage3_5b_thoughts = []


@tool
def load_method_proposals(plan_id: str = None) -> str:
    """
    Load method proposals from Stage 3.5A.

    Args:
        plan_id: Plan ID. If not provided, loads most recent.

    Returns:
        Method proposals summary
    """
    try:
        if plan_id:
            proposal_path = STAGE3_5A_OUT_DIR / f"method_proposal_{plan_id}.json"
        else:
            proposals = list(STAGE3_5A_OUT_DIR.glob("method_proposal_*.json"))
            if not proposals:
                return "No method proposals found. Run Stage 3.5A first."
            proposal_path = max(proposals, key=lambda p: p.stat().st_mtime)

        proposal = DataPassingManager.load_artifact(proposal_path)

        result = [
            f"=== Method Proposals: {proposal.get('plan_id')} ===",
            f"Target: {proposal.get('target_column')}",
            f"Date: {proposal.get('date_column')}",
            "",
            "Proposed Methods:",
        ]

        for m in proposal.get('methods_proposed', []):
            result.append(f"\n{m.get('method_id')}: {m.get('name')}")
            result.append(f"  Category: {m.get('category')}")
            result.append(f"  Description: {m.get('description')}")
            result.append(f"  Libraries: {m.get('required_libraries', [])}")

        result.append("\n\nData Split Strategy:")
        split = proposal.get('data_split_strategy', {})
        result.append(f"  Type: {split.get('strategy_type')}")
        result.append(f"  Train: {split.get('train_period') or split.get('train_size')}")
        result.append(f"  Validation: {split.get('validation_period') or split.get('validation_size')}")
        result.append(f"  Test: {split.get('test_period') or split.get('test_size')}")

        return "\n".join(result)

    except Exception as e:
        return f"Error loading proposals: {e}"


@tool
def load_checkpoint(plan_id: str) -> str:
    """
    Load existing benchmark checkpoint to resume from.

    Args:
        plan_id: Plan ID

    Returns:
        Checkpoint status and completed methods
    """
    try:
        checkpoint_path = STAGE3_5B_OUT_DIR / f"checkpoint_{plan_id}.json"

        if not checkpoint_path.exists():
            return f"No checkpoint found for {plan_id}. Starting fresh."

        checkpoint = DataPassingManager.load_artifact(checkpoint_path)

        global _benchmark_results
        _benchmark_results = checkpoint.get('completed_results', {})

        completed = checkpoint.get('methods_completed', [])

        result = [
            f"=== Checkpoint Loaded: {plan_id} ===",
            f"Methods completed: {completed}",
            "",
            "Results so far:",
        ]

        for method_id, results in _benchmark_results.items():
            avg_mae = np.mean([r.get('mae', 0) for r in results.get('iterations', [])])
            result.append(f"  {method_id}: Avg MAE = {avg_mae:.4f}")

        return "\n".join(result)

    except Exception as e:
        return f"Error loading checkpoint: {e}"


@tool
def save_checkpoint(plan_id: str, methods_completed: str, results_json: str) -> str:
    """
    Save benchmark checkpoint for resume capability.

    Args:
        plan_id: Plan ID
        methods_completed: Comma-separated list of completed method IDs
        results_json: JSON string with completed results

    Returns:
        Confirmation
    """
    try:
        results = json.loads(results_json)

        checkpoint = {
            "plan_id": plan_id,
            "methods_completed": [m.strip() for m in methods_completed.split(',')],
            "completed_results": results
        }

        output_path = DataPassingManager.save_artifact(
            data=checkpoint,
            output_dir=STAGE3_5B_OUT_DIR,
            filename=f"checkpoint_{plan_id}.json",
            metadata={"stage": "stage3_5b", "type": "checkpoint"}
        )

        return f"Checkpoint saved: {output_path}"

    except Exception as e:
        return f"Error saving checkpoint: {e}"


@tool
def record_thought_3_5b(thought: str, next_action: str) -> str:
    """Record a thought before action (ReAct)."""
    global _stage3_5b_thoughts
    entry = {"thought": thought, "next_action": next_action, "step": len(_stage3_5b_thoughts) + 1}
    _stage3_5b_thoughts.append(entry)
    return f"Thought #{entry['step']} recorded."


@tool
def run_benchmark_code(code: str, method_name: str) -> str:
    """
    Execute benchmarking code for a method.

    The code should:
    1. Load the prepared data
    2. Split into train/validation/test
    3. Train and predict
    4. Calculate metrics (MAE, RMSE, MAPE)
    5. Print results as JSON

    Args:
        code: Python code to execute
        method_name: Name of the method being tested

    Returns:
        Execution output with metrics
    """
    import sys
    from io import StringIO

    # Prepare namespace with necessary imports
    namespace = {
        'pd': pd,
        'np': np,
        'json': json,
        'Path': Path,
        'DATA_DIR': DATA_DIR,
        'STAGE3B_OUT_DIR': STAGE3B_OUT_DIR,
        'STAGE3_OUT_DIR': STAGE3_OUT_DIR,
        'time': time,
    }

    # Import common ML libraries
    try:
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression
        namespace['mean_absolute_error'] = mean_absolute_error
        namespace['mean_squared_error'] = mean_squared_error
        namespace['RandomForestRegressor'] = RandomForestRegressor
        namespace['LinearRegression'] = LinearRegression
    except ImportError:
        pass

    try:
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        namespace['ARIMA'] = ARIMA
        namespace['ExponentialSmoothing'] = ExponentialSmoothing
    except ImportError:
        pass

    # Add load_dataframe function
    namespace['load_dataframe'] = load_dataframe

    # Capture output
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = StringIO()
    sys.stderr = StringIO()

    start_time = time.time()
    success = True
    output = ""

    try:
        exec(code, namespace)
        output = sys.stdout.getvalue()
        stderr = sys.stderr.getvalue()
        if stderr:
            output += f"\n[STDERR]\n{stderr}"
    except Exception as e:
        success = False
        import traceback
        output = f"Error executing {method_name}: {e}\n{traceback.format_exc()}"
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    execution_time = time.time() - start_time

    result = [
        f"=== Benchmark: {method_name} ===",
        f"Execution time: {execution_time:.2f}s",
        f"Status: {'SUCCESS' if success else 'FAILED'}",
        "",
        "Output:",
        output
    ]

    return "\n".join(result)


@tool
def calculate_metrics(actual: str, predicted: str) -> str:
    """
    Calculate forecasting metrics from actual vs predicted values.

    Args:
        actual: Comma-separated actual values
        predicted: Comma-separated predicted values

    Returns:
        Metrics (MAE, RMSE, MAPE)
    """
    try:
        actual_vals = [float(x.strip()) for x in actual.split(',')]
        predicted_vals = [float(x.strip()) for x in predicted.split(',')]

        if len(actual_vals) != len(predicted_vals):
            return f"Error: Length mismatch ({len(actual_vals)} vs {len(predicted_vals)})"

        actual_arr = np.array(actual_vals)
        predicted_arr = np.array(predicted_vals)

        mae = np.mean(np.abs(actual_arr - predicted_arr))
        rmse = np.sqrt(np.mean((actual_arr - predicted_arr) ** 2))

        # MAPE (handle zeros)
        mask = actual_arr != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((actual_arr[mask] - predicted_arr[mask]) / actual_arr[mask])) * 100
        else:
            mape = float('inf')

        metrics = {
            "mae": round(mae, 6),
            "rmse": round(rmse, 6),
            "mape": round(mape, 4),
            "n_samples": len(actual_vals)
        }

        return json.dumps(metrics, indent=2)

    except Exception as e:
        return f"Error calculating metrics: {e}"


@tool
def validate_consistency(method_id: str, iteration_metrics: str) -> str:
    """
    Validate that benchmark results are consistent (not hallucinated).

    Checks coefficient of variation across iterations.

    Args:
        method_id: Method being validated
        iteration_metrics: JSON string with list of iteration metrics

    Returns:
        Validation result
    """
    try:
        metrics_list = json.loads(iteration_metrics)

        if len(metrics_list) < 2:
            return f"Need at least 2 iterations for consistency check. Got {len(metrics_list)}."

        result = [f"=== Consistency Check: {method_id} ===\n"]

        # Check CV for each metric
        for metric_name in ['mae', 'rmse', 'mape']:
            values = [m.get(metric_name, 0) for m in metrics_list if m.get(metric_name) is not None]

            if not values or all(v == 0 for v in values):
                result.append(f"{metric_name}: No valid values")
                continue

            mean_val = np.mean(values)
            std_val = np.std(values)
            cv = std_val / mean_val if mean_val != 0 else 0

            status = "VALID" if cv < MAX_CV_THRESHOLD else "SUSPICIOUS"
            result.append(f"{metric_name}: CV={cv:.4f} ({status})")
            result.append(f"  Values: {values}")
            result.append(f"  Mean: {mean_val:.6f}, Std: {std_val:.6f}")

        # Overall assessment
        mae_values = [m.get('mae', 0) for m in metrics_list]
        overall_cv = np.std(mae_values) / np.mean(mae_values) if np.mean(mae_values) != 0 else 0

        if overall_cv < MAX_CV_THRESHOLD:
            result.append(f"\nOVERALL: VALID (CV={overall_cv:.4f} < {MAX_CV_THRESHOLD})")
        else:
            result.append(f"\nOVERALL: SUSPICIOUS - Results may be hallucinated (CV={overall_cv:.4f})")
            result.append("Consider re-running benchmarks or checking implementation")

        return "\n".join(result)

    except Exception as e:
        return f"Error validating consistency: {e}"


@tool
def select_best_method(results_json: str) -> str:
    """
    Select the best method based on benchmark results.

    Uses average MAE as primary selection criterion.

    Args:
        results_json: JSON with all benchmark results

    Returns:
        Selection with rationale
    """
    try:
        results = json.loads(results_json)

        result = ["=== Method Selection ===\n"]
        method_scores = []

        for method_id, data in results.items():
            iterations = data.get('iterations', [])
            if not iterations:
                result.append(f"{method_id}: No valid iterations")
                continue

            # Calculate average metrics
            avg_mae = np.mean([it.get('mae', float('inf')) for it in iterations])
            avg_rmse = np.mean([it.get('rmse', float('inf')) for it in iterations])

            is_valid = data.get('is_valid', True)

            result.append(f"{method_id} ({data.get('name', 'Unknown')}):")
            result.append(f"  Avg MAE: {avg_mae:.6f}")
            result.append(f"  Avg RMSE: {avg_rmse:.6f}")
            result.append(f"  Valid: {is_valid}")

            if is_valid:
                method_scores.append({
                    'method_id': method_id,
                    'name': data.get('name', method_id),
                    'avg_mae': avg_mae,
                    'avg_rmse': avg_rmse
                })

        result.append("")

        if not method_scores:
            result.append("ERROR: No valid methods found!")
            return "\n".join(result)

        # Select best by MAE
        best = min(method_scores, key=lambda x: x['avg_mae'])

        result.append(f"SELECTED: {best['method_id']} ({best['name']})")
        result.append(f"  Reason: Lowest average MAE ({best['avg_mae']:.6f})")

        # Comparison summary
        result.append("\nComparison Summary:")
        for m in sorted(method_scores, key=lambda x: x['avg_mae']):
            pct_worse = ((m['avg_mae'] - best['avg_mae']) / best['avg_mae'] * 100) if best['avg_mae'] > 0 else 0
            if m['method_id'] == best['method_id']:
                result.append(f"  {m['method_id']}: {m['avg_mae']:.6f} (BEST)")
            else:
                result.append(f"  {m['method_id']}: {m['avg_mae']:.6f} (+{pct_worse:.1f}%)")

        return "\n".join(result)

    except Exception as e:
        return f"Error selecting method: {e}"


@tool
def save_tester_output(output_json: str) -> str:
    """
    Save the final tester output with benchmark results.

    Args:
        output_json: JSON string with TesterOutput structure

    Returns:
        Confirmation with saved path
    """
    try:
        output = json.loads(output_json)

        # Validate structure
        required = ['plan_id', 'methods_tested', 'selected_method_id', 'selection_rationale']
        missing = [f for f in required if f not in output]
        if missing:
            return f"Error: Missing required fields: {missing}"

        plan_id = output['plan_id']
        filename = f"tester_{plan_id}.json"

        output_path = DataPassingManager.save_artifact(
            data=output,
            output_dir=STAGE3_5B_OUT_DIR,
            filename=filename,
            metadata={"stage": "stage3_5b", "type": "tester_output"}
        )

        return f"Tester output saved to: {output_path}"

    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON - {e}"
    except Exception as e:
        return f"Error saving output: {e}"


def reset_benchmark_state():
    """Reset benchmark state."""
    global _benchmark_results, _stage3_5b_thoughts
    _benchmark_results = {}
    _stage3_5b_thoughts = []


# Export tools list
STAGE3_5B_TOOLS = [
    load_method_proposals,
    load_checkpoint,
    save_checkpoint,
    record_thought_3_5b,
    run_benchmark_code,
    calculate_metrics,
    validate_consistency,
    select_best_method,
    save_tester_output,
]
