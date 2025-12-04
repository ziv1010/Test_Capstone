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
    DataPassingManager, BENCHMARK_ITERATIONS, MAX_CV_THRESHOLD, logger, DEBUG
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

        raw_data = DataPassingManager.load_artifact(checkpoint_path)
        
        # Handle wrapped data structure from DataPassingManager
        if isinstance(raw_data, dict) and 'data' in raw_data:
            checkpoint = raw_data['data']
        else:
            checkpoint = raw_data
        
        # Validate checkpoint is a dict
        if not isinstance(checkpoint, dict):
            logger.warning(f"Invalid checkpoint structure: {type(checkpoint)}")
            return f"Invalid checkpoint format for {plan_id}. Starting fresh."

        global _benchmark_results
        completed_results = checkpoint.get('completed_results', {})
        
        # Handle the case where completed_results is a single method object
        # instead of a dict keyed by method_id
        if isinstance(completed_results, dict) and 'method_id' in completed_results:
            # Convert single result to keyed format
            method_id = completed_results.get('method_id')
            _benchmark_results = {method_id: completed_results}
        elif isinstance(completed_results, dict):
            _benchmark_results = completed_results
        else:
            _benchmark_results = {}

        completed = checkpoint.get('methods_completed', [])

        result = [
            f"=== Checkpoint Loaded: {plan_id} ===",
            f"Methods completed: {completed}",
            "",
            "Results so far:",
        ]

        for method_id, data in _benchmark_results.items():
            # Handle different result formats
            if isinstance(data, dict):
                if 'iterations' in data:
                    avg_mae = np.mean([r.get('mae', 0) for r in data.get('iterations', [])])
                elif 'avg_mae' in data:
                    avg_mae = data.get('avg_mae', 0)
                else:
                    avg_mae = data.get('mae', 0)
                result.append(f"  {method_id}: Avg MAE = {avg_mae:.4f}")
            else:
                result.append(f"  {method_id}: (invalid result format)")

        return "\n".join(result)

    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        import traceback
        logger.error(traceback.format_exc())
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
        if isinstance(results_json, dict):
            results = results_json
        else:
            # Clean up the JSON string
            cleaned_json = str(results_json).strip()
            if cleaned_json.startswith("```json"):
                cleaned_json = cleaned_json[7:]
            if cleaned_json.startswith("```"):
                cleaned_json = cleaned_json[3:]
            if cleaned_json.endswith("```"):
                cleaned_json = cleaned_json[:-3]
            cleaned_json = cleaned_json.strip()

            results = json.loads(cleaned_json)
            # Handle double-encoded JSON
            if isinstance(results, str):
                if DEBUG:
                    logger.debug("Detected double-encoded JSON string, parsing again...")
                results = json.loads(results)

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
    
    Accepts multiple formats:
    1. Dict with method_id keys and 'iterations' array: {"M1": {"iterations": [{"mae": ...}], ...}}
    2. Dict with method_id keys and direct 'avg_mae': {"M1": {"avg_mae": ..., "valid": true}, ...}
    3. List of method objects: [{"method_id": "M1", "avg_mae": ..., "valid": true}, ...]

    Args:
        results_json: JSON with all benchmark results

    Returns:
        Selection with rationale
    """
    try:
        results = json.loads(results_json)

        result = ["=== Method Selection ===\n"]
        method_scores = []

        # Handle list format: [{"method_id": "M1", "avg_mae": ..., ...}, ...]
        if isinstance(results, list):
            for item in results:
                method_id = item.get('method_id', item.get('id', 'Unknown'))
                name = item.get('name', method_id)
                avg_mae = item.get('avg_mae', item.get('mae', float('inf')))
                avg_rmse = item.get('avg_rmse', item.get('rmse', float('inf')))
                is_valid = item.get('valid', item.get('is_valid', True))

                result.append(f"{method_id} ({name}):")
                result.append(f"  Avg MAE: {avg_mae:.6f}")
                result.append(f"  Avg RMSE: {avg_rmse:.6f}" if avg_rmse != float('inf') else "  Avg RMSE: N/A")
                result.append(f"  Valid: {is_valid}")

                if is_valid:
                    method_scores.append({
                        'method_id': method_id,
                        'name': name,
                        'avg_mae': float(avg_mae),
                        'avg_rmse': float(avg_rmse) if avg_rmse != float('inf') else float('inf')
                    })
        else:
            # Handle dict formats
            for method_id, data in results.items():
                iterations = data.get('iterations', [])
                
                # Check if direct avg_mae is provided (new format)
                if not iterations and 'avg_mae' in data:
                    avg_mae = float(data.get('avg_mae', float('inf')))
                    avg_rmse = float(data.get('avg_rmse', data.get('rmse', float('inf'))))
                    is_valid = data.get('valid', data.get('is_valid', True))
                elif iterations:
                    # Calculate from iterations (old format)
                    avg_mae = np.mean([it.get('mae', float('inf')) for it in iterations])
                    avg_rmse = np.mean([it.get('rmse', float('inf')) for it in iterations])
                    is_valid = data.get('is_valid', True)
                else:
                    result.append(f"{method_id}: No valid data (missing iterations or avg_mae)")
                    continue

                name = data.get('name', method_id)

                result.append(f"{method_id} ({name}):")
                result.append(f"  Avg MAE: {avg_mae:.6f}")
                result.append(f"  Avg RMSE: {avg_rmse:.6f}" if avg_rmse != float('inf') else "  Avg RMSE: N/A")
                result.append(f"  Valid: {is_valid}")

                if is_valid:
                    method_scores.append({
                        'method_id': method_id,
                        'name': name,
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
        import traceback
        logger.error(f"Error in select_best_method: {e}\n{traceback.format_exc()}")
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
        if DEBUG:
            logger.debug(f"Saving tester output. Input type: {type(output_json)}")
            logger.debug(f"Input preview: {str(output_json)[:500]}...")

        if isinstance(output_json, dict):
            output = output_json
        else:
            # Clean up the JSON string
            cleaned_json = str(output_json).strip()
            if cleaned_json.startswith("```json"):
                cleaned_json = cleaned_json[7:]
            if cleaned_json.startswith("```"):
                cleaned_json = cleaned_json[3:]
            if cleaned_json.endswith("```"):
                cleaned_json = cleaned_json[:-3]
            cleaned_json = cleaned_json.strip()

            output = json.loads(cleaned_json)
            # Handle double-encoded JSON
            if isinstance(output, str):
                if DEBUG:
                    logger.debug("Detected double-encoded JSON string, parsing again...")
                output = json.loads(output)

        # Validate structure - be more lenient
        required = ['plan_id', 'methods_tested', 'selected_method_id']
        missing = [f for f in required if f not in output]
        if missing:
            return f"Error: Missing required fields: {missing}"

        # Add default selection_rationale if missing
        if 'selection_rationale' not in output:
            output['selection_rationale'] = "Selected based on best performance metrics"
            logger.warning("Added default selection_rationale")

        # Ensure selected_method_name exists
        if 'selected_method_name' not in output:
            # Try to find the name from methods_tested
            for method in output.get('methods_tested', []):
                if method.get('method_id') == output['selected_method_id']:
                    output['selected_method_name'] = method.get('method_name', method.get('name', output['selected_method_id']))
                    break
            else:
                output['selected_method_name'] = output['selected_method_id']

        plan_id = output['plan_id']
        filename = f"tester_{plan_id}.json"

        # ============================================================
        # CRITICAL: Store winning method's implementation code
        # This ensures Stage 4 can execute the exact same code/algorithm
        # ============================================================
        if 'winning_method_code' not in output:
            try:
                proposal_path = STAGE3_5A_OUT_DIR / f"method_proposal_{plan_id}.json"
                if proposal_path.exists():
                    proposal = DataPassingManager.load_artifact(proposal_path)
                    selected_id = output['selected_method_id']

                    # Find and store winning method's code
                    for method in proposal.get('methods_proposed', []):
                        if method.get('method_id') == selected_id:
                            output['winning_method_code'] = method.get('implementation_code', '')
                            output['winning_method_hyperparameters'] = method.get('hyperparameters', {})
                            output['winning_method_libraries'] = method.get('required_libraries', [])
                            logger.info(f"Stored winning method code for {selected_id}")
                            break

                    # Also store data split strategy and column info for consistency
                    output['data_split_strategy'] = proposal.get('data_split_strategy', {})
                    output['target_column'] = proposal.get('target_column')
                    output['date_column'] = proposal.get('date_column')
                    output['feature_columns'] = proposal.get('feature_columns', [])
                else:
                    logger.warning(f"Method proposal not found at {proposal_path}, cannot store winning code")
            except Exception as e:
                logger.error(f"Failed to store winning method code: {e}")

        # ============================================================
        # CRITICAL: Store benchmark metrics from winning method
        # Stage 4 should replicate these metrics
        # ============================================================
        selected_id = output.get('selected_method_id')
        for method in output.get('methods_tested', []):
            if method.get('method_id') == selected_id:
                output['benchmark_metrics'] = {
                    'mae': method.get('avg_mae', method.get('mae')),
                    'rmse': method.get('avg_rmse', method.get('rmse')),
                    'mape': method.get('avg_mape', method.get('mape')),
                }
                logger.info(f"Stored benchmark metrics for Stage 4 verification: {output['benchmark_metrics']}")
                break

        # Ensure output directory exists
        STAGE3_5B_OUT_DIR.mkdir(parents=True, exist_ok=True)

        output_path = DataPassingManager.save_artifact(
            data=output,
            output_dir=STAGE3_5B_OUT_DIR,
            filename=filename,
            metadata={"stage": "stage3_5b", "type": "tester_output"}
        )

        # Verify the file was actually saved
        if output_path.exists():
            logger.info(f"Verified tester output exists at: {output_path}")
            return f"SUCCESS: Tester output saved to: {output_path}"
        else:
            # Fallback: direct write
            logger.warning("DataPassingManager save may have failed, attempting direct write...")
            with open(STAGE3_5B_OUT_DIR / filename, 'w') as f:
                json.dump(output, f, indent=2, default=str)
            return f"SUCCESS (fallback): Tester output saved to: {STAGE3_5B_OUT_DIR / filename}"

    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        logger.error(f"Input was: {str(output_json)[:1000]}")
        return f"Error: Invalid JSON - {e}"
    except Exception as e:
        logger.error(f"Error saving tester output: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return f"Error saving output: {e}"


def reset_benchmark_state():
    """Reset benchmark state."""
    global _benchmark_results, _stage3_5b_thoughts
    _benchmark_results = {}
    _stage3_5b_thoughts = []


@tool
def finish_benchmarking() -> str:
    """
    Signal that benchmarking is complete.
    
    Call this ONLY after save_tester_output returns success.
    
    Returns:
        Completion message
    """
    return "Stage 3.5B Complete. You may now stop."


@tool
def get_prepared_data_info(plan_id: str = None) -> str:
    """
    Get information about the prepared data for benchmarking.

    Args:
        plan_id: Plan ID

    Returns:
        Data summary including shape, columns, and sample
    """
    try:
        if not plan_id:
            plans = list(STAGE3_OUT_DIR.glob("PLAN-*.json"))
            if plans:
                plan_id = max(plans, key=lambda p: p.stat().st_mtime).stem

        prepared_path = STAGE3B_OUT_DIR / f"prepared_{plan_id}.parquet"
        if not prepared_path.exists():
            return f"Prepared data not found at: {prepared_path}"

        df = pd.read_parquet(prepared_path)

        # Load plan for column info
        plan_path = STAGE3_OUT_DIR / f"{plan_id}.json"
        plan = {}
        if plan_path.exists():
            plan = DataPassingManager.load_artifact(plan_path)

        result = [
            f"=== Prepared Data Info: {plan_id} ===",
            f"Shape: {df.shape[0]} rows x {df.shape[1]} columns",
            f"Target Column: {plan.get('target_column', 'N/A')}",
            f"Date Column: {plan.get('date_column', 'N/A')}",
            "",
            "Columns:",
        ]

        for col in df.columns:
            dtype = df[col].dtype
            nulls = df[col].isna().sum()
            result.append(f"  {col}: {dtype} ({nulls} nulls)")

        result.append(f"\nSample (first 5 rows):")
        result.append(df.head().to_string())

        return "\n".join(result)

    except Exception as e:
        return f"Error getting data info: {e}"


@tool
def get_method_implementation(plan_id: str, method_id: str) -> str:
    """
    Get the implementation code for a specific method.

    Args:
        plan_id: Plan ID
        method_id: Method ID (M1, M2, or M3)

    Returns:
        Method implementation details and code
    """
    try:
        proposal_path = STAGE3_5A_OUT_DIR / f"method_proposal_{plan_id}.json"
        if not proposal_path.exists():
            return f"Method proposals not found for {plan_id}"

        proposal = DataPassingManager.load_artifact(proposal_path)

        for method in proposal.get('methods_proposed', []):
            if method.get('method_id') == method_id:
                result = [
                    f"=== Method: {method_id} - {method.get('name')} ===",
                    f"Category: {method.get('category')}",
                    f"Description: {method.get('description')}",
                    f"Libraries: {method.get('required_libraries', [])}",
                    "",
                    "Hyperparameters:",
                    json.dumps(method.get('hyperparameters', {}), indent=2),
                    "",
                    "Implementation Code:",
                    "```python",
                    method.get('implementation_code', '# No code'),
                    "```",
                ]
                return "\n".join(result)

        return f"Method {method_id} not found in proposals"

    except Exception as e:
        return f"Error getting method: {e}"


@tool
def test_single_method(plan_id: str, method_id: str) -> str:
    """
    Test a single method's implementation with the prepared data.

    Args:
        plan_id: Plan ID
        method_id: Method ID to test (M1, M2, or M3)

    Returns:
        Test results with metrics
    """
    import sys
    from io import StringIO

    try:
        # Load method proposal
        proposal_path = STAGE3_5A_OUT_DIR / f"method_proposal_{plan_id}.json"
        if not proposal_path.exists():
            return f"Method proposals not found for {plan_id}"

        proposal = DataPassingManager.load_artifact(proposal_path)

        # Find the method
        method = None
        for m in proposal.get('methods_proposed', []):
            if m.get('method_id') == method_id:
                method = m
                break

        if not method:
            return f"Method {method_id} not found"

        # Load prepared data
        prepared_path = STAGE3B_OUT_DIR / f"prepared_{plan_id}.parquet"
        if not prepared_path.exists():
            return f"Prepared data not found"

        df = pd.read_parquet(prepared_path)
        target_col = proposal.get('target_column', 'target')
        date_col = proposal.get('date_column', 'date')

        # Parse dates if needed
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.sort_values(date_col)

        # Split data
        train_size = int(len(df) * 0.7)
        val_size = int(len(df) * 0.15)

        train_df = df.iloc[:train_size].copy()
        test_df = df.iloc[train_size+val_size:].copy()

        # Setup namespace
        namespace = {
            'pd': pd,
            'np': np,
            'train_df': train_df,
            'test_df': test_df,
            'target_col': target_col,
            'date_col': date_col,
        }

        # Add ML imports
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.linear_model import LinearRegression
            namespace['RandomForestRegressor'] = RandomForestRegressor
            namespace['LinearRegression'] = LinearRegression
        except ImportError:
            pass

        try:
            from statsmodels.tsa.arima.model import ARIMA
            namespace['ARIMA'] = ARIMA
        except ImportError:
            pass

        # Execute method code
        impl_code = method.get('implementation_code', '')

        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            exec(impl_code, namespace)
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        # Find the predict function
        predict_funcs = [k for k, v in namespace.items()
                        if callable(v) and (k.startswith('predict_') or 'method' in k.lower())]

        result = [
            f"=== Test: {method_id} - {method.get('name')} ===",
            f"Code executed successfully",
            f"Output: {output[:200] if output else '(none)'}",
            f"Functions found: {predict_funcs}",
        ]

        if predict_funcs:
            func = namespace[predict_funcs[0]]
            try:
                predictions = func(train_df, test_df, target_col, date_col)

                # Calculate metrics
                actual = test_df[target_col].values
                if hasattr(predictions, 'values'):
                    pred = predictions['predicted'].values if 'predicted' in predictions.columns else predictions.values.flatten()
                else:
                    pred = np.array(predictions)

                mae = np.mean(np.abs(actual - pred))
                rmse = np.sqrt(np.mean((actual - pred) ** 2))
                mask = actual != 0
                mape = np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100 if mask.sum() > 0 else float('inf')

                result.append(f"\nMetrics:")
                result.append(f"  MAE: {mae:.4f}")
                result.append(f"  RMSE: {rmse:.4f}")
                result.append(f"  MAPE: {mape:.2f}%")
                result.append(f"  Predictions: {len(pred)} values")

            except Exception as e:
                result.append(f"\nPrediction failed: {e}")
                import traceback
                result.append(traceback.format_exc()[:500])

        return "\n".join(result)

    except Exception as e:
        import traceback
        return f"Error testing method: {e}\n{traceback.format_exc()}"


@tool
def get_react_summary_3_5b() -> str:
    """
    Get a summary of all recorded thoughts in Stage 3.5B.

    Returns:
        Summary of ReAct reasoning trail
    """
    global _stage3_5b_thoughts

    result = ["=== ReAct Summary (Stage 3.5B) ===\n"]

    if _stage3_5b_thoughts:
        result.append("Thoughts:")
        for t in _stage3_5b_thoughts:
            result.append(f"  {t['step']}. {t['thought'][:100]}...")
            result.append(f"     Next: {t['next_action'][:50]}...")
    else:
        result.append("No thoughts recorded yet.")

    result.append(f"\nBenchmark results in memory: {len(_benchmark_results)} methods")
    for method_id, data in _benchmark_results.items():
        result.append(f"  {method_id}: {len(data.get('iterations', []))} iterations")

    return "\n".join(result)


@tool
def debug_benchmark_state(plan_id: str) -> str:
    """
    Debug the current benchmark state and check for issues.

    Args:
        plan_id: Plan ID to debug

    Returns:
        Comprehensive debug information
    """
    try:
        result = ["=== Benchmark Debug State ===\n"]

        # Check method proposals
        proposal_path = STAGE3_5A_OUT_DIR / f"method_proposal_{plan_id}.json"
        result.append(f"Method Proposals: {'EXISTS' if proposal_path.exists() else 'MISSING'}")

        if proposal_path.exists():
            proposal = DataPassingManager.load_artifact(proposal_path)
            methods = proposal.get('methods_proposed', [])
            result.append(f"  Methods: {[m.get('method_id') for m in methods]}")

            for m in methods:
                code = m.get('implementation_code', '')
                result.append(f"  {m.get('method_id')}: {len(code)} chars of code")

        # Check prepared data
        prepared_path = STAGE3B_OUT_DIR / f"prepared_{plan_id}.parquet"
        result.append(f"\nPrepared Data: {'EXISTS' if prepared_path.exists() else 'MISSING'}")

        if prepared_path.exists():
            df = pd.read_parquet(prepared_path)
            result.append(f"  Shape: {df.shape}")
            result.append(f"  Nulls: {df.isnull().sum().sum()}")

        # Check checkpoint
        checkpoint_path = STAGE3_5B_OUT_DIR / f"checkpoint_{plan_id}.json"
        result.append(f"\nCheckpoint: {'EXISTS' if checkpoint_path.exists() else 'NONE'}")

        if checkpoint_path.exists():
            checkpoint = DataPassingManager.load_artifact(checkpoint_path)
            result.append(f"  Completed: {checkpoint.get('methods_completed', [])}")

        # Check final output
        tester_path = STAGE3_5B_OUT_DIR / f"tester_{plan_id}.json"
        result.append(f"\nTester Output: {'EXISTS' if tester_path.exists() else 'MISSING'}")

        if tester_path.exists():
            tester = DataPassingManager.load_artifact(tester_path)
            result.append(f"  Selected: {tester.get('selected_method_id')}")
            result.append(f"  Methods tested: {len(tester.get('methods_tested', []))}")

        # In-memory state
        result.append(f"\nIn-memory benchmark results: {len(_benchmark_results)}")
        for mid, data in _benchmark_results.items():
            result.append(f"  {mid}: {len(data.get('iterations', []))} iterations")

        return "\n".join(result)

    except Exception as e:
        return f"Error debugging benchmark: {e}"


@tool
def get_actual_columns(plan_id: str = None) -> str:
    """
    Get the ACTUAL column names from the prepared data.
    
    CRITICAL: Use this to prevent column hallucination. Only use columns
    that are returned by this tool - do not assume or invent column names.
    
    Args:
        plan_id: Plan ID to check
    
    Returns:
        List of actual columns with their data types
    """
    try:
        if not plan_id:
            plans = list(STAGE3_OUT_DIR.glob("PLAN-*.json"))
            if plans:
                plan_id = max(plans, key=lambda p: p.stat().st_mtime).stem
        
        # Load prepared data
        prepared_path = STAGE3B_OUT_DIR / f"prepared_{plan_id}.parquet"
        if not prepared_path.exists():
            return f"ERROR: Prepared data not found at {prepared_path}"
        
        df = pd.read_parquet(prepared_path)
        
        # Load plan to show what was expected vs actual
        plan_path = STAGE3_OUT_DIR / f"{plan_id}.json"
        plan = DataPassingManager.load_artifact(plan_path) if plan_path.exists() else {}
        
        result = [
            f"=== ACTUAL COLUMNS in prepared_{plan_id}.parquet ===",
            f"Total columns: {len(df.columns)}",
            f"Data shape: {df.shape}",
            "",
            "Column Name | Data Type",
            "-" * 40,
        ]
        
        for col in df.columns:
            result.append(f"{col} | {df[col].dtype}")
        
        result.append("")
        result.append("=== Plan Expectations vs Reality ===")
        
        expected_date = plan.get('date_column')
        expected_target = plan.get('target_column')
        
        if expected_date:
            status = "✓ EXISTS" if expected_date in df.columns else "✗ MISSING"
            result.append(f"Expected date_column: {expected_date} ... {status}")
            if expected_date not in df.columns:
                result.append(f"  WARNING: Use df.index or set date_col=None in your benchmark code!")
        
        if expected_target:
            status = "✓ EXISTS" if expected_target in df.columns else "✗ MISSING"
            result.append(f"Expected target_column: {expected_target} ... {status}")
        
        result.append("")
        result.append("⚠️  CRITICAL: Use ONLY the columns listed above!")
        result.append("⚠️  Do NOT assume or invent column names like 'Year', 'date', etc.")
        
        return "\n".join(result)
        
    except Exception as e:
        return f"Error getting actual columns: {e}"


# Export tools list
STAGE3_5B_TOOLS = [
    load_method_proposals,
    get_actual_columns,  # NEW: Prevent column hallucination
    load_checkpoint,
    save_checkpoint,
    record_thought_3_5b,
    run_benchmark_code,
    calculate_metrics,
    validate_consistency,
    select_best_method,
    save_tester_output,
    finish_benchmarking,
    # New debugging tools
    get_prepared_data_info,
    get_method_implementation,
    test_single_method,
    get_react_summary_3_5b,
    debug_benchmark_state,
]
