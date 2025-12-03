"""
Helper module for tracking benchmark results separately from checkpoints.

This makes it easier to parse and validate results without complex JSON manipulation.
"""

from pathlib import Path
import json
from typing import Dict, List, Any
from datetime import datetime


def save_benchmark_result(
    plan_id: str,
    method_id: str,
    iteration: int,
    metrics: Dict[str, float],
    status: str = "success",
    error_message: str = None,
    output_dir: Path = None
) -> Path:
    """Save a single benchmark iteration result.
    
    Args:
        plan_id: Plan ID
        method_id: Method ID (e.g., "METHOD-1")
        iteration: Iteration number (1, 2, or 3)
        metrics: Dictionary of metrics (MAE, RMSE, etc.)
        status: "success" or "failure"
        error_message: Error message if failed
        output_dir: Output directory (defaults to STAGE3_5B_OUT_DIR)
    
    Returns:
        Path to saved result file
    """
    if output_dir is None:
        from .config import STAGE3_5B_OUT_DIR
        output_dir = STAGE3_5B_OUT_DIR
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result = {
        "plan_id": plan_id,
        "method_id": method_id,
        "iteration": iteration,
        "metrics": metrics,
        "status": status,
        "error_message": error_message,
        "timestamp": datetime.now().isoformat()
    }
    
    result_file = output_dir / f"result_{plan_id}_{method_id}_iter{iteration}.json"
    result_file.write_text(json.dumps(result, indent=2))
    
    return result_file


def load_method_results(plan_id: str, method_id: str, output_dir: Path = None) -> List[Dict[str, Any]]:
    """Load all iteration results for a method.
    
    Args:
        plan_id: Plan ID
        method_id: Method ID
        output_dir: Output directory
    
    Returns:
        List of result dictionaries
    """
    if output_dir is None:
        from .config import STAGE3_5B_OUT_DIR
        output_dir = STAGE3_5B_OUT_DIR
    
    results = []
    for i in range(1, 4):  # 3 iterations
        result_file = output_dir / f"result_{plan_id}_{method_id}_iter{i}.json"
        if result_file.exists():
            results.append(json.loads(result_file.read_text()))
    
    return results


def calculate_averaged_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate averaged metrics from multiple iterations.
    
    Args:
        results: List of result dictionaries
    
    Returns:
        Dictionary of averaged metrics
    """
    if not results:
        return {}
    
    # Collect all metric values
    metric_values = {}
    for result in results:
        if result.get("status") != "success":
            continue
        for metric_name, value in result.get("metrics", {}).items():
            if metric_name not in metric_values:
                metric_values[metric_name] = []
            metric_values[metric_name].append(value)
    
    # Calculate averages
    averaged = {}
    for metric_name, values in metric_values.items():
        if values:
            averaged[metric_name] = sum(values) / len(values)
    
    return averaged


def calculate_cv(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate coefficient of variation for each metric.
    
    Args:
        results: List of result dictionaries
    
    Returns:
        Dictionary of CV values for each metric
    """
    import numpy as np
    
    if not results:
        return {}
    
    # Collect all metric values
    metric_values = {}
    for result in results:
        if result.get("status") != "success":
            continue
        for metric_name, value in result.get("metrics", {}).items():
            if metric_name not in metric_values:
                metric_values[metric_name] = []
            metric_values[metric_name].append(value)
    
    # Calculate CV
    cv_dict = {}
    for metric_name, values in metric_values.items():
        if len(values) > 1:
            mean = np.mean(values)
            std = np.std(values)
            cv_dict[metric_name] = std / mean if mean != 0 else 0.0
        else:
            cv_dict[metric_name] = 0.0
    
    return cv_dict
