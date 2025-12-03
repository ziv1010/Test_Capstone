"""
Stage 3B Tools: Data Preparation

Tools for preparing data according to the execution plan.
"""

import json
from pathlib import Path
from typing import Optional
from langchain_core.tools import tool
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from code.config import (
    DATA_DIR, STAGE3_OUT_DIR, STAGE3B_OUT_DIR,
    DataPassingManager, logger
)
from code.utils import load_dataframe, execute_python_sandbox, safe_json_dumps


# State tracking for ReAct framework
_stage3b_thoughts = []
_stage3b_observations = []


@tool
def load_execution_plan(plan_id: str = None) -> str:
    """
    Load an execution plan from Stage 3.

    Args:
        plan_id: Plan ID (e.g., 'PLAN-TSK-001'). If not provided, loads the most recent.

    Returns:
        Execution plan as formatted string
    """
    try:
        if plan_id:
            plan_path = STAGE3_OUT_DIR / f"{plan_id}.json"
        else:
            # Find most recent plan
            plans = list(STAGE3_OUT_DIR.glob("PLAN-*.json"))
            if not plans:
                return "No execution plans found. Run Stage 3 first."
            plan_path = max(plans, key=lambda p: p.stat().st_mtime)

        plan = DataPassingManager.load_artifact(plan_path)

        result = [
            f"=== Execution Plan: {plan.get('plan_id')} ===",
            f"Task: {plan.get('selected_task_id')}",
            f"Goal: {plan.get('goal')}",
            f"Category: {plan.get('task_category')}",
            "",
            f"Target Column: {plan.get('target_column')}",
            f"Date Column: {plan.get('date_column')}",
            "",
            "File Instructions:",
        ]

        for fi in plan.get('file_instructions', []):
            result.append(f"  - {fi.get('filename')}")
            if fi.get('columns_to_use'):
                result.append(f"    Columns: {fi.get('columns_to_use')}")
            if fi.get('filters'):
                result.append(f"    Filters: {fi.get('filters')}")

        if plan.get('join_steps'):
            result.append("\nJoin Steps:")
            for js in plan['join_steps']:
                result.append(
                    f"  - {js.get('left_dataset')} JOIN {js.get('right_dataset')} "
                    f"ON {js.get('left_on')}={js.get('right_on')}"
                )

        if plan.get('feature_engineering'):
            result.append(f"\nFeatures to create: {len(plan['feature_engineering'])}")

        result.append("\n\nFull JSON:")
        result.append(json.dumps(plan, indent=2, default=str))

        return "\n".join(result)

    except Exception as e:
        return f"Error loading plan: {e}"


@tool
def record_thought(thought: str, next_action: str) -> str:
    """
    Record a thought before taking an action (ReAct framework).

    This helps track reasoning and prevents loops.

    Args:
        thought: What you're thinking about the current situation
        next_action: What you plan to do next

    Returns:
        Confirmation of recorded thought
    """
    global _stage3b_thoughts
    entry = {
        "thought": thought,
        "next_action": next_action,
        "step": len(_stage3b_thoughts) + 1
    }
    _stage3b_thoughts.append(entry)

    return f"Thought #{entry['step']} recorded. Proceeding with: {next_action}"


@tool
def record_observation(what_happened: str, what_learned: str, next_step: str) -> str:
    """
    Record an observation after taking an action (ReAct framework).

    Args:
        what_happened: Result of the previous action
        what_learned: Key insight from the result
        next_step: What to do next based on this observation

    Returns:
        Confirmation with observation summary
    """
    global _stage3b_observations
    entry = {
        "what_happened": what_happened,
        "what_learned": what_learned,
        "next_step": next_step,
        "step": len(_stage3b_observations) + 1
    }
    _stage3b_observations.append(entry)

    return f"Observation #{entry['step']} recorded. Key insight: {what_learned}"


@tool
def run_data_prep_code(code: str, description: str = "") -> str:
    """
    Execute data preparation code in a sandboxed environment.

    The code should work with DataFrames and prepare data for modeling.

    Available:
    - pd, np, json
    - DATA_DIR, STAGE3_OUT_DIR, STAGE3B_OUT_DIR
    - load_dataframe()

    Args:
        code: Python code for data preparation
        description: What the code does

    Returns:
        Execution output
    """
    additional = {
        'DATA_DIR': DATA_DIR,
        'STAGE3_OUT_DIR': STAGE3_OUT_DIR,
        'STAGE3B_OUT_DIR': STAGE3B_OUT_DIR,
    }
    return execute_python_sandbox(code, additional, description)


@tool
def check_data_quality(data_description: str) -> str:
    """
    Check data quality of a prepared DataFrame.

    Should be called with code that creates a 'df' variable.

    Args:
        data_description: Description of the data being checked

    Returns:
        Quality report
    """
    code = """
import pandas as pd
import numpy as np

# Assume df is already loaded in namespace
if 'df' not in dir():
    print("ERROR: No DataFrame 'df' found in namespace")
else:
    print(f"=== Data Quality Report ===")
    print(f"Shape: {df.shape}")
    print(f"\\nNull counts by column:")
    null_counts = df.isnull().sum()
    for col, count in null_counts.items():
        if count > 0:
            pct = count / len(df) * 100
            print(f"  {col}: {count} ({pct:.1f}%)")

    total_nulls = null_counts.sum()
    if total_nulls == 0:
        print("  No nulls found - data is clean!")
    else:
        print(f"\\nTotal null cells: {total_nulls}")

    print(f"\\nDuplicate rows: {df.duplicated().sum()}")
    print(f"\\nData types:")
    for col, dtype in df.dtypes.items():
        print(f"  {col}: {dtype}")
"""
    return execute_python_sandbox(code, {}, data_description)


@tool
def save_prepared_data(
    plan_id: str,
    data_code: str,
    transformations: str,
    quality_notes: str
) -> str:
    """
    Save prepared data to parquet file with metadata.

    Args:
        plan_id: Plan ID (e.g., 'PLAN-TSK-001')
        data_code: Python code that creates 'df' DataFrame
        transformations: Comma-separated list of transformations applied
        quality_notes: Notes about data quality

    Returns:
        Confirmation with saved paths
    """
    try:
        # Execute code to get DataFrame
        success, output, namespace = execute_python_sandbox(data_code, {}, "Create prepared DataFrame")

        if not success or 'df' not in namespace:
            return f"Error: Code must create a 'df' DataFrame.\nOutput: {output}"

        df = namespace['df']

        # Verify no nulls
        null_count = df.isnull().sum().sum()
        if null_count > 0:
            return f"Error: DataFrame still has {null_count} null values. Handle all nulls before saving."

        # Save parquet
        output_filename = f"prepared_{plan_id}.parquet"
        output_path = STAGE3B_OUT_DIR / output_filename

        df.to_parquet(output_path, index=False)

        # Save metadata
        meta = {
            "plan_id": plan_id,
            "prepared_file_path": str(output_path),
            "original_row_count": len(df),
            "final_row_count": len(df),
            "columns_created": list(df.columns),
            "transformations_applied": [t.strip() for t in transformations.split(',')],
            "data_quality_report": {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "null_counts": {},
                "duplicate_rows": int(df.duplicated().sum()),
            },
            "has_no_nulls": True,
            "ready_for_modeling": True,
            "quality_notes": quality_notes
        }

        meta_path = DataPassingManager.save_artifact(
            data=meta,
            output_dir=STAGE3B_OUT_DIR,
            filename=f"prepared_{plan_id}_meta.json",
            metadata={"stage": "stage3b", "type": "prepared_data_metadata"}
        )

        return f"Prepared data saved:\n  Parquet: {output_path}\n  Metadata: {meta_path}\n  Shape: {df.shape}"

    except Exception as e:
        return f"Error saving prepared data: {e}"


@tool
def verify_prepared_data(plan_id: str) -> str:
    """
    Verify that prepared data exists and is ready for modeling.

    Args:
        plan_id: Plan ID to check

    Returns:
        Verification report
    """
    try:
        parquet_path = STAGE3B_OUT_DIR / f"prepared_{plan_id}.parquet"
        meta_path = STAGE3B_OUT_DIR / f"prepared_{plan_id}_meta.json"

        result = [f"=== Verification: {plan_id} ===\n"]

        if not parquet_path.exists():
            return f"Prepared data not found: {parquet_path}"

        # Load and check parquet
        df = pd.read_parquet(parquet_path)
        result.append(f"Parquet file: EXISTS")
        result.append(f"Shape: {df.shape}")

        # Check for nulls
        null_count = df.isnull().sum().sum()
        if null_count > 0:
            result.append(f"WARNING: {null_count} null values found!")
        else:
            result.append("Null values: 0 (clean)")

        # Check metadata
        if meta_path.exists():
            result.append(f"Metadata: EXISTS")
            meta = DataPassingManager.load_artifact(meta_path)
            result.append(f"Transformations: {meta.get('transformations_applied', [])}")
        else:
            result.append("Metadata: MISSING")

        result.append("\nVERIFICATION: PASSED" if null_count == 0 else "\nVERIFICATION: FAILED")

        return "\n".join(result)

    except Exception as e:
        return f"Error verifying data: {e}"


@tool
def get_react_summary() -> str:
    """
    Get a summary of all recorded thoughts and observations.

    Useful for reviewing the reasoning process.

    Returns:
        Summary of ReAct steps
    """
    global _stage3b_thoughts, _stage3b_observations

    result = ["=== ReAct Summary ===\n"]

    result.append(f"Total thoughts: {len(_stage3b_thoughts)}")
    result.append(f"Total observations: {len(_stage3b_observations)}\n")

    if _stage3b_thoughts:
        result.append("Thoughts:")
        for t in _stage3b_thoughts[-5:]:  # Last 5
            result.append(f"  #{t['step']}: {t['thought'][:100]}...")

    if _stage3b_observations:
        result.append("\nObservations:")
        for o in _stage3b_observations[-5:]:  # Last 5
            result.append(f"  #{o['step']}: {o['what_learned'][:100]}...")

    return "\n".join(result)


def reset_react_state():
    """Reset ReAct tracking state (called at start of stage)."""
    global _stage3b_thoughts, _stage3b_observations
    _stage3b_thoughts = []
    _stage3b_observations = []


# Export tools list
STAGE3B_TOOLS = [
    load_execution_plan,
    record_thought,
    record_observation,
    run_data_prep_code,
    check_data_quality,
    save_prepared_data,
    verify_prepared_data,
    get_react_summary,
]
