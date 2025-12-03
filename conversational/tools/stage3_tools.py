"""
Stage 3 Tools: Execution Planning

Tools for creating detailed execution plans based on task proposals.
"""

import json
from pathlib import Path
from typing import Optional
from langchain_core.tools import tool

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from code.config import (
    DATA_DIR, STAGE2_OUT_DIR, STAGE3_OUT_DIR,
    DataPassingManager, MIN_NON_NULL_FRACTION, logger
)
from code.utils import (
    load_dataframe, inspect_data_file, execute_python_sandbox,
    safe_json_dumps, list_data_files
)


@tool
def load_task_proposal(task_id: str) -> str:
    """
    Load a specific task proposal from Stage 2 output.

    Args:
        task_id: Task ID (e.g., 'TSK-001')

    Returns:
        Task proposal details as formatted string
    """
    try:
        proposals_path = STAGE2_OUT_DIR / "task_proposals.json"
        if not proposals_path.exists():
            return "Error: No task proposals found. Run Stage 2 first."

        data = DataPassingManager.load_artifact(proposals_path)
        proposals = data.get('proposals', data) if isinstance(data, dict) else data

        # Find the requested task
        for proposal in proposals:
            if proposal.get('id') == task_id:
                result = [
                    f"=== Task Proposal: {task_id} ===",
                    f"Title: {proposal.get('title')}",
                    f"Category: {proposal.get('category')}",
                    f"Feasibility: {proposal.get('feasibility_score', 'N/A')}",
                    "",
                    "Problem Statement:",
                    proposal.get('problem_statement', 'N/A'),
                    "",
                    f"Target Column: {proposal.get('target_column')}",
                    f"Target Dataset: {proposal.get('target_dataset')}",
                    f"Required Datasets: {proposal.get('required_datasets', [])}",
                    "",
                    "Full JSON:",
                    json.dumps(proposal, indent=2, default=str)
                ]
                return "\n".join(result)

        # Task not found - list available
        available = [p.get('id') for p in proposals]
        return f"Task '{task_id}' not found. Available tasks: {available}"

    except Exception as e:
        return f"Error loading proposal: {e}"


@tool
def list_all_proposals() -> str:
    """
    List all available task proposals from Stage 2.

    Returns summary of each proposal with ID, title, and category.
    """
    try:
        proposals_path = STAGE2_OUT_DIR / "task_proposals.json"
        if not proposals_path.exists():
            return "No task proposals found. Run Stage 2 first."

        data = DataPassingManager.load_artifact(proposals_path)
        proposals = data.get('proposals', data) if isinstance(data, dict) else data

        if not proposals:
            return "No proposals in file."

        result = ["=== Available Task Proposals ===\n"]
        for p in proposals:
            result.append(f"ID: {p.get('id')}")
            result.append(f"  Title: {p.get('title')}")
            result.append(f"  Category: {p.get('category')}")
            result.append(f"  Target: {p.get('target_column')} from {p.get('target_dataset')}")
            result.append(f"  Feasibility: {p.get('feasibility_score', 'N/A')}")
            result.append("")

        return "\n".join(result)

    except Exception as e:
        return f"Error listing proposals: {e}"


@tool
def list_data_files_stage3() -> str:
    """
    List all available data files for execution planning.

    Returns list of files with basic info.
    """
    try:
        files = list_data_files(DATA_DIR)
        if not files:
            return "No data files found in data directory."

        result = ["Available Data Files:"]
        for f in files:
            filepath = DATA_DIR / f
            if filepath.exists():
                size_mb = filepath.stat().st_size / (1024 * 1024)
                result.append(f"  - {f} ({size_mb:.2f} MB)")

        return "\n".join(result)

    except Exception as e:
        return f"Error listing files: {e}"


@tool
def inspect_data_file_stage3(filename: str, n_rows: int = 10) -> str:
    """
    Inspect a data file to understand its structure.

    Args:
        filename: Name of the file to inspect
        n_rows: Number of sample rows to show

    Returns:
        File inspection with schema, types, nulls, and sample data
    """
    return inspect_data_file(filename, n_rows, DATA_DIR)


@tool
def validate_columns_for_task(
    filename: str,
    columns: str,
    min_non_null: float = 0.65
) -> str:
    """
    Validate that columns meet quality requirements for modeling.

    Args:
        filename: Data file to check
        columns: Comma-separated list of column names
        min_non_null: Minimum fraction of non-null values required

    Returns:
        Validation report with pass/fail for each column
    """
    try:
        filepath = DATA_DIR / filename
        df = load_dataframe(filepath, nrows=5000)

        col_list = [c.strip() for c in columns.split(',')]
        result = [f"=== Column Validation: {filename} ===\n"]
        all_valid = True

        for col in col_list:
            if col not in df.columns:
                result.append(f"  {col}: MISSING - column not found")
                all_valid = False
                continue

            non_null_frac = 1 - df[col].isna().mean()
            valid = non_null_frac >= min_non_null

            status = "PASS" if valid else "FAIL"
            result.append(f"  {col}: {status} ({non_null_frac:.1%} non-null, min: {min_non_null:.1%})")

            if not valid:
                all_valid = False

        result.append("")
        result.append(f"Overall: {'VALID' if all_valid else 'INVALID'}")

        if not all_valid:
            result.append("\nRecommendation: Consider imputation or dropping rows with nulls")

        return "\n".join(result)

    except Exception as e:
        return f"Error validating columns: {e}"


@tool
def analyze_join_feasibility(
    left_file: str,
    right_file: str,
    left_key: str,
    right_key: str
) -> str:
    """
    Analyze if a join between two datasets is feasible.

    Args:
        left_file: First dataset filename
        right_file: Second dataset filename
        left_key: Join key in first dataset
        right_key: Join key in second dataset

    Returns:
        Join analysis with expected result size and potential issues
    """
    try:
        left_path = DATA_DIR / left_file
        right_path = DATA_DIR / right_file

        left_df = load_dataframe(left_path, nrows=10000)
        right_df = load_dataframe(right_path, nrows=10000)

        result = [f"=== Join Feasibility Analysis ===\n"]

        # Check keys exist
        if left_key not in left_df.columns:
            return f"Error: Key '{left_key}' not found in {left_file}"
        if right_key not in right_df.columns:
            return f"Error: Key '{right_key}' not found in {right_file}"

        # Analyze left key
        left_unique = left_df[left_key].nunique()
        left_nulls = left_df[left_key].isna().sum()
        result.append(f"{left_file}.{left_key}:")
        result.append(f"  Unique values: {left_unique}")
        result.append(f"  Null values: {left_nulls}")

        # Analyze right key
        right_unique = right_df[right_key].nunique()
        right_nulls = right_df[right_key].isna().sum()
        result.append(f"\n{right_file}.{right_key}:")
        result.append(f"  Unique values: {right_unique}")
        result.append(f"  Null values: {right_nulls}")

        # Check overlap
        left_vals = set(left_df[left_key].dropna().unique())
        right_vals = set(right_df[right_key].dropna().unique())
        overlap = left_vals & right_vals

        result.append(f"\nKey Overlap:")
        result.append(f"  Common values: {len(overlap)}")
        result.append(f"  Left-only values: {len(left_vals - right_vals)}")
        result.append(f"  Right-only values: {len(right_vals - left_vals)}")

        # Estimate join result
        if len(overlap) == 0:
            result.append("\nWARNING: No common values - inner join will be empty!")
            result.append("Consider: Check key types, data cleaning, or use outer join")
        else:
            overlap_pct = len(overlap) / min(len(left_vals), len(right_vals)) * 100
            result.append(f"\nOverlap percentage: {overlap_pct:.1f}%")

            if overlap_pct < 50:
                result.append("WARNING: Low overlap - consider data quality issues")

        # Check for many-to-many
        left_dups = left_df[left_key].duplicated().sum()
        right_dups = right_df[right_key].duplicated().sum()

        if left_dups > 0 and right_dups > 0:
            result.append("\nWARNING: Both keys have duplicates - many-to-many join!")
            result.append("Result may have more rows than expected")

        return "\n".join(result)

    except Exception as e:
        return f"Error analyzing join: {e}"


@tool
def python_sandbox_stage3(code: str, description: str = "") -> str:
    """
    Execute Python code in a sandboxed environment for Stage 3.

    Available: pd, np, json, DATA_DIR, load_dataframe()

    Args:
        code: Python code to execute
        description: What the code does

    Returns:
        Execution output
    """
    return execute_python_sandbox(code, {'DATA_DIR': DATA_DIR}, description)


@tool
def save_stage3_plan(plan_json: str) -> str:
    """
    Save the execution plan to Stage 3 output directory.

    Args:
        plan_json: JSON string containing the execution plan

    Returns:
        Confirmation with saved path
    """
    try:
        plan = json.loads(plan_json)

        # Validate required fields
        required = ['plan_id', 'selected_task_id', 'goal', 'task_category', 'target_column']
        missing = [f for f in required if f not in plan]
        if missing:
            return f"Error: Missing required fields: {missing}"

        plan_id = plan['plan_id']
        filename = f"{plan_id}.json"

        output_path = DataPassingManager.save_artifact(
            data=plan,
            output_dir=STAGE3_OUT_DIR,
            filename=filename,
            metadata={"stage": "stage3", "type": "execution_plan"}
        )

        return f"Execution plan saved to: {output_path}"

    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON - {e}"
    except Exception as e:
        return f"Error saving plan: {e}"


@tool
def get_execution_plan_template() -> str:
    """
    Get a template for creating an execution plan.

    Returns template with all fields explained.
    """
    template = {
        "plan_id": "PLAN-TSK-001",
        "selected_task_id": "TSK-001",
        "goal": "Brief description of what we're trying to achieve",
        "task_category": "forecasting",

        "file_instructions": [
            {
                "filename": "data.csv",
                "filepath": "/path/to/data.csv",
                "columns_to_use": ["col1", "col2", "target"],
                "filters": ["col1 > 0", "col2.notna()"],
                "date_column": "date",
                "parse_dates": ["date"]
            }
        ],

        "join_steps": [
            {
                "left_dataset": "data1.csv",
                "right_dataset": "data2.csv",
                "left_on": "key",
                "right_on": "key",
                "how": "inner",
                "validation_rule": "Expect 1:1 mapping"
            }
        ],

        "feature_engineering": [
            {
                "name": "lag_1",
                "description": "1-period lag of target",
                "source_columns": ["target"],
                "implementation_code": "df['lag_1'] = df['target'].shift(1)",
                "dtype": "float64"
            }
        ],

        "target_column": "target",
        "date_column": "date",
        "validation_strategy": "temporal",
        "train_end_date": "2023-06-30",
        "validation_end_date": "2023-09-30",

        "expected_model_types": ["arima", "random_forest", "linear_regression"],
        "evaluation_metrics": ["mae", "rmse", "mape"],

        "output_columns": ["date", "actual", "predicted"],
        "artifacts_to_save": ["model", "predictions", "feature_importance"]
    }

    return "Execution Plan Template:\n\n" + json.dumps(template, indent=2)


# Export tools list
STAGE3_TOOLS = [
    load_task_proposal,
    list_all_proposals,
    list_data_files_stage3,
    inspect_data_file_stage3,
    validate_columns_for_task,
    analyze_join_feasibility,
    python_sandbox_stage3,
    save_stage3_plan,
    get_execution_plan_template,
]
