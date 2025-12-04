"""
Stage 2 Tools: Task Proposal Generation

Tools for exploring dataset summaries and proposing analytical tasks.
"""

import json
from pathlib import Path
from typing import Optional
from langchain_core.tools import tool

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from code.config import (
    DATA_DIR, SUMMARIES_DIR, STAGE2_OUT_DIR,
    DataPassingManager, logger
)
from code.utils import (
    list_summary_files, read_summary_file,
    execute_python_sandbox, safe_json_dumps
)


@tool
def list_dataset_summaries() -> str:
    """
    List all available dataset summaries from Stage 1.

    Returns list of summary files with basic info about each dataset.
    """
    try:
        files = list_summary_files(SUMMARIES_DIR)
        if not files:
            return "No dataset summaries found. Run Stage 1 first."

        result = ["Available Dataset Summaries:"]
        for f in files:
            try:
                summary = read_summary_file(f, SUMMARIES_DIR)
                data = summary.get('data', summary)  # Handle wrapped format
                n_rows = data.get('n_rows', 'unknown')
                n_cols = data.get('n_cols', 'unknown')
                result.append(f"  - {f}: {n_rows} rows, {n_cols} columns")
            except:
                result.append(f"  - {f}")

        return "\n".join(result)

    except Exception as e:
        return f"Error listing summaries: {e}"


@tool
def read_dataset_summary(filename: str) -> str:
    """
    Read a specific dataset summary.

    Args:
        filename: Name of the summary file (e.g., 'dataset.summary.json')

    Returns:
        Full summary content as formatted string
    """
    try:
        summary = read_summary_file(filename, SUMMARIES_DIR)
        data = summary.get('data', summary)  # Handle wrapped format

        # Format for readability
        result = [
            f"=== Summary: {filename} ===",
            f"Dataset: {data.get('filename', 'unknown')}",
            f"Shape: {data.get('n_rows', '?')} rows x {data.get('n_cols', '?')} columns",
            "",
            "Columns:",
        ]

        for col in data.get('columns', []):
            name = col.get('name', 'unknown')
            ltype = col.get('logical_type', 'unknown')
            nulls = col.get('null_fraction', 0)
            result.append(f"  - {name}: {ltype} (nulls: {nulls:.1%})")
            
            # Add semantic info for categorical columns (helps model understand values)
            if col.get('value_interpretation'):
                result.append(f"    → {col['value_interpretation']}")
            elif col.get('unique_values') and len(col['unique_values']) <= 10:
                result.append(f"    → Values: {', '.join(col['unique_values'])}")

        if data.get('candidate_keys'):
            result.append(f"\nCandidate Keys: {data.get('candidate_keys')}")

        if data.get('has_target_candidates'):
            result.append(f"Target Candidates: {data.get('has_target_candidates')}")

        if data.get('has_datetime_column'):
            result.append("Has datetime column: Yes")

        result.append("")
        result.append("Full JSON available in summary file.")

        return "\n".join(result)

    except FileNotFoundError:
        return f"Summary file not found: {filename}"
    except Exception as e:
        return f"Error reading summary: {e}"


@tool
def explore_data_relationships(summary_files: str) -> str:
    """
    Explore potential relationships between multiple datasets.

    Args:
        summary_files: Comma-separated list of summary filenames

    Returns:
        Analysis of potential joins and relationships
    """
    try:
        files = [f.strip() for f in summary_files.split(',')]
        summaries = {}

        for f in files:
            try:
                summary = read_summary_file(f, SUMMARIES_DIR)
                summaries[f] = summary.get('data', summary)
            except:
                return f"Could not read: {f}"

        if len(summaries) < 2:
            return "Need at least 2 datasets to explore relationships."

        result = ["=== Data Relationship Analysis ===\n"]

        # Find common column names
        all_columns = {}
        for fname, data in summaries.items():
            for col in data.get('columns', []):
                col_name = col.get('name', '').lower()
                if col_name not in all_columns:
                    all_columns[col_name] = []
                all_columns[col_name].append({
                    'file': fname,
                    'original_name': col.get('name'),
                    'type': col.get('logical_type'),
                    'unique_fraction': col.get('unique_fraction', 0)
                })

        # Find potential join keys
        result.append("Potential Join Keys (columns in multiple datasets):")
        join_candidates = []
        for col_name, occurrences in all_columns.items():
            if len(occurrences) > 1:
                files_with_col = [o['file'] for o in occurrences]
                types = [o['type'] for o in occurrences]
                result.append(f"  - '{col_name}' found in: {files_with_col}")
                result.append(f"    Types: {types}")

                # Good join key if high uniqueness in at least one dataset
                max_unique = max(o['unique_fraction'] for o in occurrences)
                if max_unique > 0.5:
                    join_candidates.append(col_name)

        if join_candidates:
            result.append(f"\nBest join candidates (high uniqueness): {join_candidates}")
        else:
            result.append("\nNo strong join candidates found.")

        # Suggest relationships
        result.append("\n--- Suggested Relationships ---")
        for i, (f1, d1) in enumerate(summaries.items()):
            for f2, d2 in list(summaries.items())[i+1:]:
                cols1 = {c['name'].lower(): c for c in d1.get('columns', [])}
                cols2 = {c['name'].lower(): c for c in d2.get('columns', [])}
                common = set(cols1.keys()) & set(cols2.keys())
                if common:
                    result.append(f"\n{f1} <-> {f2}:")
                    result.append(f"  Common columns: {list(common)}")

        return "\n".join(result)

    except Exception as e:
        return f"Error exploring relationships: {e}"


@tool
def python_sandbox_stage2(code: str, description: str = "") -> str:
    """
    Execute Python code in a sandboxed environment.

    Available in sandbox:
    - pd (pandas), np (numpy), json
    - DATA_DIR, SUMMARIES_DIR
    - load_dataframe() function

    Args:
        code: Python code to execute
        description: Description of what the code does

    Returns:
        Output from code execution
    """
    try:
        # Add stage-specific imports to sandbox
        additional = {
            'SUMMARIES_DIR': SUMMARIES_DIR,
        }
        return execute_python_sandbox(code, additional, description)
    except Exception as e:
        return f"Sandbox error: {e}"


@tool
def evaluate_forecasting_feasibility(
    target_column: str,
    date_column: str,
    dataset_summaries: str
) -> str:
    """
    Evaluate if a forecasting task is feasible given the data.

    Args:
        target_column: The column to predict
        date_column: The datetime column to use
        dataset_summaries: Comma-separated list of relevant summary files

    Returns:
        Feasibility assessment with score and recommendations
    """
    try:
        files = [f.strip() for f in dataset_summaries.split(',')]

        report = ["=== Forecasting Feasibility Assessment ===\n"]
        issues = []
        score = 100

        for f in files:
            try:
                summary = read_summary_file(f, SUMMARIES_DIR)
                data = summary.get('data', summary)
                columns = {c['name']: c for c in data.get('columns', [])}

                report.append(f"Dataset: {f}")

                # Check target column
                if target_column in columns:
                    target = columns[target_column]
                    report.append(f"  Target '{target_column}': {target.get('logical_type')}")

                    if target.get('logical_type') not in ['integer', 'float', 'numeric']:
                        issues.append(f"Target '{target_column}' is not numeric")
                        score -= 30

                    if target.get('null_fraction', 0) > 0.2:
                        issues.append(f"Target has {target.get('null_fraction'):.1%} nulls")
                        score -= 20
                else:
                    issues.append(f"Target '{target_column}' not found in {f}")
                    score -= 50

                # Check date column
                if date_column in columns:
                    date = columns[date_column]
                    report.append(f"  Date '{date_column}': {date.get('logical_type')}")

                    if date.get('logical_type') != 'datetime':
                        issues.append(f"'{date_column}' is not datetime type")
                        score -= 20
                else:
                    issues.append(f"Date column '{date_column}' not found in {f}")
                    score -= 40

                # Check data size
                n_rows = data.get('n_rows', 0)
                report.append(f"  Rows: {n_rows}")
                if n_rows < 50:
                    issues.append("Very few data points for forecasting")
                    score -= 30
                elif n_rows < 100:
                    issues.append("Limited data points - results may be unreliable")
                    score -= 10

            except Exception as e:
                issues.append(f"Could not read {f}: {e}")
                score -= 25

        report.append("")
        score = max(0, score)

        if issues:
            report.append("Issues Found:")
            for issue in issues:
                report.append(f"  - {issue}")
            report.append("")

        report.append(f"Feasibility Score: {score}/100")

        if score >= 70:
            report.append("VERDICT: Forecasting is feasible")
        elif score >= 40:
            report.append("VERDICT: Forecasting may work with preprocessing")
        else:
            report.append("VERDICT: Forecasting is not recommended")
            report.append("Consider: classification, clustering, or descriptive analysis")

        return "\n".join(report)

    except Exception as e:
        return f"Error assessing feasibility: {e}"


@tool
def save_task_proposals(proposals_json: str) -> str:
    """
    Save task proposals to the Stage 2 output directory.

    Args:
        proposals_json: JSON string containing task proposals

    Returns:
        Confirmation with saved path
    """
    try:
        proposals = json.loads(proposals_json)

        # Validate structure
        if 'proposals' not in proposals:
            return "Error: JSON must contain 'proposals' key"

        # Save using DataPassingManager for robust saving
        output_path = DataPassingManager.save_artifact(
            data=proposals,
            output_dir=STAGE2_OUT_DIR,
            filename="task_proposals.json",
            metadata={"stage": "stage2", "type": "task_proposals"}
        )

        n_proposals = len(proposals.get('proposals', []))
        return f"Saved {n_proposals} task proposals to: {output_path}"

    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON - {e}"
    except Exception as e:
        return f"Error saving proposals: {e}"


@tool
def get_proposal_template() -> str:
    """
    Get a template for creating task proposals.

    Returns JSON template with all required fields explained.
    """
    template = {
        "proposals": [
            {
                "id": "TSK-001",
                "category": "forecasting | classification | regression | clustering | anomaly_detection | descriptive",
                "title": "Short descriptive title",
                "problem_statement": "Detailed description of the analytical problem",
                "required_datasets": ["dataset1.csv", "dataset2.csv"],
                "target_column": "column_to_predict",
                "target_dataset": "dataset_containing_target.csv",
                "feature_columns": ["feature1", "feature2"],
                "join_plan": {
                    "datasets": ["dataset1.csv", "dataset2.csv"],
                    "join_keys": {"dataset1.csv": "key1", "dataset2.csv": "key2"},
                    "join_type": "inner"
                },
                "validation_plan": {
                    "train_fraction": 0.7,
                    "validation_fraction": 0.15,
                    "test_fraction": 0.15,
                    "split_strategy": "temporal",
                    "date_column": "date_column_name"
                },
                "feasibility_score": 0.8,
                "feasibility_notes": "Notes on why this task is feasible",
                "forecast_horizon": 30,
                "forecast_granularity": "daily"
            }
        ]
    }

    return "Task Proposal Template:\n\n" + json.dumps(template, indent=2)


# Export tools list
STAGE2_TOOLS = [
    list_dataset_summaries,
    read_dataset_summary,
    explore_data_relationships,
    python_sandbox_stage2,
    evaluate_forecasting_feasibility,
    save_task_proposals,
    get_proposal_template,
]
