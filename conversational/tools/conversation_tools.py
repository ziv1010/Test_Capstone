"""
Conversation Tools: User Interaction

Tools for the conversational agent to interact with users and manage the pipeline.
"""

import json
from pathlib import Path
from typing import Optional, List
from langchain_core.tools import tool
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from code.config import (
    DATA_DIR, SUMMARIES_DIR, STAGE2_OUT_DIR, STAGE3_OUT_DIR,
    STAGE3B_OUT_DIR, STAGE3_5A_OUT_DIR, STAGE3_5B_OUT_DIR,
    STAGE4_OUT_DIR, STAGE5_OUT_DIR,
    CONVERSATION_STATE_DIR, DataPassingManager, logger
)
from code.utils import list_data_files, list_summary_files, load_dataframe


@tool
def get_available_data() -> str:
    """
    Get information about available datasets.

    Returns list of datasets with basic info for user reference.
    """
    try:
        files = list_data_files(DATA_DIR)

        if not files:
            return "No datasets found in the data directory."

        result = ["Available Datasets:"]
        for f in files:
            filepath = DATA_DIR / f
            if filepath.exists():
                size_mb = filepath.stat().st_size / (1024 * 1024)
                result.append(f"  - {f} ({size_mb:.2f} MB)")

        result.append(f"\nTotal: {len(files)} datasets")
        return "\n".join(result)

    except Exception as e:
        return f"Error listing data: {e}"


@tool
def get_summaries() -> str:
    """
    Get summaries of analyzed datasets (from Stage 1).

    Returns overview of what data is available and its characteristics.
    """
    try:
        summary_files = list_summary_files(SUMMARIES_DIR)

        if not summary_files:
            return "No dataset summaries available. Would you like me to analyze the datasets first?"

        result = ["Dataset Summaries:\n"]

        for sf in summary_files:
            try:
                summary = DataPassingManager.load_artifact(SUMMARIES_DIR / sf)
                data = summary.get('data', summary) if isinstance(summary, dict) else summary

                result.append(f"{data.get('filename', sf)}:")
                result.append(f"  - {data.get('n_rows', '?')} rows, {data.get('n_cols', '?')} columns")

                if data.get('has_datetime_column'):
                    result.append("  - Has datetime column (suitable for time series)")

                if data.get('has_target_candidates'):
                    result.append(f"  - Potential targets: {data.get('has_target_candidates')[:3]}")

                quality = data.get('data_quality_score')
                if quality:
                    result.append(f"  - Quality score: {quality:.1%}")

                result.append("")

            except Exception as e:
                result.append(f"{sf}: Error reading - {e}\n")

        return "\n".join(result)

    except Exception as e:
        return f"Error getting summaries: {e}"


@tool
def get_task_proposals() -> str:
    """
    Get proposed analytical tasks (from Stage 2).

    Returns list of tasks that can be executed on the data.
    """
    try:
        proposals_path = STAGE2_OUT_DIR / "task_proposals.json"

        if not proposals_path.exists():
            return "No task proposals available. Would you like me to generate some based on your data?"

        data = DataPassingManager.load_artifact(proposals_path)
        proposals = data.get('proposals', data) if isinstance(data, dict) else data

        if not proposals:
            return "No proposals found in file."

        result = ["Proposed Analysis Tasks:\n"]

        for p in proposals:
            result.append(f"{p.get('id')}: {p.get('title')}")
            result.append(f"  Category: {p.get('category')}")
            result.append(f"  Target: {p.get('target_column')}")
            result.append(f"  Feasibility: {p.get('feasibility_score', 'N/A')}")
            result.append(f"  Description: {p.get('problem_statement', 'N/A')[:200]}...")
            result.append("")

        result.append("Use 'run task TSK-XXX' to execute a specific task.")
        return "\n".join(result)

    except Exception as e:
        return f"Error getting proposals: {e}"


@tool
def check_pipeline_status() -> str:
    """
    Check the current status of the pipeline.

    Returns which stages have been completed and what's available.
    """
    try:
        status = {
            "Stage 1 (Summarization)": "not_run",
            "Stage 2 (Task Proposals)": "not_run",
            "Stage 3 (Execution Plan)": "not_run",
            "Stage 3B (Data Preparation)": "not_run",
            "Stage 3.5A (Method Proposals)": "not_run",
            "Stage 3.5B (Benchmarking)": "not_run",
            "Stage 4 (Execution)": "not_run",
            "Stage 5 (Visualization)": "not_run",
        }

        # Check Stage 1
        summaries = list(SUMMARIES_DIR.glob("*.summary.json"))
        if summaries:
            status["Stage 1 (Summarization)"] = f"completed ({len(summaries)} datasets)"

        # Check Stage 2
        if (STAGE2_OUT_DIR / "task_proposals.json").exists():
            status["Stage 2 (Task Proposals)"] = "completed"

        # Check Stage 3+
        plans = list(STAGE3_OUT_DIR.glob("PLAN-*.json"))
        if plans:
            plan_ids = [p.stem for p in plans]
            status["Stage 3 (Execution Plan)"] = f"completed ({plan_ids})"

            for plan_id in plan_ids:
                # Check downstream stages
                if (STAGE3B_OUT_DIR / f"prepared_{plan_id}.parquet").exists():
                    status["Stage 3B (Data Preparation)"] = f"completed ({plan_id})"

                if list(Path(STAGE3_5A_OUT_DIR).glob(f"method_proposal_{plan_id}.json")):
                    status["Stage 3.5A (Method Proposals)"] = f"completed ({plan_id})"

                if list(Path(STAGE3_5B_OUT_DIR).glob(f"tester_{plan_id}.json")):
                    status["Stage 3.5B (Benchmarking)"] = f"completed ({plan_id})"

                if (STAGE4_OUT_DIR / f"execution_result_{plan_id}.json").exists():
                    status["Stage 4 (Execution)"] = f"completed ({plan_id})"

                if list(Path(STAGE5_OUT_DIR).glob(f"visualization_report_{plan_id}.json")):
                    status["Stage 5 (Visualization)"] = f"completed ({plan_id})"

        result = ["Pipeline Status:\n"]
        for stage, state in status.items():
            icon = "Y" if "completed" in state else " "
            result.append(f"  [{icon}] {stage}: {state}")

        return "\n".join(result)

    except Exception as e:
        return f"Error checking status: {e}"


@tool
def evaluate_user_query(query: str) -> str:
    """
    Evaluate if a user's forecasting query is feasible with available data.

    Uses Chain-of-Thought reasoning to assess:
    - Required data
    - Required columns
    - Feasibility of the prediction

    Args:
        query: User's natural language query about forecasting

    Returns:
        Feasibility assessment with reasoning
    """
    try:
        result = ["=== Query Feasibility Analysis ===", f"Query: {query}\n"]

        # Step 1: Identify what's being requested
        result.append("Step 1: Understanding the Request")
        query_lower = query.lower()

        target_hints = []
        if 'price' in query_lower:
            target_hints.append('price')
        if 'sales' in query_lower:
            target_hints.append('sales')
        if 'demand' in query_lower:
            target_hints.append('demand')
        if 'revenue' in query_lower:
            target_hints.append('revenue')
        if 'quantity' in query_lower:
            target_hints.append('quantity')

        if target_hints:
            result.append(f"  Detected target variables: {target_hints}")
        else:
            result.append("  No specific target detected - will look for numeric columns")

        # Step 2: Check available data
        result.append("\nStep 2: Checking Available Data")
        summary_files = list_summary_files(SUMMARIES_DIR)

        if not summary_files:
            result.append("  No data summaries available!")
            result.append("\nConclusion: INFEASIBLE - No data has been analyzed yet")
            result.append("Recommendation: Run Stage 1 to analyze available datasets")
            return "\n".join(result)

        result.append(f"  Found {len(summary_files)} analyzed datasets")

        # Step 3: Match query to data
        result.append("\nStep 3: Matching Query to Data")
        matches = []

        for sf in summary_files:
            try:
                summary = DataPassingManager.load_artifact(SUMMARIES_DIR / sf)
                data = summary.get('data', summary)

                # Check for matching columns
                columns = data.get('columns', [])
                col_names = [c.get('name', '').lower() for c in columns]

                for hint in target_hints:
                    matching_cols = [c for c in col_names if hint in c]
                    if matching_cols:
                        matches.append({
                            'dataset': data.get('filename'),
                            'columns': matching_cols,
                            'has_datetime': data.get('has_datetime_column', False),
                            'quality': data.get('data_quality_score', 0)
                        })

                # Also check target candidates
                if data.get('has_target_candidates'):
                    for target in data.get('has_target_candidates', []):
                        if any(h in target.lower() for h in target_hints) or not target_hints:
                            matches.append({
                                'dataset': data.get('filename'),
                                'columns': [target],
                                'has_datetime': data.get('has_datetime_column', False),
                                'quality': data.get('data_quality_score', 0)
                            })

            except Exception:
                continue

        if matches:
            result.append(f"  Found {len(matches)} potential matches:")
            for m in matches[:5]:
                result.append(f"    - {m['dataset']}: {m['columns']}")
                if m['has_datetime']:
                    result.append("      (has datetime - suitable for time series)")
        else:
            result.append("  No direct column matches found")

        # Step 4: Feasibility assessment
        result.append("\nStep 4: Feasibility Assessment")

        if matches:
            best_match = max(matches, key=lambda x: (x['has_datetime'], x['quality']))

            if best_match['has_datetime']:
                result.append("  FEASIBLE: Data with datetime column found")
                result.append(f"  Recommended dataset: {best_match['dataset']}")
                result.append(f"  Target column(s): {best_match['columns']}")
                result.append("\nRecommendation: This query can be executed as a forecasting task")
            else:
                result.append("  PARTIALLY FEASIBLE: Matching data found but no datetime column")
                result.append("  May need to use row index as time proxy")
                result.append("\nRecommendation: Consider if data has implicit temporal ordering")
        else:
            result.append("  UNCERTAIN: No exact matches, but forecasting may still be possible")
            result.append("  Available target candidates may work")
            result.append("\nRecommendation: Run Stage 2 to get detailed task proposals")

        return "\n".join(result)

    except Exception as e:
        return f"Error evaluating query: {e}"


@tool
def create_custom_task_from_query(
    query: str,
    dataset: str,
    target_column: str,
    date_column: str = None
) -> str:
    """
    Create a custom task proposal based on user's query.

    Args:
        query: User's forecasting query
        dataset: Dataset filename to use
        target_column: Column to predict
        date_column: Date column for temporal analysis (optional)

    Returns:
        Generated task proposal
    """
    try:
        # Generate task ID
        import time
        task_id = f"TSK-{int(time.time()) % 10000:04d}"

        proposal = {
            "id": task_id,
            "category": "forecasting",
            "title": f"Custom Forecasting: {target_column}",
            "problem_statement": query,
            "required_datasets": [dataset],
            "target_column": target_column,
            "target_dataset": dataset,
            "feature_columns": [],
            "validation_plan": {
                "train_fraction": 0.7,
                "validation_fraction": 0.15,
                "test_fraction": 0.15,
                "split_strategy": "temporal" if date_column else "random",
                "date_column": date_column
            },
            "feasibility_score": 0.7,
            "feasibility_notes": f"Custom task created from user query: {query}",
            "forecast_horizon": 30,
            "forecast_granularity": "daily" if date_column else "sequential"
        }

        result = [
            "=== Custom Task Created ===",
            f"Task ID: {task_id}",
            f"Target: {target_column} from {dataset}",
            f"Category: forecasting",
            "",
            "Proposal:",
            json.dumps(proposal, indent=2),
            "",
            f"To execute this task, use: 'run task {task_id}'"
        ]

        # Save to proposals
        proposals_path = STAGE2_OUT_DIR / "task_proposals.json"
        if proposals_path.exists():
            existing = DataPassingManager.load_artifact(proposals_path)
            proposals = existing.get('proposals', [])
        else:
            proposals = []

        proposals.append(proposal)

        DataPassingManager.save_artifact(
            data={"proposals": proposals},
            output_dir=STAGE2_OUT_DIR,
            filename="task_proposals.json"
        )

        return "\n".join(result)

    except Exception as e:
        return f"Error creating task: {e}"


@tool
def get_execution_results(plan_id: str = None) -> str:
    """
    Get results from executed tasks.

    Args:
        plan_id: Specific plan ID. If not provided, shows all results.

    Returns:
        Execution results summary
    """
    try:
        if plan_id:
            result_path = STAGE4_OUT_DIR / f"execution_result_{plan_id}.json"
            if not result_path.exists():
                return f"No results found for {plan_id}"

            result_data = DataPassingManager.load_artifact(result_path)

            result = [
                f"=== Execution Results: {plan_id} ===",
                f"Status: {result_data.get('status')}",
                f"Summary: {result_data.get('summary')}",
                "",
                "Metrics:",
            ]

            for metric, value in result_data.get('metrics', {}).items():
                result.append(f"  - {metric}: {value}")

            return "\n".join(result)

        else:
            # List all results
            results = list(STAGE4_OUT_DIR.glob("execution_result_*.json"))

            if not results:
                return "No execution results available."

            output = ["=== All Execution Results ===\n"]

            for r in results:
                try:
                    data = DataPassingManager.load_artifact(r)
                    plan_id = r.stem.replace("execution_result_", "")
                    output.append(f"{plan_id}:")
                    output.append(f"  Status: {data.get('status')}")
                    output.append(f"  Metrics: {data.get('metrics', {})}")
                    output.append("")
                except:
                    continue

            return "\n".join(output)

    except Exception as e:
        return f"Error getting results: {e}"


@tool
def get_visualizations(plan_id: str = None) -> str:
    """
    Get visualization reports and plot locations.

    Args:
        plan_id: Specific plan ID. If not provided, shows all.

    Returns:
        Visualization report
    """
    try:
        if plan_id:
            report_path = STAGE5_OUT_DIR / f"visualization_report_{plan_id}.json"
            plots = list(STAGE5_OUT_DIR.glob(f"{plan_id}_*.png"))

            result = [f"=== Visualizations: {plan_id} ===\n"]

            if plots:
                result.append("Generated Plots:")
                for p in plots:
                    result.append(f"  - {p.name}")

            if report_path.exists():
                report = DataPassingManager.load_artifact(report_path)
                result.append(f"\nInsights:")
                for insight in report.get('insights', []):
                    result.append(f"  - {insight}")

            return "\n".join(result)

        else:
            # List all visualizations
            reports = list(STAGE5_OUT_DIR.glob("visualization_report_*.json"))
            plots = list(STAGE5_OUT_DIR.glob("*.png"))

            result = ["=== All Visualizations ===\n"]
            result.append(f"Total plots: {len(plots)}")
            result.append(f"Reports: {len(reports)}")

            if plots:
                result.append("\nPlot files:")
                for p in plots[:20]:  # First 20
                    result.append(f"  - {p.name}")

            return "\n".join(result)

    except Exception as e:
        return f"Error getting visualizations: {e}"


# Export tools list
CONVERSATION_TOOLS = [
    get_available_data,
    get_summaries,
    get_task_proposals,
    check_pipeline_status,
    evaluate_user_query,
    create_custom_task_from_query,
    get_execution_results,
    get_visualizations,
]
