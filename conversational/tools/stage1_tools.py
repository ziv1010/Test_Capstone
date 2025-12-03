"""
Stage 1 Tools: Dataset Summarization

Tools for profiling datasets and generating summaries.
"""

import json
from pathlib import Path
from typing import List
from langchain_core.tools import tool

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from code.config import DATA_DIR, SUMMARIES_DIR, logger
from code.utils import profile_csv, list_data_files
from code.models import DatasetSummary


@tool
def list_available_datasets() -> str:
    """
    List all available datasets in the data directory.

    Returns a list of filenames with their sizes.
    """
    try:
        files = list_data_files(DATA_DIR)
        if not files:
            return "No data files found in the data directory."

        result = ["Available datasets:"]
        for f in files:
            filepath = DATA_DIR / f
            if filepath.exists():
                size_mb = filepath.stat().st_size / (1024 * 1024)
                result.append(f"  - {f} ({size_mb:.2f} MB)")
            else:
                result.append(f"  - {f}")

        return "\n".join(result)

    except Exception as e:
        return f"Error listing datasets: {e}"


@tool
def profile_dataset(filename: str) -> str:
    """
    Profile a single dataset and return a detailed summary.

    Args:
        filename: Name of the file to profile (must be in data directory)

    Returns:
        JSON string with dataset summary including column types, statistics, and quality metrics
    """
    try:
        filepath = DATA_DIR / filename
        if not filepath.exists():
            return f"Error: File not found: {filename}"

        summary = profile_csv(filepath)

        # Convert to dict for JSON serialization
        summary_dict = summary.model_dump()

        # Add a human-readable section
        readable = [
            f"\n=== Dataset Summary: {filename} ===",
            f"Rows: {summary.n_rows}, Columns: {summary.n_cols}",
            f"Quality Score: {summary.data_quality_score:.2%}",
            "",
            "Column Details:",
        ]

        for col in summary.columns:
            readable.append(
                f"  - {col.name}: {col.logical_type.value} "
                f"(nulls: {col.null_fraction:.1%}, unique: {col.n_unique})"
            )

        if summary.datetime_columns:
            readable.append(f"\nDatetime columns: {summary.datetime_columns}")

        if summary.has_target_candidates:
            readable.append(f"Potential target columns: {summary.has_target_candidates}")

        readable.append("\n" + "=" * 50)

        return "\n".join(readable) + "\n\nFull JSON:\n" + json.dumps(summary_dict, indent=2, default=str)

    except Exception as e:
        logger.error(f"Error profiling {filename}: {e}")
        return f"Error profiling dataset: {e}"


@tool
def save_dataset_summary(filename: str, summary_json: str) -> str:
    """
    Save a dataset summary to the summaries directory.

    Args:
        filename: Original dataset filename
        summary_json: JSON string containing the summary

    Returns:
        Confirmation message with saved path
    """
    try:
        # Parse JSON to validate
        summary_data = json.loads(summary_json)

        # Generate output filename
        output_name = f"{Path(filename).stem}.summary.json"
        output_path = SUMMARIES_DIR / output_name

        # Save with metadata
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, default=str)

        return f"Summary saved to: {output_path}"

    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON - {e}"
    except Exception as e:
        return f"Error saving summary: {e}"


@tool
def get_existing_summaries() -> str:
    """
    Get a list of existing dataset summaries.

    Returns list of summary files that have already been created.
    """
    try:
        summaries = list(SUMMARIES_DIR.glob("*.summary.json"))

        if not summaries:
            return "No existing summaries found."

        result = ["Existing summaries:"]
        for s in summaries:
            result.append(f"  - {s.name}")

        return "\n".join(result)

    except Exception as e:
        return f"Error listing summaries: {e}"


@tool
def analyze_dataset_for_forecasting(filename: str) -> str:
    """
    Analyze a dataset specifically for forecasting potential.

    Identifies:
    - Datetime columns (for temporal features)
    - Numeric columns (for targets)
    - Data frequency and range
    - Quality issues

    Args:
        filename: Dataset filename to analyze

    Returns:
        Analysis report for forecasting suitability
    """
    try:
        filepath = DATA_DIR / filename
        if not filepath.exists():
            return f"Error: File not found: {filename}"

        summary = profile_csv(filepath)

        report = [
            f"=== Forecasting Analysis: {filename} ===",
            "",
        ]

        # Check for datetime columns
        dt_cols = summary.datetime_columns
        if dt_cols:
            report.append(f"Datetime columns found: {dt_cols}")
            for col in summary.columns:
                if col.name in dt_cols:
                    report.append(f"  - {col.name}:")
                    if col.min_date:
                        report.append(f"    Range: {col.min_date} to {col.max_date}")
                    if col.date_frequency:
                        report.append(f"    Frequency: {col.date_frequency}")
        else:
            report.append("WARNING: No datetime columns detected.")
            report.append("  - Look for columns that might be dates in text format")
            report.append("  - Or consider using row index as time proxy")

        report.append("")

        # Check for numeric target candidates
        targets = summary.has_target_candidates
        if targets:
            report.append(f"Potential target columns: {targets}")
            for col in summary.columns:
                if col.name in targets:
                    stats = ""
                    if col.min_value is not None:
                        stats = f"(range: {col.min_value:.2f} to {col.max_value:.2f})"
                    report.append(f"  - {col.name}: {col.n_unique} unique values {stats}")
        else:
            report.append("WARNING: No clear target candidates found.")

        report.append("")

        # Quality assessment
        report.append("Data Quality:")
        report.append(f"  - Overall quality score: {summary.data_quality_score:.2%}")
        report.append(f"  - Total rows: {summary.n_rows}")

        high_null_cols = [c for c in summary.columns if c.null_fraction > 0.3]
        if high_null_cols:
            report.append(f"  - Columns with >30% nulls: {[c.name for c in high_null_cols]}")

        report.append("")

        # Forecasting suitability score
        score = 0
        if dt_cols:
            score += 40
        if targets:
            score += 30
        if summary.data_quality_score > 0.7:
            score += 20
        if summary.n_rows > 100:
            score += 10

        report.append(f"Forecasting Suitability Score: {score}/100")

        if score >= 70:
            report.append("VERDICT: Good candidate for forecasting")
        elif score >= 40:
            report.append("VERDICT: May work for forecasting with preprocessing")
        else:
            report.append("VERDICT: Poor candidate - consider other analysis types")

        return "\n".join(report)

    except Exception as e:
        return f"Error analyzing dataset: {e}"


# Export tools list
STAGE1_TOOLS = [
    list_available_datasets,
    profile_dataset,
    save_dataset_summary,
    get_existing_summaries,
    analyze_dataset_for_forecasting,
]
