"""
Utility Functions for the Conversational AI Pipeline.

This module provides:
- Data profiling and type inference
- File I/O operations
- Python code execution sandbox
- Robust JSON parsing
- DataFrame utilities
"""

import json
import re
import ast
import sys
import traceback
from io import StringIO
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import pandas as pd
import numpy as np

from .config import (
    DATA_DIR, OUTPUT_ROOT, SUMMARIES_DIR, STAGE3B_OUT_DIR,
    STAGE4_OUT_DIR, logger, JSONSanitizer
)
from .models import LogicalDataType, ColumnSummary, DatasetSummary


# ============================================================================
# TYPE INFERENCE
# ============================================================================

def infer_logical_type(series: pd.Series) -> LogicalDataType:
    """
    Infer the logical data type from a pandas Series.

    Handles edge cases and provides more detailed type inference
    than simple dtype checking.
    """
    if series.empty or series.isna().all():
        return LogicalDataType.UNKNOWN

    # Get non-null values for inference
    non_null = series.dropna()
    if len(non_null) == 0:
        return LogicalDataType.UNKNOWN

    dtype = series.dtype

    # Boolean check (before numeric since bool is technically numeric)
    if pd.api.types.is_bool_dtype(dtype):
        return LogicalDataType.BOOLEAN

    # Check for boolean-like values
    if dtype == object:
        unique_vals = set(non_null.astype(str).str.lower().unique())
        if unique_vals.issubset({'true', 'false', 'yes', 'no', '1', '0', 't', 'f', 'y', 'n'}):
            return LogicalDataType.BOOLEAN

    # Datetime check
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return LogicalDataType.DATETIME

    # Try parsing as datetime if object type
    if dtype == object:
        sample = non_null.head(100)
        try:
            parsed = pd.to_datetime(sample, errors='coerce')
            if parsed.notna().mean() > 0.8:
                return LogicalDataType.DATETIME
        except:
            pass

    # Numeric types
    if pd.api.types.is_integer_dtype(dtype):
        return LogicalDataType.INTEGER

    if pd.api.types.is_float_dtype(dtype):
        # Check if actually integers stored as float
        if non_null.apply(lambda x: float(x).is_integer()).all():
            return LogicalDataType.INTEGER
        return LogicalDataType.FLOAT

    if pd.api.types.is_numeric_dtype(dtype):
        return LogicalDataType.NUMERIC

    # Try converting to numeric
    if dtype == object:
        try:
            numeric = pd.to_numeric(non_null, errors='coerce')
            if numeric.notna().mean() > 0.8:
                if (numeric.dropna() % 1 == 0).all():
                    return LogicalDataType.INTEGER
                return LogicalDataType.FLOAT
        except:
            pass

    # Categorical vs Text
    if dtype == object or dtype.name == 'category':
        unique_ratio = len(non_null.unique()) / len(non_null)
        avg_length = non_null.astype(str).str.len().mean()

        # Categorical: few unique values or short strings
        if unique_ratio < 0.05 or (unique_ratio < 0.2 and avg_length < 50):
            return LogicalDataType.CATEGORICAL

        # Text: many unique values and longer strings
        if avg_length > 100:
            return LogicalDataType.TEXT

        return LogicalDataType.CATEGORICAL

    return LogicalDataType.UNKNOWN


def get_column_statistics(series: pd.Series, logical_type: LogicalDataType) -> Dict[str, Any]:
    """Get statistics appropriate for the column type."""
    stats = {}

    if series.empty or series.isna().all():
        return stats

    non_null = series.dropna()

    if logical_type in [LogicalDataType.INTEGER, LogicalDataType.FLOAT, LogicalDataType.NUMERIC]:
        try:
            numeric = pd.to_numeric(non_null, errors='coerce').dropna()
            if len(numeric) > 0:
                stats['min_value'] = float(numeric.min())
                stats['max_value'] = float(numeric.max())
                stats['mean_value'] = float(numeric.mean())
                stats['std_value'] = float(numeric.std()) if len(numeric) > 1 else 0.0
        except:
            pass

    elif logical_type == LogicalDataType.DATETIME:
        try:
            dates = pd.to_datetime(non_null, errors='coerce').dropna()
            if len(dates) > 0:
                stats['min_date'] = str(dates.min())
                stats['max_date'] = str(dates.max())
                # Try to infer frequency
                if len(dates) > 2:
                    diffs = dates.sort_values().diff().dropna()
                    median_diff = diffs.median()
                    if median_diff.days == 1:
                        stats['date_frequency'] = 'daily'
                    elif median_diff.days == 7:
                        stats['date_frequency'] = 'weekly'
                    elif 28 <= median_diff.days <= 31:
                        stats['date_frequency'] = 'monthly'
                    elif 365 <= median_diff.days <= 366:
                        stats['date_frequency'] = 'yearly'
        except:
            pass

    return stats


# ============================================================================
# DATA PROFILING
# ============================================================================

def profile_csv(filepath: Union[str, Path], sample_rows: int = 5000) -> DatasetSummary:
    """
    Profile a CSV file and return a comprehensive summary.

    Args:
        filepath: Path to CSV file
        sample_rows: Maximum rows to sample for profiling

    Returns:
        DatasetSummary object with column metadata
    """
    filepath = Path(filepath)

    # Read file
    try:
        df = pd.read_csv(filepath, nrows=sample_rows, low_memory=False)
    except Exception as e:
        logger.error(f"Failed to read {filepath}: {e}")
        raise

    # Get file size
    file_size_mb = filepath.stat().st_size / (1024 * 1024)

    # Profile each column
    columns = []
    datetime_cols = []
    target_candidates = []

    for col_name in df.columns:
        series = df[col_name]
        logical_type = infer_logical_type(series)

        # Get basic stats
        n_null = series.isna().sum()
        n_total = len(series)
        null_fraction = n_null / n_total if n_total > 0 else 0

        n_unique = series.nunique()
        unique_fraction = n_unique / n_total if n_total > 0 else 0

        # Get examples (non-null values)
        examples = series.dropna().head(5).tolist()

        # Get type-specific statistics
        stats = get_column_statistics(series, logical_type)

        col_summary = ColumnSummary(
            name=col_name,
            dtype=str(series.dtype),
            logical_type=logical_type,
            null_fraction=null_fraction,
            unique_fraction=unique_fraction,
            n_unique=n_unique,
            examples=examples,
            **stats
        )
        columns.append(col_summary)

        # Track datetime columns
        if logical_type == LogicalDataType.DATETIME:
            datetime_cols.append(col_name)

        # Track potential target columns (numeric with reasonable unique values)
        if logical_type in [LogicalDataType.INTEGER, LogicalDataType.FLOAT, LogicalDataType.NUMERIC]:
            if 0.01 < unique_fraction < 0.99 and null_fraction < 0.3:
                target_candidates.append(col_name)

    # Identify candidate primary keys (high uniqueness, low nulls)
    candidate_keys = [
        c.name for c in columns
        if c.unique_fraction > 0.95 and c.null_fraction < 0.01
    ]

    # Calculate data quality score
    avg_null_fraction = np.mean([c.null_fraction for c in columns])
    data_quality_score = 1.0 - avg_null_fraction

    return DatasetSummary(
        filename=filepath.name,
        filepath=str(filepath),
        n_rows=len(df),
        n_cols=len(df.columns),
        columns=columns,
        candidate_keys=candidate_keys,
        file_size_mb=file_size_mb,
        has_datetime_column=len(datetime_cols) > 0,
        has_target_candidates=target_candidates[:5],  # Top 5
        data_quality_score=data_quality_score
    )


def profile_all_datasets(data_dir: Path = None) -> List[DatasetSummary]:
    """Profile all CSV files in a directory."""
    data_dir = Path(data_dir or DATA_DIR)
    summaries = []

    for filepath in data_dir.glob("*.csv"):
        try:
            summary = profile_csv(filepath)
            summaries.append(summary)
            logger.info(f"Profiled: {filepath.name}")
        except Exception as e:
            logger.error(f"Failed to profile {filepath.name}: {e}")

    return summaries


# ============================================================================
# FILE I/O OPERATIONS
# ============================================================================

def list_data_files(data_dir: Path = None, extensions: List[str] = None) -> List[str]:
    """List all data files in the data directory."""
    data_dir = Path(data_dir or DATA_DIR)
    extensions = extensions or ['.csv', '.tsv', '.parquet', '.feather']

    files = []
    for ext in extensions:
        files.extend([f.name for f in data_dir.glob(f"*{ext}")])

    return sorted(files)


def list_summary_files(summaries_dir: Path = None) -> List[str]:
    """List all summary JSON files."""
    summaries_dir = Path(summaries_dir or SUMMARIES_DIR)
    return sorted([f.name for f in summaries_dir.glob("*.summary.json")])


def read_summary_file(filename: str, summaries_dir: Path = None) -> Dict[str, Any]:
    """Read a summary JSON file."""
    summaries_dir = Path(summaries_dir or SUMMARIES_DIR)
    filepath = summaries_dir / filename

    if not filepath.exists():
        raise FileNotFoundError(f"Summary file not found: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_dataframe(
    filepath: Union[str, Path],
    nrows: Optional[int] = None,
    base_dirs: List[Path] = None
) -> pd.DataFrame:
    """
    Load a DataFrame from various file formats.

    Searches multiple directories if file not found at given path.
    """
    filepath = Path(filepath)

    # If absolute path exists, use it directly
    if filepath.is_absolute() and filepath.exists():
        return _read_dataframe(filepath, nrows)

    # Search in base directories
    base_dirs = base_dirs or [DATA_DIR, STAGE3B_OUT_DIR, STAGE4_OUT_DIR]

    for base_dir in base_dirs:
        full_path = Path(base_dir) / filepath
        if full_path.exists():
            return _read_dataframe(full_path, nrows)

    raise FileNotFoundError(f"File not found: {filepath}")


def _read_dataframe(filepath: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    """Read DataFrame based on file extension."""
    suffix = filepath.suffix.lower()

    if suffix == '.csv':
        return pd.read_csv(filepath, nrows=nrows, low_memory=False)
    elif suffix == '.tsv':
        return pd.read_csv(filepath, sep='\t', nrows=nrows, low_memory=False)
    elif suffix == '.parquet':
        df = pd.read_parquet(filepath)
        if nrows:
            df = df.head(nrows)
        return df
    elif suffix == '.feather':
        df = pd.read_feather(filepath)
        if nrows:
            df = df.head(nrows)
        return df
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def inspect_data_file(
    filename: str,
    n_rows: int = 10,
    base_dir: Path = None
) -> str:
    """
    Inspect a data file and return a formatted summary.

    Returns string with shape, columns, dtypes, nulls, and sample rows.
    """
    base_dir = Path(base_dir or DATA_DIR)
    filepath = base_dir / filename

    if not filepath.exists():
        # Try other directories
        for alt_dir in [STAGE3B_OUT_DIR, STAGE4_OUT_DIR]:
            alt_path = Path(alt_dir) / filename
            if alt_path.exists():
                filepath = alt_path
                break
        else:
            return f"File not found: {filename}"

    try:
        df = load_dataframe(filepath, nrows=1000)

        lines = [
            f"File: {filepath.name}",
            f"Shape: {df.shape[0]} rows x {df.shape[1]} columns",
            "",
            "Columns and Types:",
        ]

        for col in df.columns:
            null_count = df[col].isna().sum()
            lines.append(f"  - {col}: {df[col].dtype} ({null_count} nulls)")

        lines.append("")
        lines.append(f"Sample Rows (first {min(n_rows, len(df))}):")
        lines.append(df.head(n_rows).to_string())

        return "\n".join(lines)

    except Exception as e:
        return f"Error inspecting {filename}: {e}"


# ============================================================================
# PYTHON SANDBOX EXECUTION
# ============================================================================

class PythonSandbox:
    """
    Safe Python code execution environment.

    Provides:
    - Isolated namespace with common libraries
    - Output capture
    - Error handling
    - Timeout support (via external mechanism)
    """

    DEFAULT_IMPORTS = {
        'pd': pd,
        'np': np,
        'json': json,
        'Path': Path,
        'datetime': datetime,
        'DATA_DIR': DATA_DIR,
        'OUTPUT_ROOT': OUTPUT_ROOT,
        'STAGE3B_OUT_DIR': STAGE3B_OUT_DIR,
        'STAGE4_OUT_DIR': STAGE4_OUT_DIR,
    }

    @classmethod
    def execute(
        cls,
        code: str,
        additional_globals: Dict[str, Any] = None,
        description: str = ""
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Execute Python code in a sandboxed environment.

        Args:
            code: Python code to execute
            additional_globals: Additional variables to include in namespace
            description: Description of what the code does

        Returns:
            Tuple of (success, output_string, namespace_dict)
        """
        # Build namespace
        namespace = dict(cls.DEFAULT_IMPORTS)
        namespace['load_dataframe'] = load_dataframe

        if additional_globals:
            namespace.update(additional_globals)

        # Capture stdout
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()

        success = True
        output = ""

        try:
            # Execute code
            exec(code, namespace)
            output = sys.stdout.getvalue()
            stderr = sys.stderr.getvalue()
            if stderr:
                output += f"\n[STDERR]\n{stderr}"

        except Exception as e:
            success = False
            output = f"Error: {type(e).__name__}: {e}\n"
            output += traceback.format_exc()

        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        return success, output, namespace


def execute_python_sandbox(
    code: str,
    globals_dict: Dict[str, Any] = None,
    description: str = ""
) -> str:
    """
    Execute Python code and return output string.

    Convenience wrapper around PythonSandbox.execute().
    """
    success, output, _ = PythonSandbox.execute(code, globals_dict, description)

    if success:
        return output if output else "Code executed successfully (no output)"
    else:
        return output


# ============================================================================
# ROBUST JSON PARSING
# ============================================================================

def extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON from text with multiple fallback strategies.

    Uses JSONSanitizer for robust extraction.
    """
    return JSONSanitizer.extract_json(text)


def parse_tool_call(raw: str) -> Optional[Dict[str, Any]]:
    """
    Parse a tool call from LLM output.

    Handles various formats:
    - Direct JSON
    - Markdown code blocks
    - Python dict literals
    """
    if not raw:
        return None

    # Try JSON extraction first
    result = extract_json_block(raw)
    if result:
        return result

    # Try Python literal eval
    try:
        # Find dict-like structure
        match = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
        if match:
            return ast.literal_eval(match.group())
    except:
        pass

    return None


def parse_proposals_json(raw: str) -> Optional[Dict[str, Any]]:
    """
    Parse task proposals JSON from LLM output.

    Specifically handles the {"proposals": [...]} format.
    """
    result = extract_json_block(raw)

    if result:
        # Normalize key names
        if 'proposals' in result:
            return result
        elif 'tasks' in result:
            result['proposals'] = result.pop('tasks')
            return result
        elif isinstance(result, list):
            return {'proposals': result}

    return None


def safe_json_dumps(obj: Any, **kwargs) -> str:
    """
    Safely convert object to JSON string.

    Handles non-serializable types gracefully.
    """
    def default_handler(o):
        if isinstance(o, datetime):
            return o.isoformat()
        elif isinstance(o, Path):
            return str(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, (np.integer, np.floating)):
            return float(o)
        elif hasattr(o, 'to_dict'):
            return o.to_dict()
        elif hasattr(o, '__dict__'):
            return o.__dict__
        else:
            return str(o)

    kwargs.setdefault('default', default_handler)
    kwargs.setdefault('indent', 2)
    kwargs.setdefault('ensure_ascii', False)

    return json.dumps(obj, **kwargs)


# ============================================================================
# DATAFRAME UTILITIES
# ============================================================================

def validate_dataframe(
    df: pd.DataFrame,
    required_columns: List[str] = None,
    max_null_fraction: float = 0.35
) -> Tuple[bool, List[str]]:
    """
    Validate a DataFrame for pipeline requirements.

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []

    # Check required columns
    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            issues.append(f"Missing required columns: {missing}")

    # Check for excessive nulls
    for col in df.columns:
        null_frac = df[col].isna().mean()
        if null_frac > max_null_fraction:
            issues.append(f"Column '{col}' has {null_frac:.1%} nulls (max: {max_null_fraction:.1%})")

    # Check for empty DataFrame
    if len(df) == 0:
        issues.append("DataFrame is empty")

    return len(issues) == 0, issues


def prepare_dataframe_for_modeling(
    df: pd.DataFrame,
    target_column: str,
    date_column: Optional[str] = None,
    drop_columns: List[str] = None
) -> pd.DataFrame:
    """
    Prepare a DataFrame for modeling.

    - Handles missing values
    - Converts datetime columns
    - Drops specified columns
    - Ensures target column is present
    """
    df = df.copy()

    # Drop specified columns
    if drop_columns:
        df = df.drop(columns=[c for c in drop_columns if c in df.columns], errors='ignore')

    # Ensure target exists
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found")

    # Parse date column if specified
    if date_column and date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        df = df.sort_values(date_column)

    # Handle missing values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            # Use median imputation
            df[col] = df[col].fillna(df[col].median())

    # Handle missing values in categorical columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        if col != date_column and df[col].isna().any():
            df[col] = df[col].fillna('UNKNOWN')

    # Drop remaining rows with nulls in target
    df = df.dropna(subset=[target_column])

    return df


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Type inference
    "infer_logical_type", "get_column_statistics",
    # Profiling
    "profile_csv", "profile_all_datasets",
    # File I/O
    "list_data_files", "list_summary_files", "read_summary_file",
    "load_dataframe", "inspect_data_file",
    # Sandbox
    "PythonSandbox", "execute_python_sandbox",
    # JSON
    "extract_json_block", "parse_tool_call", "parse_proposals_json", "safe_json_dumps",
    # DataFrame
    "validate_dataframe", "prepare_dataframe_for_modeling",
]
