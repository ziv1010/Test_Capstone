"""
Shared utility functions for the unified agentic AI pipeline.

Contains common functions for file I/O, data profiling, JSON parsing, etc.
"""

from __future__ import annotations

import json
import io
import contextlib
import ast
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
from pandas.api.types import (
    is_numeric_dtype,
    is_integer_dtype,
    is_float_dtype,
    is_bool_dtype,
    is_datetime64_any_dtype,
)

from .config import DATA_DIR, SUMMARIES_DIR, STAGE3B_OUT_DIR


# ===========================
# Type Inference (Stage 1)
# ===========================

def infer_logical_type(series: pd.Series) -> str:
    """Map pandas dtype to a logical type label.
    
    Args:
        series: Pandas Series to analyze
        
    Returns:
        Logical type string
    """
    if is_bool_dtype(series):
        return "boolean"
    if is_datetime64_any_dtype(series):
        return "datetime"
    if is_integer_dtype(series):
        return "integer"
    if is_float_dtype(series):
        return "float"
    if is_numeric_dtype(series):
        return "numeric"
    
    # Treat low-cardinality non-numeric as categorical, else text
    nunique = series.nunique(dropna=True)
    if 0 < nunique <= 50:
        return "categorical"
    return "text"


# ===========================
# Data Profiling (Stage 1)
# ===========================

def profile_csv(path: Path, sample_rows: int = 5000) -> Dict[str, Any]:
    """Read up to `sample_rows` rows and compute basic stats for each column.
    
    Args:
        path: Path to CSV file
        sample_rows: Number of rows to sample
        
    Returns:
        Dictionary with profiling information
    """
    print(f"Profiling {path.name} ...")
    df = pd.read_csv(path, nrows=sample_rows)

    n_rows_sampled = len(df)
    cols_profile = []

    for col in df.columns:
        series = df[col]
        total = len(series)
        non_null = series.notna().sum()
        null_fraction = float(1.0 - non_null / total) if total else 0.0
        
        logical_type = infer_logical_type(series)
        physical_dtype = str(series.dtype)

        # Sample up to 5 distinct non-null values as strings
        examples = (
            series.dropna().astype(str).drop_duplicates().head(5).tolist()
        )

        n_unique = series.nunique(dropna=True)
        unique_fraction = float(n_unique / non_null) if non_null else 0.0

        cols_profile.append(
            {
                "name": col,
                "physical_dtype": physical_dtype,
                "logical_type_guess": logical_type,
                "null_fraction": null_fraction,
                "n_unique": int(n_unique),
                "unique_fraction": unique_fraction,
                "examples": examples,
            }
        )

    profile = {
        "dataset_name": path.name,
        "path": str(path),
        "n_rows_sampled": int(n_rows_sampled),
        "columns": cols_profile,
    }
    return profile


# ===========================
# JSON Parsing
# ===========================

def extract_json_block(text: str) -> str:
    """Extract the first top-level JSON object from a string.
    
    Grabs from the first '{' to the last '}'.
    
    Args:
        text: String potentially containing JSON
        
    Returns:
        Extracted JSON string
        
    Raises:
        ValueError: If no JSON object found
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in LLM output:\n{text[:200]}...")
    return text[start : end + 1]


def parse_tool_call(raw: str) -> Dict[str, Any]:
    """Try to extract a Python dict literal describing a tool call from model output.
    
    Tolerates:
    - Plain dicts: {"tool_name": "...", "tool_args": {...}}
    - Dicts wrapped in markdown code fences
    - Dicts preceded/followed by explanatory text
    
    Args:
        raw: Raw model output
        
    Returns:
        Parsed tool call dictionary
        
    Raises:
        ValueError: If tool call cannot be parsed
    """
    raw = raw.strip()

    # 1) If there is a code fence ```...```, try to isolate content inside it
    if "```" in raw:
        first = raw.find("```")
        second = raw.find("```", first + 3)
        if second != -1:
            block = raw[first + 3 : second].strip()
            # Drop language tag like 'python' on the first line
            first_line, *rest = block.split("\n", 1)
            if first_line.strip().lower().startswith("python") and rest:
                block = rest[0].strip()
            raw_candidate = block
        else:
            raw_candidate = raw
    else:
        raw_candidate = raw

    # 2) Try direct literal_eval on the candidate
    try:
        obj = ast.literal_eval(raw_candidate)
        if isinstance(obj, dict) and "tool_name" in obj:
            return obj
    except Exception:
        pass

    # 3) Fallback: grab from first '{' to last '}' and literal_eval that
    start = raw_candidate.find("{")
    end = raw_candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = raw_candidate[start : end + 1]
        obj = ast.literal_eval(snippet)
        if isinstance(obj, dict) and "tool_name" in obj:
            return obj

    raise ValueError("Could not find a valid tool call dict in model output.")


def parse_proposals_json(raw: str) -> Dict[str, Any]:
    """Extract a strict JSON object with top-level key 'proposals' from model output.
    
    Tolerates:
    - Plain JSON
    - JSON inside ``` ``` fences
    - Extra explanation text before/after
    
    Args:
        raw: Raw model output
        
    Returns:
        Parsed proposals dictionary
        
    Raises:
        ValueError: If proposals JSON cannot be parsed
    """
    raw = raw.strip()

    # 1) If there is a code fence ```...```, try to isolate content inside it
    if "```" in raw:
        first = raw.find("```")
        second = raw.find("```", first + 3)
        if second != -1:
            block = raw[first + 3 : second].strip()
            # Drop language tag like 'json' or 'python' on the first line
            first_line, *rest = block.split("\n", 1)
            if first_line.strip().lower() in ("json", "python") and rest:
                block = rest[0].strip()
            raw_candidate = block
        else:
            raw_candidate = raw
    else:
        raw_candidate = raw

    # 2) Try direct json.loads on the candidate
    try:
        obj = json.loads(raw_candidate)
        if isinstance(obj, dict) and "proposals" in obj:
            return obj
    except Exception:
        pass

    # 3) Fallback: grab from first '{' to last '}' and json.loads that
    start = raw_candidate.find("{")
    end = raw_candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = raw_candidate[start : end + 1]
        obj = json.loads(snippet)
        if isinstance(obj, dict) and "proposals" in obj:
            return obj

    raise ValueError("Could not parse proposals JSON from model output.")


# ===========================
# File I/O Helpers
# ===========================

def list_data_files(extensions: Optional[List[str]] = None) -> List[str]:
    """List available data files in DATA_DIR.
    
    Args:
        extensions: List of extensions to filter (e.g., ['.csv', '.parquet'])
                   If None, defaults to ['.csv', '.tsv', '.parquet', '.feather']
        
    Returns:
        Sorted list of filenames
    """
    if extensions is None:
        extensions = [".csv", ".tsv", ".parquet", ".feather"]
    
    exts = set(ext.lower() for ext in extensions)
    files = [p.name for p in DATA_DIR.iterdir() if p.suffix.lower() in exts]
    return sorted(files)


def list_summary_files() -> List[str]:
    """List all dataset summary JSON files from Stage 1.
    
    Returns:
        List of summary filenames
    """
    return [p.name for p in SUMMARIES_DIR.glob("*.summary.json")]


def read_summary_file(filename: str) -> str:
    """Read a single dataset summary JSON file.
    
    Args:
        filename: Name of the summary file
        
    Returns:
        File contents as string
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    path = SUMMARIES_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"No such summary file: {filename}")
    return path.read_text()


def load_dataframe(
    filepath: Path | str,
    nrows: Optional[int] = None,
    base_dir: Optional[Path] = None
) -> pd.DataFrame:
    """Load a dataframe from various formats.
    
    Args:
        filepath: Path to the file (absolute or relative)
        nrows: Number of rows to load (for CSV/TSV only)
        base_dir: Base directory to resolve relative paths (defaults to DATA_DIR)
        
    Returns:
        Loaded DataFrame
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is not supported
    """
    filepath = Path(filepath)

    # If not absolute, try resolving against provided base_dir, DATA_DIR, then STAGE3B_OUT_DIR
    if not filepath.is_absolute():
        search_roots = []
        if base_dir:
            search_roots.append(base_dir)
        search_roots.extend([DATA_DIR, STAGE3B_OUT_DIR])
        for root in search_roots:
            candidate = root / filepath
            if candidate.exists():
                filepath = candidate
                break
    # As a final fallback, if absolute was provided but doesn't exist, check STAGE3B_OUT_DIR by name
    if not filepath.exists():
        alt = STAGE3B_OUT_DIR / filepath.name
        if alt.exists():
            filepath = alt
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    suffix = filepath.suffix.lower()
    
    if suffix == ".parquet":
        return pd.read_parquet(filepath)
    elif suffix == ".feather":
        return pd.read_feather(filepath)
    elif suffix in [".csv", ".tsv"]:
        if nrows:
            return pd.read_csv(filepath, nrows=nrows)
        return pd.read_csv(filepath)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def inspect_data_file(filename: str, n_rows: int = 10, base_dir: Optional[Path] = None) -> str:
    """Inspect a data file - shows head, dtypes, nulls.
    
    Args:
        filename: Name of the file
        n_rows: Number of rows to show
        base_dir: Base directory (defaults to DATA_DIR)
        
    Returns:
        String with inspection results
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if base_dir is None:
        base_dir = DATA_DIR
        
    df = load_dataframe(filename, nrows=max(n_rows, 100), base_dir=base_dir)
    
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print(f"=== FILE: {filename} ===")
        print(f"Shape: {df.shape}")
        print(f"\nColumns ({len(df.columns)}):")
        for col in df.columns:
            print(f"  - {col}: {df[col].dtype} (nulls: {df[col].isnull().sum()})")
        print(f"\nFirst {min(n_rows, len(df))} rows:")
        print(df.head(n_rows).to_string())
    return buf.getvalue()


# ===========================
# Python Sandbox Execution
# ===========================

def execute_python_sandbox(
    code: str,
    globals_dict: Optional[Dict[str, Any]] = None,
    description: str = "Executing code"
) -> str:
    """Execute Python code in a sandboxed environment.
    
    Args:
        code: Python code to execute
        globals_dict: Global variables to make available
        description: Description of what the code does
        
    Returns:
        Stdout output or error message
    """
    if globals_dict is None:
        globals_dict = {
            "__name__": "__sandbox__",
            "json": json,
            "Path": Path,
            "DATA_DIR": DATA_DIR,
        }
    
    local_env = {}
    buf = io.StringIO()
    
    try:
        with contextlib.redirect_stdout(buf):
            print(f"=== {description} ===")
            exec(code, globals_dict, local_env)
    except Exception as e:
        return f"[ERROR] {e}\n\nTraceback: {e.__class__.__name__}"
    
    output = buf.getvalue()
    return output if output else "[Code executed successfully, no output]"
