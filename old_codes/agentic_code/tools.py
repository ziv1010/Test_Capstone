"""
Centralized tool definitions for all stages of the agentic AI pipeline.

Tools are organized by stage and can be imported individually or as groups.
"""

from __future__ import annotations

import json
import io
import contextlib
import re
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np

from langchain_core.tools import tool

from .config import (
    PROJECT_ROOT,
    OUTPUT_ROOT,
    DATA_DIR,
    SUMMARIES_DIR,
    STAGE2_OUT_DIR,
    STAGE3_OUT_DIR,
    STAGE3B_OUT_DIR,
    STAGE3_5_OUT_DIR,
    STAGE4_OUT_DIR,
    STAGE5_OUT_DIR,
    STAGE4_WORKSPACE,
    STAGE4_WORKSPACE,
    STAGE5_WORKSPACE,
    STAGE_FILE_PATHS,
    FILE_NAMING_PATTERNS,
)
from .models import Stage2Output, Stage3Plan, ExecutionResult
from .utils import (
    list_summary_files as _list_summary_files,
    read_summary_file as _read_summary_file,
    list_data_files as _list_data_files,
    inspect_data_file as _inspect_data_file,
    load_dataframe,
)

# Shared scratch slot to make the last tool result available to the sandbox
LAST_TOOL_RESULT: Any = None

# ===========================
# Generic / Failsafe Tools
# ===========================

@tool
def failsafe_python(code: str, description: str = "Failsafe scratchpad") -> str:
    """Execute arbitrary Python for diagnostics/debugging.
    
    Environment includes: pd, np, json, Path, DATA_DIR, OUTPUT_ROOT, PROJECT_ROOT,
    load_dataframe(filename, nrows=None), and print output is returned.
    """
    def load_dataframe_helper(filename: str, nrows: Optional[int] = None):
        return load_dataframe(filename, nrows=nrows, base_dir=DATA_DIR)

    globals_dict = {
        "__name__": "__failsafe_scratch__",
        "pd": pd,
        "np": np,
        "json": json,
        "Path": Path,
        "DATA_DIR": DATA_DIR,
        "OUTPUT_ROOT": OUTPUT_ROOT,
        "PROJECT_ROOT": PROJECT_ROOT,
        "load_dataframe": load_dataframe_helper,
        "description": description,
    }

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            print(f"=== {description} ===")
            exec(code, globals_dict, globals_dict)
    except Exception as e:
        import traceback
        return f"[failsafe_python error] {e}\n{traceback.format_exc()}"

    return buf.getvalue() or "[failsafe_python done]"


@tool
def search(
    query: str,
    within: str = "project",
    file_glob: str = "**/*",
    max_matches: int = 30,
    case_sensitive: bool = False,
) -> str:
    """Search workspace text files for a pattern (regex supported).
    
    Args:
        query: Pattern to search for (regex).
        within: One of 'project', 'output', 'code', 'data', 'all'. Defaults to 'project'.
        file_glob: Glob filter for files (e.g., '*.json', '*.log').
        max_matches: Maximum number of line-level matches to return.
        case_sensitive: Whether the search is case sensitive.
        
    Returns:
        Matched lines with file paths and line numbers, or a message if none found.
    """
    root_map = {
        "project": PROJECT_ROOT,
        "output": OUTPUT_ROOT,
        "code": PROJECT_ROOT / "final_code",
        "data": DATA_DIR,
        "all": PROJECT_ROOT,
    }
    root = root_map.get(within, PROJECT_ROOT)

    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        pattern = re.compile(query, flags)
    except re.error as e:
        return f"[search error] Invalid regex: {e}"

    hits: List[str] = []
    for path in root.rglob(file_glob):
        if not path.is_file():
            continue
        try:
            if path.stat().st_size > 2_000_000:  # avoid huge files
                continue
        except OSError:
            continue

        try:
            with path.open("r", errors="ignore") as f:
                for lineno, line in enumerate(f, start=1):
                    if pattern.search(line):
                        rel = path.relative_to(root)
                        hits.append(f"{rel}:{lineno}: {line.strip()}")
                        if len(hits) >= max_matches:
                            return "\n".join(hits)
        except (OSError, UnicodeDecodeError):
            continue

    return "\n".join(hits) if hits else "[search] No matches found."

# ===========================
# Stage 2: Task Proposal Tools
# ===========================

@tool
def list_summary_files() -> List[str]:
    """List all dataset summary JSON files available from Stage 1.
    
    Returns filenames (not full paths).
    """
    return _list_summary_files()


@tool
def read_summary_file(filename: str) -> str:
    """Read a single dataset summary JSON file and return its contents as a string."""
    return _read_summary_file(filename)


@tool
def python_sandbox(code: str) -> str:
    """Execute arbitrary Python code to help analyze dataset summaries and design tasks.
    
    The code can:
    - import standard libraries like json, math, statistics, pandas
    - access PROJECT_ROOT, DATA_DIR, SUMMARIES_DIR
    - call read_summary_file('<summary-filename>')
    - call list_summary_files() (sandbox helper)
    - open and inspect files directly
    - print intermediate results
    
    Returns whatever is printed to stdout (or an error string).
    """
    def _read_summary_file_py(filename: str) -> str:
        """Helper for sandbox: read a summary file as text."""
        path = SUMMARIES_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"No such summary file: {filename}")
        return path.read_text()
    
    def _list_summary_files_py() -> List[str]:
        return [p.name for p in SUMMARIES_DIR.glob("*.summary.json")]
    
    PYTHON_GLOBALS: Dict[str, Any] = {
        "__name__": "__agent_python__",
        "json": json,
        "Path": Path,
        "DATA_DIR": DATA_DIR,
        "SUMMARIES_DIR": SUMMARIES_DIR,
        "read_summary_file": _read_summary_file_py,
        "list_summary_files": _list_summary_files_py,
        # Make prior tool outputs accessible for LLM convenience
        "result": LAST_TOOL_RESULT,
        "last_result": LAST_TOOL_RESULT,
        "last_tool_result": LAST_TOOL_RESULT,
    }
    
    local_env: Dict[str, Any] = {}
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, PYTHON_GLOBALS, local_env)
    except Exception as e:
        return f"[python_sandbox error] {e}"
    return buf.getvalue() or "[python_sandbox done]"


# Stage 2 tool list
STAGE2_TOOLS = [list_summary_files, read_summary_file, python_sandbox, search]


# =========================== 
# Stage 3: Planning Tools
# ===========================

@tool
def load_task_proposal(task_id: str) -> str:
    """Load a single TaskProposal by ID."""
    path = STAGE2_OUT_DIR / "task_proposals.json"
    if not path.exists():
        raise FileNotFoundError(f"Could not find task_proposals.json in {STAGE2_OUT_DIR}")
    
    raw = path.read_text()
    data = json.loads(raw)
    stage2 = Stage2Output.model_validate(data)
    for p in stage2.proposals:
        if p.id == task_id:
            return p.model_dump_json(indent=2)
    raise ValueError(f"No TaskProposal with id={task_id!r} found.")


@tool
def list_data_files() -> List[str]:
    """List available data files."""
    return _list_data_files()


@tool
def inspect_data_file(filename: str, n_rows: int = 10) -> str:
    """Inspect a data file - shows head, dtypes, nulls."""
    return _inspect_data_file(filename, n_rows)


@tool
def python_sandbox_stage3(code: str) -> str:
    """Execute Python code for data exploration in Stage 3."""
    def load_dataframe_helper(filename: str, n_rows: Optional[int] = None):
        return load_dataframe(filename, nrows=n_rows, base_dir=DATA_DIR)
    
    globals_dict = {
        "__name__": "__stage3_sandbox__",
        "pd": pd,
        "json": json,
        "Path": Path,
        "DATA_DIR": DATA_DIR,
        "load_dataframe": load_dataframe_helper,
    }
    
    local_env = {}
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, globals_dict, local_env)
    except Exception as e:
        return f"[ERROR] {e}"
    return buf.getvalue() or "[No output]"


@tool
def save_stage3_plan(plan_json: str) -> str:
    """Validate and save a Stage3Plan.
    
    Args:
        plan_json: Complete JSON string of Stage3Plan
        
    Returns:
        Success message with path, or raises ValueError
    """
    # Parse JSON (with a light sanitization pass for invalid escapes)
    invalid_escape = re.compile(r'\\([^"\\/bfnrtu])')
    sanitized_payload = invalid_escape.sub(r"\1", plan_json)
    try:
        raw_obj = json.loads(sanitized_payload)
    except json.JSONDecodeError as e:
        debug_path = STAGE3_OUT_DIR / "failed_stage3_plan.json"
        debug_path.write_text(plan_json)
        start = max(e.pos - 40, 0)
        end = min(e.pos + 40, len(plan_json))
        snippet = plan_json[start:end]
        raise ValueError(
            f"Invalid JSON: {e}. Saved raw payload to {debug_path}. "
            f"Context: {snippet}"
        ) from e

    # Schema validation
    try:
        plan = Stage3Plan.model_validate(raw_obj)
    except Exception as e:
        raise ValueError(f"Schema validation failed: {e}") from e

    # Validate files exist
    available_files = set(_list_data_files())
    for fi in plan.file_instructions:
        if fi.original_name not in available_files:
            raise ValueError(f"File {fi.original_name!r} not found in DATA_DIR")

    # For join steps, do basic validation
    if plan.join_steps:
        file_cache = {}
        
        # Load files
        for fi in plan.file_instructions:
            try:
                df = load_dataframe(fi.original_name, nrows=100, base_dir=DATA_DIR)
                file_cache[fi.alias] = df
            except Exception as e:
                raise ValueError(f"Failed to load {fi.original_name}: {e}") from e
        
        # Validate joins
        for idx, js in enumerate(plan.join_steps):
            if js.join_type == "base":
                if js.left_table not in file_cache:
                    raise ValueError(f"Base table {js.left_table!r} not found")
                continue
                
            if js.left_table not in file_cache:
                raise ValueError(f"Left table {js.left_table!r} not found")
            if js.right_table and js.right_table not in file_cache:
                raise ValueError(f"Right table {js.right_table!r} not found")
            
            # Check keys exist
            df_left = file_cache[js.left_table]
            df_right = file_cache.get(js.right_table) if js.right_table else None
            
            # Case 1: Equijoin with join_keys
            if js.join_keys:
                missing_left = [k for k in js.join_keys if k not in df_left.columns]
                if missing_left:
                    raise ValueError(f"Join {idx}: keys {missing_left} missing in {js.left_table}")
                
                if df_right is not None:
                    missing_right = [k for k in js.join_keys if k not in df_right.columns]
                    if missing_right:
                        raise ValueError(f"Join {idx}: keys {missing_right} missing in {js.right_table}")

            # Case 2: Different keys with left_on/right_on
            if js.left_on:
                missing_left = [k for k in js.left_on if k not in df_left.columns]
                if missing_left:
                    raise ValueError(f"Join {idx}: left_on keys {missing_left} missing in {js.left_table}")

            if js.right_on and df_right is not None:
                missing_right = [k for k in js.right_on if k not in df_right.columns]
                if missing_right:
                    raise ValueError(f"Join {idx}: right_on keys {missing_right} missing in {js.right_table}")
            
            # Ensure at least one join condition
            if not js.join_keys and not (js.left_on and js.right_on):
                raise ValueError(f"Join {idx}: Must specify either join_keys OR (left_on and right_on)")

    # Save
    out_path = STAGE3_OUT_DIR / f"{plan.plan_id}.json"
    out_path.write_text(plan.model_dump_json(indent=2))
    
    return f"✅ Plan saved successfully to: {out_path}"


# Stage 3 tool list
STAGE3_TOOLS = [
    load_task_proposal,
    list_data_files,
    inspect_data_file,
    search,
    python_sandbox_stage3,
    save_stage3_plan,
]


# ===========================
# Stage 3B: Data Preparation Tools
# ===========================

@tool
def load_stage3_plan_for_prep(plan_id: str) -> str:
    """Load a Stage 3 plan for data preparation.
    
    Args:
        plan_id: Plan identifier (e.g., 'PLAN-TSK-001')
        
    Returns:
        JSON string of the execution plan
    """
    from .config import STAGE3_OUT_DIR
    
    plan_path = STAGE3_OUT_DIR / f"{plan_id}.json"
    if not plan_path.exists():
        matches = list(STAGE3_OUT_DIR.glob(f"*{plan_id}*.json"))
        if not matches:
            raise FileNotFoundError(f"No plan found matching: {plan_id}")
        plan_path = matches[0]
    
    return plan_path.read_text()


@tool
def python_sandbox_stage3b(code: str) -> str:
    """Execute Python code in a sandbox for Stage 3B data exploration.
    
    Args:
        code: Python code to execute. Can access load_dataframe(), DATA_DIR, etc.
        
    Returns:
        Output from the code execution
    """
    from .config import DATA_DIR, STAGE3B_OUT_DIR
    import pandas as pd
    import numpy as np
    from pathlib import Path
    from io import StringIO
    import sys
    import os
    import traceback
    
    # Helper to resolve files relative to DATA_DIR when a bare filename is provided
    def _resolve_path(file: str) -> Path:
        path = Path(file)
        # Try the provided path, then DATA_DIR / <file>, then DATA_DIR / <name>
        candidates = [path]
        if not path.is_absolute():
            candidates.extend([
                DATA_DIR / file,
                STAGE3B_OUT_DIR / file,
            ])
        else:
            candidates.extend([
                DATA_DIR / path.name,
                STAGE3B_OUT_DIR / path.name,
            ])
        for cand in candidates:
            if cand.exists():
                return cand
        
        # Robust lookup: Try globbing if exact match fails
        name = path.name
        for search_dir in [DATA_DIR, STAGE3B_OUT_DIR]:
            # 1. Try exact name glob
            matches = list(search_dir.glob(f"*{name}*"))
            if matches:
                return matches[0]
            
            # 2. If name contains TSK-XXX but not PLAN-, try adding PLAN- (via wildcard)
            if "TSK-" in name and "PLAN-" not in name:
                pattern = name.replace("TSK-", "*TSK-")
                matches = list(search_dir.glob(pattern))
                if matches:
                    return matches[0]
                    
        return path

    # Safe CSV reader that falls back to DATA_DIR if needed
    _pd_read_csv = pd.read_csv
    def _safe_read_csv(file, *args, **kwargs):
        target = _resolve_path(file)
        if target.exists():
            return _pd_read_csv(target, *args, **kwargs)
        return _pd_read_csv(file, *args, **kwargs)

    def _safe_load_dataframe(file, **kwargs):
        target = _resolve_path(file)
        if target.suffix in {".parquet", ".parq"}:
            return pd.read_parquet(target, **kwargs)
        return _safe_read_csv(target, **kwargs)
    
    # Inject helpers and monkeypatch read_csv so agent code using pd.read_csv works
    pd.read_csv = _safe_read_csv
    
    # Setup environment
    local_env = {
        "pd": pd,
        "np": np,
        "DATA_DIR": DATA_DIR,
        "STAGE3B_OUT_DIR": STAGE3B_OUT_DIR,
        "Path": Path,
        "load_dataframe": _safe_load_dataframe,
    }
    
    # Capture output
    old_stdout = sys.stdout
    sys.stdout = captured = StringIO()
    old_cwd = os.getcwd()
    
    try:
        exec(code, local_env)
        output = captured.getvalue()
        return output if output else "[No output]"
    except Exception as e:
        return f"[ERROR] {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
    finally:
        pd.read_csv = _pd_read_csv
        os.chdir(old_cwd)
        sys.stdout = old_stdout


@tool
def run_data_prep_code(code: str, description: str) -> str:
    """Execute data preparation code and return results.
    
    This is the main tool for data loading, merging, filtering, and transformation.
    The code should save the prepared DataFrame to STAGE3B_OUT_DIR.
    
    Args:
        code: Python code for data preparation
        description: Brief description of what this code does
        
Returns:
        Execution status and preview of prepared data
    """
    from .config import DATA_DIR, STAGE3B_OUT_DIR
    import pandas as pd
    import numpy as np
    from pathlib import Path
    from io import StringIO
    import sys
    import os
    import traceback
    
    print(f"\n=== Running data preparation: {description} ===")

    # Helper to resolve files relative to DATA_DIR when a bare filename is provided
    def _resolve_path(file: str) -> Path:
        path = Path(file)
        candidates = [path]
        if not path.is_absolute():
            candidates.extend([
                DATA_DIR / file,
                STAGE3B_OUT_DIR / file,
            ])
        else:
            candidates.extend([
                DATA_DIR / path.name,
                STAGE3B_OUT_DIR / path.name,
            ])
        for cand in candidates:
            if cand.exists():
                return cand
                
        # Robust lookup: Try globbing if exact match fails
        name = path.name
        for search_dir in [DATA_DIR, STAGE3B_OUT_DIR]:
            # 1. Try exact name glob
            matches = list(search_dir.glob(f"*{name}*"))
            if matches:
                return matches[0]
            
            # 2. If name contains TSK-XXX but not PLAN-, try adding PLAN- (via wildcard)
            if "TSK-" in name and "PLAN-" not in name:
                pattern = name.replace("TSK-", "*TSK-")
                matches = list(search_dir.glob(pattern))
                if matches:
                    return matches[0]

        return path

    _pd_read_csv = pd.read_csv
    def _safe_read_csv(file, *args, **kwargs):
        target = _resolve_path(file)
        if target.exists():
            return _pd_read_csv(target, *args, **kwargs)
        return _pd_read_csv(file, *args, **kwargs)

    def _safe_load_dataframe(file, **kwargs):
        target = _resolve_path(file)
        if target.suffix in {".parquet", ".parq"}:
            return pd.read_parquet(target, **kwargs)
        return _safe_read_csv(target, **kwargs)

    # Monkeypatch pandas.read_csv so agent code using pd.read_csv(...) works
    pd.read_csv = _safe_read_csv
    
    # Setup environment with helper functions
    local_env = {
        "pd": pd,
        "np": np,
        "DATA_DIR": DATA_DIR,
        "STAGE3B_OUT_DIR": STAGE3B_OUT_DIR,
        "Path": Path,
        "load_dataframe": _safe_load_dataframe,
    }
    
    # Capture output
    old_stdout = sys.stdout
    sys.stdout = captured = StringIO()
    old_cwd = os.getcwd()
    
    try:
        # Run prep code inside STAGE3B_OUT_DIR so relative writes land there
        os.chdir(STAGE3B_OUT_DIR)
        exec(code, local_env)
        output = captured.getvalue()
        
        # Check if prepared_df was created
        if "prepared_df" in local_env:
            df = local_env["prepared_df"]
            preview = f"\n\nPrepared DataFrame Info:\n"
            preview += f"  Shape: {df.shape}\n"
            preview += f"  Columns: {list(df.columns)}\n"
            preview += f"\nFirst 3 rows:\n{df.head(3).to_string()}\n"
            return output + preview
        else:
            return output if output else "[Code executed successfully, no output]"
            
    except Exception as e:
        error_msg = f"[ERROR] {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(error_msg, file=sys.stderr)
        return error_msg
    finally:
        pd.read_csv = _pd_read_csv
        os.chdir(old_cwd)
        sys.stdout = old_stdout


@tool
def save_prepared_data(
    plan_id: str,
    prepared_file_name: str,
    original_row_count: int,
    prepared_row_count: int,
    columns_created: List[str],
    transformations_applied: List[str],
    data_quality_report: Dict[str, Any]
) -> str:
    """Save prepared data output for Stage 3B.
    
    This should be called after successfully preparing the data.
    The prepared DataFrame should already be saved as parquet/csv.
    
    Args:
        plan_id: Plan ID (e.g., 'PLAN-TSK-001')
        prepared_file_name: Name of the saved prepared data file
        original_row_count: Number of rows before preparation
        prepared_row_count: Number of rows after preparation
        columns_created: List of feature engineering columns created
        transformations_applied: List of transformations (filters, joins, etc.)
        data_quality_report: Data quality metrics
        
    Returns:
        Success message with file path
    """
    from .config import STAGE3B_OUT_DIR
    from .models import PreparedDataOutput
    import json
    from datetime import datetime
    
    output = PreparedDataOutput(
        plan_id=plan_id,
        prepared_file_path=str(STAGE3B_OUT_DIR / prepared_file_name),
        original_row_count=original_row_count,
        prepared_row_count=prepared_row_count,
        columns_created=columns_created,
        transformations_applied=transformations_applied,
        data_quality_report=data_quality_report,
    )
    
    # Save metadata
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = STAGE3B_OUT_DIR / f"prep_{plan_id}_{timestamp}.json"
    output_path.write_text(output.model_dump_json(indent=2))
    
    return f"✅ Prepared data output saved to: {output_path}\n\nsaved::prep_{plan_id}"


# ===========================
# Stage 3.5: Method Testing & Benchmarking Tools
# ===========================

@tool
def load_stage3_plan_for_tester(plan_id: str) -> str:
    """Load a Stage 3 plan for method testing.
    
    Args:
        plan_id: Plan identifier (e.g., 'PLAN-TSK-001')
        
    Returns:
        JSON string of the plan
    """
    from .config import STAGE3_OUT_DIR
    
    plan_path = STAGE3_OUT_DIR / f"{plan_id}.json"
    if not plan_path.exists():
        # Try finding by pattern
        matches = list(STAGE3_OUT_DIR.glob(f"*{plan_id}*.json"))
        if not matches:
            raise FileNotFoundError(f"No plan found matching: {plan_id}")
        plan_path = matches[0]
    
    return plan_path.read_text()

@tool
def run_benchmark_code(code: str, description: str = "Running benchmark") -> str:
    """Execute Python code for benchmarking forecasting methods.

    The caller is responsible for every modeling choice.
    The code you provide should:
    - Choose and implement the forecasting methods (nothing is pre-selected)
    - Import any trusted libraries it needs, handling missing packages gracefully
    - Design the time-based train/validation/test splits
    - Compute task-appropriate metrics of your choosing
    - Optionally save artifacts (plots, models, CSVs) under STAGE3_5_OUT_DIR

    Predefined objects in the execution environment:
    - pd, np
    - json, Path
    - DATA_DIR, STAGE3_5_OUT_DIR
    - load_dataframe(filename, nrows=None) -> pd.DataFrame
    - time

    Everything else (models, metrics, imports, logic) is fully under the code's control.
    """
    from .config import DATA_DIR, STAGE3_5_OUT_DIR
    import time

    def load_dataframe_helper(filename: str, nrows: Optional[int] = None):
        """Load a dataframe, preferring DATA_DIR then Stage 3B output for prepared parquet."""
        try:
            return load_dataframe(filename, nrows=nrows, base_dir=DATA_DIR)
        except FileNotFoundError:
            # Try exact match in STAGE3B
            prepared_path = STAGE3B_OUT_DIR / filename
            if prepared_path.exists():
                return load_dataframe(prepared_path)
            
            # Robust lookup in STAGE3B_OUT_DIR
            name = Path(filename).name
            
            # 1. If name contains TSK-XXX but not PLAN-, try adding PLAN- (via wildcard)
            if "TSK-" in name and "PLAN-" not in name:
                pattern = name.replace("TSK-", "*TSK-")
                matches = list(STAGE3B_OUT_DIR.glob(pattern))
                if matches:
                     return load_dataframe(matches[0])
            
            # 2. Try general glob
            matches = list(STAGE3B_OUT_DIR.glob(f"*{name}*"))
            if matches:
                return load_dataframe(matches[0])
                
            raise

    globals_dict = {
        "__name__": "__stage3_5_tester__",
        "__builtins__": __builtins__,  # allow normal imports
        "pd": pd,
        "np": np,
        "json": json,
        "Path": Path,
        "DATA_DIR": DATA_DIR,
        "STAGE3_5_OUT_DIR": STAGE3_5_OUT_DIR,
        "load_dataframe": load_dataframe_helper,
        "time": time,
    }

    buf = io.StringIO()

    try:
        with contextlib.redirect_stdout(buf):
            print(f"=== {description} ===")
            # Use globals_dict for both globals and locals so code can define functions/vars and reuse them
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                warnings.filterwarnings("ignore", category=UserWarning)
                exec(code, globals_dict, globals_dict)
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return f"[ERROR] {e}\n\nTraceback:\n{error_details}"

    output = buf.getvalue()
    return output if output else "[Code executed successfully, no output]"


@tool
def python_sandbox_stage3_5(code: str) -> str:
    """Quick Python sandbox for Stage 3.5 data exploration and prep.

    Use this to:
    - Inspect columns, dtypes, and sample rows
    - Prototype lightweight cleaning/prep helpers shared across methods
    - Verify parsing of date/time columns before full benchmarks

    Available:
    - pd, np, json, Path
    - DATA_DIR, STAGE3_5_OUT_DIR
    - load_dataframe(filename, nrows=None)

    Nothing is pre-hardcoded—your code decides what to do.
    """
    from .config import DATA_DIR, STAGE3_5_OUT_DIR

    def load_dataframe_helper(filename: str, nrows: Optional[int] = None):
        try:
            return load_dataframe(filename, nrows=nrows, base_dir=DATA_DIR)
        except FileNotFoundError:
            # Try exact match in STAGE3B
            prepared_path = STAGE3B_OUT_DIR / filename
            if prepared_path.exists():
                return load_dataframe(prepared_path)
            
            # Robust lookup in STAGE3B_OUT_DIR
            name = Path(filename).name
            
            # 1. If name contains TSK-XXX but not PLAN-, try adding PLAN- (via wildcard)
            if "TSK-" in name and "PLAN-" not in name:
                pattern = name.replace("TSK-", "*TSK-")
                matches = list(STAGE3B_OUT_DIR.glob(pattern))
                if matches:
                     return load_dataframe(matches[0])
            
            # 2. Try general glob
            matches = list(STAGE3B_OUT_DIR.glob(f"*{name}*"))
            if matches:
                return load_dataframe(matches[0])
                
            raise

    globals_dict = {
        "__name__": "__stage3_5_sandbox__",
        "pd": pd,
        "np": np,
        "json": json,
        "Path": Path,
        "DATA_DIR": DATA_DIR,
        "STAGE3_5_OUT_DIR": STAGE3_5_OUT_DIR,
        "load_dataframe": load_dataframe_helper,
    }

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, globals_dict, globals_dict)
    except Exception as e:
        return f"[ERROR] {e}"
    return buf.getvalue() or "[No output]"


@tool
def save_checkpoint_stage3_5(checkpoint_json: Dict[str, Any]) -> str:
    """Save checkpoint with VERBOSE confirmation and auto-verification.

    This checkpoint should contain:
    - plan_id: The plan being tested
    - data_split_strategy: Summary of how data is split
    - date_column: Name of date column (if identified)
    - target_column: Name of target column (if identified)
    - train_period: Training period (e.g., "2020-2023")
    - validation_period: Validation period (e.g., "2024")
    - test_period: Test period if applicable
    - methods_to_test: List of 3 ForecastingMethod objects (as dicts)
    - methods_completed: List of method_ids fully benchmarked (3 iterations)
    - benchmark_results: List of all BenchmarkResult objects so far (as dicts)
    - iteration_counts: Dict mapping method_id to number of successful iterations

    Args:
        checkpoint_json: Dictionary containing checkpoint data

    Returns:
        Detailed confirmation message with verification
    """
    from .config import STAGE3_5_OUT_DIR
    from .models import Stage3_5Checkpoint
    from datetime import datetime

    # Validate against schema
    try:
        checkpoint = Stage3_5Checkpoint.model_validate(checkpoint_json)
    except Exception as e:
        return f"❌ [ERROR] Checkpoint validation failed: {e}\n\nPlease fix the JSON and try again."

    # Update timestamp
    checkpoint_dict = checkpoint.model_dump()
    checkpoint_dict["updated_at"] = datetime.now().isoformat()

    plan_id = checkpoint.plan_id
    # Enforce PLAN- prefix if missing, as per user requirement
    if not plan_id.startswith("PLAN-") and "PLAN-" not in plan_id:
        # Check if it's just TSK-XXX
        if plan_id.startswith("TSK-"):
            plan_id = f"PLAN-{plan_id}"
            
    STAGE3_5_OUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = STAGE3_5_OUT_DIR / f"checkpoint_{plan_id}.json"

    # Save checkpoint
    try:
        checkpoint_path.write_text(json.dumps(checkpoint_dict, indent=2))
    except Exception as e:
        return f"❌ [ERROR] Failed to write checkpoint file: {e}"

    # VERIFY: Re-load the file to confirm it was actually written
    try:
        saved_data = json.loads(checkpoint_path.read_text())
        verified_results = len(saved_data.get("benchmark_results", []))
        verified_iterations = saved_data.get("iteration_counts", {})
    except Exception as e:
        return f"❌ [ERROR] Checkpoint was saved but verification failed: {e}\n\nFile may be corrupt!"

    # Detailed summary
    methods_left = [m["method_id"] for m in checkpoint.methods_to_test
                   if m["method_id"] not in checkpoint.methods_completed]

    summary = f"""✓ CHECKPOINT SAVED AND VERIFIED: {checkpoint_path.name}

Progress Summary:
- Methods completed: {len(checkpoint.methods_completed)}/3
  ✓ Completed: {checkpoint.methods_completed if checkpoint.methods_completed else 'None'}
  ⏳ Remaining: {methods_left if methods_left else 'None - all done!'}

- Total benchmark results stored: {verified_results}
- Iteration counts per method: {verified_iterations}
- Data split: {checkpoint.data_split_strategy}

File location: {checkpoint_path.absolute()}

✓ Verification: Re-loaded file confirms {verified_results} results saved successfully.

You can verify this checkpoint anytime by calling verify_checkpoint_stage3_5().
"""

    return summary


@tool
def load_checkpoint_stage3_5(plan_id: str) -> str:
    """Load existing checkpoint for Stage 3.5 to resume progress.

    This retrieves:
    - What methods need to be tested
    - What methods are already completed
    - The data split strategy to maintain consistency
    - All benchmark results collected so far
    - Progress on iterations per method

    Args:
        plan_id: Plan ID (e.g., 'PLAN-TSK-001')

    Returns:
        JSON string of the checkpoint, or error if not found
    """
    from .config import STAGE3_5_OUT_DIR

    checkpoint_path = STAGE3_5_OUT_DIR / f"checkpoint_{plan_id}.json"
    
    if not checkpoint_path.exists():
        # Try finding by pattern using FILE_NAMING_PATTERNS
        # This handles the case where plan_id is 'TSK-001' but file is 'checkpoint_PLAN-TSK-001.json'
        pattern = f"checkpoint_*{plan_id}*.json"
        matches = list(STAGE3_5_OUT_DIR.glob(pattern))
        if matches:
            checkpoint_path = matches[0]

    if not checkpoint_path.exists():
        return f"[INFO] No checkpoint found for {plan_id}. Starting fresh."

    try:
        checkpoint_data = json.loads(checkpoint_path.read_text())

        # Format a helpful summary
        methods_to_test = checkpoint_data.get("methods_to_test", [])
        methods_completed = checkpoint_data.get("methods_completed", [])
        iteration_counts = checkpoint_data.get("iteration_counts", {})

        methods_left = [m["method_id"] for m in methods_to_test
                       if m["method_id"] not in methods_completed]

        summary = (
            f"=== CHECKPOINT LOADED for {plan_id} ===\n\n"
            f"Data Split Strategy: {checkpoint_data.get('data_split_strategy', 'Not set')}\n"
            f"Date Column: {checkpoint_data.get('date_column', 'Not identified')}\n"
            f"Target Column: {checkpoint_data.get('target_column', 'Not identified')}\n"
            f"Train Period: {checkpoint_data.get('train_period', 'Not set')}\n"
            f"Validation Period: {checkpoint_data.get('validation_period', 'Not set')}\n\n"
            f"Methods to Test ({len(methods_to_test)} total):\n"
        )

        for method in methods_to_test:
            method_id = method["method_id"]
            status = "✓ COMPLETED" if method_id in methods_completed else f"⏳ In Progress ({iteration_counts.get(method_id, 0)}/3 iterations)"
            summary += f"  - {method_id} ({method['name']}): {status}\n"

        summary += f"\nMethods Remaining: {methods_left if methods_left else 'None - ready to select best method!'}\n"
        summary += f"Total Benchmark Results: {len(checkpoint_data.get('benchmark_results', []))}\n\n"
        summary += f"Full checkpoint data:\n{json.dumps(checkpoint_data, indent=2)}"

        return summary
    except Exception as e:
        return f"[ERROR] Failed to load checkpoint: {e}"


@tool
def verify_checkpoint_stage3_5(plan_id: str, expected_iterations: Dict[str, int]) -> str:
    """Verify checkpoint was saved correctly by comparing expected vs actual iteration counts.

    Use this tool after save_checkpoint_stage3_5() to confirm the checkpoint was properly updated.

    Args:
        plan_id: Plan ID to verify (e.g., 'PLAN-TSK-001')
        expected_iterations: Dict mapping method_id to expected iteration count
            Example: {"METHOD-1": 3, "METHOD-2": 1, "METHOD-3": 0}

    Returns:
        Verification result with details - either ✓ PASSED or ❌ FAILED
    """
    from .config import STAGE3_5_OUT_DIR

    checkpoint_path = STAGE3_5_OUT_DIR / f"checkpoint_{plan_id}.json"
    
    if not checkpoint_path.exists():
        # Try finding by pattern
        pattern = f"checkpoint_*{plan_id}*.json"
        matches = list(STAGE3_5_OUT_DIR.glob(pattern))
        if matches:
            checkpoint_path = matches[0]

    if not checkpoint_path.exists():
        return f"❌ VERIFICATION FAILED: No checkpoint file found at {checkpoint_path}"

    try:
        checkpoint_data = json.loads(checkpoint_path.read_text())
        actual_iterations = checkpoint_data.get("iteration_counts", {})
        actual_results = len(checkpoint_data.get("benchmark_results", []))

        # Compare expected vs actual
        mismatches = []
        for method_id, expected_count in expected_iterations.items():
            actual_count = actual_iterations.get(method_id, 0)
            if actual_count != expected_count:
                mismatches.append(
                    f"  {method_id}: Expected {expected_count}, Got {actual_count}"
                )

        if mismatches:
            return f"""❌ VERIFICATION FAILED: Iteration counts don't match!

Mismatches:
{chr(10).join(mismatches)}

Actual checkpoint state:
- Iterations: {actual_iterations}
- Results count: {actual_results}

ACTION REQUIRED: Re-save checkpoint with correct data!
"""

        return f"""✓ VERIFICATION PASSED!

All iteration counts match expected values:
{chr(10).join(f'  {mid}: {cnt} iterations ✓' for mid, cnt in expected_iterations.items())}

Total results in checkpoint: {actual_results}
Checkpoint is valid and up-to-date.
"""

    except Exception as e:
        return f"❌ VERIFICATION ERROR: {e}"


@tool
def save_tester_output(output_json: Dict[str, Any]) -> str:
    """Save the final tester output with method selection results.
    
    Args:
        output_json: JSON payload containing:
            - plan_id: ID of the plan being tested
            - task_category: predictive/descriptive/unsupervised
            - methods_proposed: list of ForecastingMethod objects
            - benchmark_results: list of BenchmarkResult objects
            - selected_method_id: ID of the winning method
            - selected_method: The complete ForecastingMethod object for the winner
            - selection_rationale: Why this method was selected
            - data_split_strategy: How data was split for testing
            
    Returns:
        Confirmation message with save path
    """
    from .config import STAGE3_5B_OUT_DIR  # Changed from STAGE3_5_OUT_DIR to consolidate outputs
    from .models import TesterOutput
    from datetime import datetime

    # Allow lenient inputs: accept dict or JSON string
    if isinstance(output_json, str):
        try:
            output_data = json.loads(output_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}") from e
    elif isinstance(output_json, dict):
        output_data = output_json
    else:
        raise ValueError("output_json must be a dict or JSON string")

    # Validate against schema
    try:
        tester_output = TesterOutput.model_validate(output_data)
    except Exception as e:
        raise ValueError(f"Schema validation failed: {e}")

    plan_id = tester_output.plan_id
    # Enforce PLAN- prefix if missing
    if not plan_id.startswith("PLAN-") and "PLAN-" not in plan_id:
        if plan_id.startswith("TSK-"):
            plan_id = f"PLAN-{plan_id}"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save to stage3_5b_benchmarking (consolidated with checkpoints, not legacy stage3_5_tester)
    STAGE3_5B_OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = STAGE3_5B_OUT_DIR / f"tester_{plan_id}_{timestamp}.json"
    output_path.write_text(json.dumps(output_data, indent=2))

    return f"saved::{output_path.name}"


@tool
def record_thought(thought: str, what_im_about_to_do: str) -> str:
    """Record your reasoning BEFORE taking an action.
    
    Use this to explicitly document your thinking before calling other tools.
    This helps you stay strategic and avoid repeating mistakes.
    
    Args:
        thought: Your current reasoning - what you know, what's uncertain, what you're considering
        what_im_about_to_do: What action you plan to take next and WHY
        
    Returns:
        Confirmation message
        
    Example:
        record_thought(
            thought="I've seen that the export data has yearly columns but the production data only has 2020-2025. "
                    "A cross-file join won't work because there's no common key.",
            what_im_about_to_do="I'll use python_sandbox to test loading just the export data and reshaping it to long format"
        )
    """
    return f"💭 Thought recorded. Proceeding with: {what_im_about_to_do[:80]}..."


@tool
def record_observation(what_happened: str, what_i_learned: str, next_step: str) -> str:
    """Record what you observed and learned AFTER an action.
    
    Use this to reflect on tool results before deciding what to do next.
    This helps you learn from errors and adjust your strategy.
    
    Args:
        what_happened: What the last tool/action resulted in (success, error, unexpected result)
        what_i_learned: Key insight or lesson from this result
        next_step: What you'll do next based on what you learned
        
    Returns:
        Confirmation message
        
    Example:
        record_observation(
            what_happened="run_benchmark_code failed with 'Found array with 0 samples'",
            what_i_learned="The validation set is empty after dropna() - this means my slicing strategy is wrong",
            next_step="I'll inspect the data shape before/after split to understand the actual structure"
        )
    """
    return f"👁️ Observation recorded. Learning: {what_i_learned[:80]}... → Next: {next_step[:60]}..."


# Stage 3B tool list
STAGE3B_TOOLS = [
    record_thought,  # ReAct: explicit reasoning
    record_observation,  # ReAct: reflection
    load_stage3_plan_for_prep,
    list_data_files,
    inspect_data_file,
    python_sandbox_stage3b,
    run_data_prep_code,
    save_prepared_data,
    search,
]


# ===========================
# Stage 3.5a Tools (Method Proposal)
# ===========================

@tool
def save_method_proposal_output(output_json: Dict[str, Any]) -> str:
    """Save the method proposal output from Stage 3.5a.

    Args:
        output_json: JSON payload containing:
            - plan_id: ID of the plan
            - task_category: predictive/descriptive/unsupervised
            - methods_proposed: list of 3 ForecastingMethod objects
            - data_split_strategy: How data will be split
            - date_column, target_column: Identified columns
            - train_period, validation_period, test_period: Period specifications
            - data_preprocessing_steps: Ordered list of preprocessing steps

    Returns:
        Confirmation message with save path
    """
    from .config import STAGE3_5A_OUT_DIR
    from .models import MethodProposalOutput
    from datetime import datetime

    # Allow lenient inputs
    if isinstance(output_json, str):
        try:
            output_data = json.loads(output_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}") from e
    elif isinstance(output_json, dict):
        output_data = output_json
    else:
        raise ValueError("output_json must be a dict or JSON string")

    # Validate against schema
    try:
        proposal_output = MethodProposalOutput.model_validate(output_data)
    except Exception as e:
        raise ValueError(f"Schema validation failed: {e}")

    plan_id = proposal_output.plan_id
    # Enforce PLAN- prefix if missing
    if not plan_id.startswith("PLAN-") and "PLAN-" not in plan_id:
        if plan_id.startswith("TSK-"):
            plan_id = f"PLAN-{plan_id}"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    STAGE3_5A_OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = STAGE3_5A_OUT_DIR / f"method_proposal_{plan_id}_{timestamp}.json"
    output_path.write_text(json.dumps(output_data, indent=2))

    return f"✅ Method proposal saved: {output_path.name}\n\nNext step: Run Stage 3.5b to benchmark these methods."


# ===========================
# Stage 3.5b Tools (Method Benchmarking)
# ===========================

@tool
def load_method_proposals(plan_id: str) -> str:
    """Load method proposals from Stage 3.5a.

    Args:
        plan_id: Plan ID (e.g., 'PLAN-TSK-001')

    Returns:
        JSON string containing MethodProposalOutput
    """
    from .config import STAGE3_5A_OUT_DIR

    # Find the latest method proposal file for this plan
    proposal_files = sorted(STAGE3_5A_OUT_DIR.glob(f"method_proposal_*{plan_id}*.json"))

    if not proposal_files:
        return f"ERROR: No method proposals found for plan_id '{plan_id}' in {STAGE3_5A_OUT_DIR}"

    latest_file = proposal_files[-1]
    proposal_data = json.loads(latest_file.read_text())

    return json.dumps(proposal_data, indent=2)


@tool
def save_checkpoint_stage3_5b(checkpoint_json: Dict[str, Any]) -> str:
    """Save checkpoint for Stage 3.5b.
    
    SIMPLIFIED STRATEGY: Save checkpoint only when a method completes all 3 iterations
    with consistent results (CV < 0.3).

    This checkpoint should contain:
    - plan_id: The plan being tested
    - data_split_strategy: Summary of how data is split
    - date_column, target_column: Column names
    - train_period, validation_period, test_period: Period specifications
    - methods_to_test: List of ForecastingMethod dicts
    - methods_completed: List of method_ids that completed 3 iterations successfully
    - completed_results: List of BenchmarkResult dicts (averaged, one per completed method)

    Args:
        checkpoint_json: Complete checkpoint data as dict

    Returns:
        Confirmation message with save path
    """
    from .config import STAGE3_5B_OUT_DIR
    from .models import Stage3_5Checkpoint
    from datetime import datetime

    # Allow lenient inputs
    if isinstance(checkpoint_json, str):
        try:
            checkpoint_data = json.loads(checkpoint_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}") from e
    elif isinstance(checkpoint_json, dict):
        checkpoint_data = checkpoint_json
    else:
        raise ValueError("checkpoint_json must be a dict or JSON string")

    # Add/update timestamp
    checkpoint_data["updated_at"] = datetime.now().isoformat()
    if "created_at" not in checkpoint_data:
        checkpoint_data["created_at"] = checkpoint_data["updated_at"]

    # Validate against schema
    try:
        checkpoint = Stage3_5Checkpoint.model_validate(checkpoint_data)
    except Exception as e:
        raise ValueError(f"Checkpoint validation failed: {e}")

    plan_id = checkpoint.plan_id

    STAGE3_5B_OUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = STAGE3_5B_OUT_DIR / f"checkpoint_{plan_id}.json"
    checkpoint_path.write_text(json.dumps(checkpoint_data, indent=2))

    # Build progress summary
    methods_completed = checkpoint.methods_completed
    total_methods = len(checkpoint.methods_to_test)
    completed_count = len(methods_completed)

    summary = (
        f"✅ Checkpoint saved: {checkpoint_path.name}\n"
        f"📊 Progress: {completed_count}/{total_methods} methods completed\n"
    )

    if methods_completed:
        summary += "   Completed methods:\n"
        for method_id in methods_completed:
            summary += f"   - {method_id} ✓\n"

    remaining = [m.get("method_id", "?") for m in checkpoint.methods_to_test 
                 if m.get("method_id") not in methods_completed]
    if remaining:
        summary += f"\n   Remaining methods: {', '.join(remaining)}\n"

    return summary


@tool
def load_checkpoint_stage3_5b(plan_id: str) -> str:
    """Load existing checkpoint for Stage 3.5b to resume progress.

    Args:
        plan_id: Plan ID (e.g., 'PLAN-TSK-001')

    Returns:
        JSON string containing checkpoint data, or error message if no checkpoint
    """
    from .config import STAGE3_5B_OUT_DIR

    checkpoint_path = STAGE3_5B_OUT_DIR / f"checkpoint_{plan_id}.json"

    if not checkpoint_path.exists():
        return f"No checkpoint found for plan '{plan_id}'. Starting fresh."

    checkpoint_data = json.loads(checkpoint_path.read_text())

    # Build summary
    methods_completed = checkpoint_data.get("methods_completed", [])
    methods_to_test = checkpoint_data.get("methods_to_test", [])
    total_methods = len(methods_to_test)
    completed_count = len(methods_completed)

    summary = (
        f"✅ Checkpoint loaded: {checkpoint_path.name}\n"
        f"📊 Progress: {completed_count}/{total_methods} methods completed\n"
    )

    if methods_completed:
        summary += "   Completed methods:\n"
        for method_id in methods_completed:
            summary += f"   - {method_id} ✓\n"

    remaining = [m.get("method_id", "?") for m in methods_to_test 
                 if m.get("method_id") not in methods_completed]
    if remaining:
        summary += f"\n   ⚠️  Need to benchmark: {', '.join(remaining)}\n"
        summary += "   Run 3 iterations for each remaining method, check consistency, then save checkpoint.\n"

    summary += f"\n📋 Full checkpoint data:\n{json.dumps(checkpoint_data, indent=2)}"

    return summary


@tool
def python_sandbox_stage3_5b(code: str) -> str:
    """Quick Python sandbox for Stage 3.5b data exploration.

    Use this to:
    - Inspect data structure before benchmarking
    - Test code snippets
    - Debug data loading issues

    Args:
        code: Python code to execute

    Returns:
        Execution result (stdout + stderr)
    """
    from .config import DATA_DIR, STAGE3_5B_OUT_DIR
    
    def load_dataframe_helper(filename: str, nrows: Optional[int] = None):
        try:
            return load_dataframe(filename, nrows=nrows, base_dir=DATA_DIR)
        except FileNotFoundError:
            # Try exact match in STAGE3B
            prepared_path = STAGE3B_OUT_DIR / filename
            if prepared_path.exists():
                return load_dataframe(prepared_path)
            
            # Robust lookup in STAGE3B_OUT_DIR
            name = Path(filename).name
            
            # 1. If name contains TSK-XXX but not PLAN-, try adding PLAN- (via wildcard)
            if "TSK-" in name and "PLAN-" not in name:
                pattern = name.replace("TSK-", "*TSK-")
                matches = list(STAGE3B_OUT_DIR.glob(pattern))
                if matches:
                     return load_dataframe(matches[0])
            
            # 2. Try general glob
            matches = list(STAGE3B_OUT_DIR.glob(f"*{name}*"))
            if matches:
                return load_dataframe(matches[0])
                
            raise

    globals_dict = {
        "__name__": "__stage3_5b_sandbox__",
        "pd": pd,
        "np": np,
        "json": json,
        "Path": Path,
        "DATA_DIR": DATA_DIR,
        "STAGE3_5B_OUT_DIR": STAGE3_5B_OUT_DIR,
        "load_dataframe": load_dataframe_helper,
    }

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, globals_dict, globals_dict)
    except Exception as e:
        return f"[ERROR] {e}"
    return buf.getvalue() or "[No output]"


# Stage 3.5a tool list
STAGE3_5A_TOOLS = [
    record_thought,  # ReAct: explicit reasoning before action
    record_observation,  # ReAct: reflection after action
    load_stage3_plan_for_tester,
    list_data_files,
    inspect_data_file,
    python_sandbox_stage3_5,
    search,
    save_method_proposal_output,
]

# Stage 3.5b tool list (SIMPLIFIED CHECKPOINT STRATEGY)
STAGE3_5B_TOOLS = [
    # ReAct tools
    record_thought,  # Record thoughts before actions
    record_observation,  # Record observations after actions
    
    # Checkpoint tools (SIMPLIFIED - no verification)
    load_checkpoint_stage3_5b,  # Load existing checkpoint
    save_checkpoint_stage3_5b,  # Save checkpoint when method completes
    
    # Method Proposals
    load_method_proposals,  # Load proposals from Stage 3.5a
    
    # Data exploration
    list_data_files,  # List available data files
    inspect_data_file,  # Inspect data file structure
    python_sandbox_stage3_5b,  # Quick Python execution
    
    # Benchmarking
    run_benchmark_code,  # Execute benchmarking code
    search,  # Search for examples
    
    # Final output
    save_tester_output,  # Save final recommendation
]

# Stage 3.5 tool list (Legacy - kept for compatibility)
STAGE3_5_TOOLS = [
    record_thought,  # ReAct: explicit reasoning before action
    record_observation,  # ReAct: reflection after action
    load_checkpoint_stage3_5,  # Load checkpoint to resume progress
    save_checkpoint_stage3_5,  # Save checkpoint to maintain memory
    verify_checkpoint_stage3_5,  # Verify checkpoint was saved correctly
    load_stage3_plan_for_tester,
    list_summary_files,  # Access Stage 1 summaries
    read_summary_file,  # Read dataset summaries
    search,
    list_data_files,
    inspect_data_file,
    python_sandbox_stage3_5,
    run_benchmark_code,
    save_tester_output,
]



@tool
def list_stage3_plans() -> List[str]:
    """List all available Stage 3 plans."""
    plans = sorted([p.name for p in STAGE3_OUT_DIR.glob("*.json")])
    return plans


@tool
def load_stage3_plan(plan_id: str) -> str:
    """Load a Stage 3 plan by ID.
    
    Args:
        plan_id: Plan identifier (e.g., 'PLAN-TSK-002')
        
    Returns:
        JSON string of the plan
    """
    # Try exact match first
    plan_path = STAGE3_OUT_DIR / f"{plan_id}.json"
    if not plan_path.exists():
        # Try finding by pattern
        matches = list(STAGE3_OUT_DIR.glob(f"*{plan_id}*.json"))
        if not matches:
            raise FileNotFoundError(f"No plan found matching: {plan_id}")
        plan_path = matches[0]
    
    return plan_path.read_text()


@tool
def execute_python_code(code: str, description: str = "Executing code") -> str:
    """Execute arbitrary Python code for data processing, modeling, and analysis.
    
    This is the primary tool for implementing the Stage 3 plan.
    You can:
    - Load and transform data
    - Perform joins and aggregations
    - Build and train models
    - Generate predictions
    - Calculate metrics
    - Save intermediate and final results
    
    Available in the execution environment:
    - pandas as pd
    - numpy as np
    - sklearn (all modules)
    - json, pathlib.Path
    - DATA_DIR, STAGE4_OUT_DIR, STAGE4_WORKSPACE
    - Helper: load_dataframe(filename) for loading files from DATA_DIR
    
    Args:
        code: Python code to execute
        description: Brief description of what this code does
        
    Returns:
        Output printed to stdout, or error message
    """
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    def load_dataframe_helper(filename: str):
        """Load a dataframe from DATA_DIR."""
        return load_dataframe(filename, base_dir=DATA_DIR)
    
    globals_dict = {
        "__name__": "__stage4_executor__",
        "pd": pd,
        "np": np,
        "json": json,
        "Path": Path,
        "DATA_DIR": DATA_DIR,
        "STAGE4_OUT_DIR": STAGE4_OUT_DIR,
        "STAGE4_WORKSPACE": STAGE4_WORKSPACE,
        "load_dataframe": load_dataframe_helper,
        "LinearRegression": LinearRegression,
        "Ridge": Ridge,
        "Lasso": Lasso,
        "RandomForestRegressor": RandomForestRegressor,
        "GradientBoostingRegressor": GradientBoostingRegressor,
        "DecisionTreeRegressor": DecisionTreeRegressor,
        "mean_absolute_error": mean_absolute_error,
        "mean_squared_error": mean_squared_error,
        "r2_score": r2_score,
        "train_test_split": train_test_split,
        "StandardScaler": StandardScaler,
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


@tool
def save_execution_result(result_json: str) -> str:
    """Save the final execution result.
    
    Args:
        result_json: JSON string containing:
            - plan_id: ID of the executed plan
            - task_category: descriptive/predictive/unsupervised
            - status: success/failure/partial
            - outputs: dict with output file paths
            - metrics: dict with performance metrics (if applicable)
            - summary: text summary of results
            - errors: list of any errors encountered
    
    Returns:
        Confirmation message with save path
    """
    from datetime import datetime
    
    try:
        result = json.loads(result_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")
    
    plan_id = result.get("plan_id", "UNKNOWN")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_path = STAGE4_OUT_DIR / f"execution_{plan_id}_{timestamp}.json"
    output_path.write_text(json.dumps(result, indent=2))
    
    return f"✅ Execution result saved to: {output_path}"


# Stage 4 tool list
STAGE4_TOOLS = [
    list_stage3_plans,
    load_stage3_plan,
    list_data_files,
    execute_python_code,
    save_execution_result,
]


# ===========================
# Stage 5: Visualization Tools
# ===========================

@tool
def list_stage4_results() -> List[str]:
    """List all available Stage 4 execution results."""
    results = sorted([p.name for p in STAGE4_OUT_DIR.glob("execution_*.json")])
    return results


@tool
def load_stage4_result(result_id: str) -> str:
    """Load a Stage 4 execution result.
    
    Args:
        result_id: Result filename or pattern
        
    Returns:
        JSON string of the result
    """
    # Try exact match first
    result_path = STAGE4_OUT_DIR / result_id
    if not result_path.exists():
        # Try finding by pattern
        matches = list(STAGE4_OUT_DIR.glob(f"*{result_id}*.json"))
        if not matches:
            raise FileNotFoundError(f"No result found matching: {result_id}")
        result_path = matches[0]
    
    return result_path.read_text()


@tool
def load_stage3_plan_viz(plan_id: str) -> str:
    """Load a Stage 3 plan for context.
    
    Args:
        plan_id: Plan identifier
        
    Returns:
        JSON string of the plan
    """
    plan_path = STAGE3_OUT_DIR / f"{plan_id}.json"
    if not plan_path.exists():
        matches = list(STAGE3_OUT_DIR.glob(f"*{plan_id}*.json"))
        if not matches:
            raise FileNotFoundError(f"No plan found matching: {plan_id}")
        plan_path = matches[0]
    
    return plan_path.read_text()


@tool
def create_visualizations(code: str, description: str = "Creating visualizations") -> str:
    """Execute Python code to create visualizations and reports.
    
    Use this to:
    - Load data from Stage 4 outputs
    - Create plots (matplotlib, seaborn, plotly)
    - Generate summary tables
    - Create HTML reports
    - Save visualizations to STAGE5_OUT_DIR
    
    Available in the environment:
    - pandas as pd
    - numpy as np
    - matplotlib.pyplot as plt
    - seaborn as sns
    - plotly.express as px (if available)
    - plotly.graph_objects as go (if available)
    - json, pathlib.Path
    - STAGE4_OUT_DIR, STAGE5_OUT_DIR, STAGE5_WORKSPACE
    
    Args:
        code: Python code to execute
        description: Brief description of what this code creates
        
    Returns:
        Output printed to stdout, or error message
    """
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Try to import plotly (optional)
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        has_plotly = True
    except ImportError:
        px = None
        go = None
        has_plotly = False
    
    def load_dataframe_viz(filepath):
        """Load a dataframe from any supported format."""
        filepath = Path(filepath)
        
        # 1. Try exact path
        if filepath.exists():
            return load_dataframe(filepath, base_dir=STAGE4_OUT_DIR)
            
        # 2. Try relative to STAGE4_OUT_DIR (exact)
        candidate = STAGE4_OUT_DIR / filepath.name
        if candidate.exists():
            return load_dataframe(candidate, base_dir=STAGE4_OUT_DIR)
            
        # 3. Robust lookup in STAGE4_OUT_DIR
        name = filepath.name
        
        # 3a. If name contains TSK-XXX but not PLAN-, try adding PLAN- (via wildcard)
        if "TSK-" in name and "PLAN-" not in name:
            pattern = name.replace("TSK-", "*TSK-")
            matches = list(STAGE4_OUT_DIR.glob(pattern))
            if matches:
                 return load_dataframe(matches[0], base_dir=STAGE4_OUT_DIR)
        
        # 3b. Try general glob
        matches = list(STAGE4_OUT_DIR.glob(f"*{name}*"))
        if matches:
            return load_dataframe(matches[0], base_dir=STAGE4_OUT_DIR)
        
        raise FileNotFoundError(f"File not found: {filepath} (tried exact and glob in {STAGE4_OUT_DIR})")
    
    globals_dict = {
        "__name__": "__stage5_visualizer__",
        "pd": pd,
        "np": np,
        "plt": plt,
        "sns": sns,
        "json": json,
        "Path": Path,
        "STAGE4_OUT_DIR": STAGE4_OUT_DIR,
        "STAGE5_OUT_DIR": STAGE5_OUT_DIR,
        "STAGE5_WORKSPACE": STAGE5_WORKSPACE,
        "load_dataframe": load_dataframe_viz,
    }
    
    if has_plotly:
        globals_dict.update({"px": px, "go": go})
    
    # Use globals_dict for both globals and locals to avoid scope issues
    buf = io.StringIO()
    
    try:
        with contextlib.redirect_stdout(buf):
            print(f"=== {description} ===")
            # Use same dict for both to keep variables in scope
            exec(code, globals_dict, globals_dict)
            # Close any open plots
            plt.close('all')
    except Exception as e:
        plt.close('all')
        import traceback
        error_details = traceback.format_exc()
        return f"[ERROR] {e}\n\n{error_details}"
    
    output = buf.getvalue()
    return output if output else "[Visualizations created successfully]"


@tool
def analyze_data_columns(parquet_path: str) -> str:
    """Analyze the columns in a Stage 4 output parquet file to understand what data is available.

    This tool helps the visualization agent understand:
    - What columns represent given/original data
    - What columns represent predictions/model outputs
    - What columns could be useful for visualization
    - Data types, ranges, and basic statistics

    Args:
        parquet_path: Path to the parquet file (can be relative to STAGE4_OUT_DIR)

    Returns:
        Detailed analysis of columns and data structure
    """
    filepath = Path(parquet_path)
    if not filepath.exists():
        # Try relative to STAGE4_OUT_DIR
        filepath = STAGE4_OUT_DIR / filepath.name

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {parquet_path}")

    df = load_dataframe(filepath, base_dir=STAGE4_OUT_DIR)

    analysis = []
    analysis.append(f"=" * 80)
    analysis.append(f"DATA ANALYSIS: {filepath.name}")
    analysis.append(f"=" * 80)
    analysis.append(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns\n")

    # Categorize columns
    given_cols = []
    predicted_cols = []
    engineered_cols = []
    temporal_cols = []
    categorical_cols = []

    for col in df.columns:
        col_lower = col.lower()

        # Identify predicted/model outputs
        if any(x in col_lower for x in ['predicted', 'forecast', 'residual', 'error', 'confidence']):
            predicted_cols.append(col)
        # Identify engineered features
        elif any(x in col_lower for x in ['lagged', 'growth', 'rolling', 'diff', 'rate', 'ratio']):
            engineered_cols.append(col)
        # Identify temporal columns
        elif any(x in col_lower for x in ['date', 'time', 'year', 'month', 'day', 'period']):
            temporal_cols.append(col)
        # Identify categorical
        elif df[col].dtype == 'object' or df[col].nunique() < 20:
            categorical_cols.append(col)
        # Otherwise, it's likely given/original data
        else:
            given_cols.append(col)

    # Report categorization
    analysis.append("COLUMN CATEGORIZATION:")
    analysis.append(f"\n📥 GIVEN/ORIGINAL DATA ({len(given_cols)} columns):")
    for col in given_cols[:10]:  # Show first 10
        analysis.append(f"  - {col}: {df[col].dtype}, range [{df[col].min():.2f}, {df[col].max():.2f}]" if df[col].dtype != 'object' else f"  - {col}: {df[col].dtype}, {df[col].nunique()} unique values")
    if len(given_cols) > 10:
        analysis.append(f"  ... and {len(given_cols) - 10} more")

    analysis.append(f"\n🔮 PREDICTED/MODEL OUTPUTS ({len(predicted_cols)} columns):")
    for col in predicted_cols:
        analysis.append(f"  - {col}: {df[col].dtype}, range [{df[col].min():.2f}, {df[col].max():.2f}]" if df[col].dtype != 'object' else f"  - {col}: {df[col].dtype}")

    analysis.append(f"\n🔧 ENGINEERED FEATURES ({len(engineered_cols)} columns):")
    for col in engineered_cols:
        analysis.append(f"  - {col}: {df[col].dtype}, range [{df[col].min():.2f}, {df[col].max():.2f}]" if df[col].dtype != 'object' else f"  - {col}: {df[col].dtype}")

    analysis.append(f"\n📅 TEMPORAL COLUMNS ({len(temporal_cols)} columns):")
    for col in temporal_cols:
        analysis.append(f"  - {col}: {df[col].dtype}")

    analysis.append(f"\n🏷️  CATEGORICAL COLUMNS ({len(categorical_cols)} columns):")
    for col in categorical_cols:
        analysis.append(f"  - {col}: {df[col].nunique()} unique values → {list(df[col].unique()[:5])}")

    # Add summary statistics for key columns
    analysis.append(f"\n" + "=" * 80)
    analysis.append("KEY INSIGHTS:")
    analysis.append("=" * 80)

    # Find actual vs predicted columns
    actual_cols = [c for c in df.columns if any(x in c.lower() for x in ['production', 'yield', 'area', 'sales', 'value']) and c not in predicted_cols and c not in engineered_cols]

    if predicted_cols and actual_cols:
        analysis.append(f"\n✓ This dataset contains PREDICTIONS that can be compared to ACTUAL values")
        analysis.append(f"  - Actual value columns: {actual_cols[:3]}")
        analysis.append(f"  - Predicted columns: {predicted_cols}")

    if categorical_cols:
        analysis.append(f"\n✓ Categorical breakdowns possible by: {categorical_cols[:3]}")

    analysis.append(f"\n" + "=" * 80)

    return "\n".join(analysis)


@tool
def plan_visualization(
    thought: str,
    plot_type: str,
    columns_to_use: List[str],
    purpose: str,
    why_this_plot: str
) -> str:
    """Plan a single visualization before creating it.

    Use this tool to explicitly think through what plot you want to make and why.
    This is part of the ReAct framework - reasoning before acting.

    Args:
        thought: Your reasoning about what you've learned from the data so far
        plot_type: Type of plot (e.g., 'line', 'bar', 'scatter', 'heatmap', 'residual')
        columns_to_use: List of column names that will be used in this plot
        purpose: What this plot aims to show (e.g., 'Compare predictions vs actual', 'Show temporal trends')
        why_this_plot: Justification for why this specific plot type and columns will achieve the purpose

    Returns:
        Confirmation with the plan summary
    """
    plan = f"""
📋 VISUALIZATION PLAN
{'=' * 80}
💭 Reasoning: {thought}

📊 Plot Details:
  - Type: {plot_type}
  - Columns: {', '.join(columns_to_use)}

🎯 Purpose: {purpose}

💡 Rationale: {why_this_plot}
{'=' * 80}

✓ Plan recorded. Proceed to create this visualization using create_visualizations().
"""
    return plan


@tool
def create_plot_with_explanation(
    code: str,
    plot_number: int,
    plot_title: str,
    what_it_shows: str,
    what_was_given: str,
    what_was_predicted: str,
    key_insights: str
) -> str:
    """Execute visualization code and record detailed explanation.

    This tool combines code execution with explicit documentation of what the plot shows.

    Args:
        code: Python code to create and save the plot
        plot_number: Sequential number for this plot (1, 2, 3, ...)
        plot_title: Title of the plot
        what_it_shows: What this visualization displays
        what_was_given: Which columns/data were provided as input (given data)
        what_was_predicted: Which columns/data were predicted by the model
        key_insights: What insights can be drawn from this plot

    Returns:
        Execution result with saved plot path
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Try to import plotly (optional)
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        has_plotly = True
    except ImportError:
        px = None
        go = None
        has_plotly = False

    def load_dataframe_viz(filepath):
        """Load a dataframe from any supported format."""
        filepath = Path(filepath)
        if not filepath.exists():
            filepath = STAGE4_OUT_DIR / filepath.name

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        return load_dataframe(filepath, base_dir=STAGE4_OUT_DIR)

    globals_dict = {
        "__name__": "__stage5_plot_maker__",
        "pd": pd,
        "np": np,
        "plt": plt,
        "sns": sns,
        "json": json,
        "Path": Path,
        "STAGE4_OUT_DIR": STAGE4_OUT_DIR,
        "STAGE5_OUT_DIR": STAGE5_OUT_DIR,
        "STAGE5_WORKSPACE": STAGE5_WORKSPACE,
        "load_dataframe": load_dataframe_viz,
        "plot_number": plot_number,
    }

    if has_plotly:
        globals_dict.update({"px": px, "go": go})

    buf = io.StringIO()

    try:
        with contextlib.redirect_stdout(buf):
            print(f"=" * 80)
            print(f"PLOT {plot_number}: {plot_title}")
            print(f"=" * 80)
            print(f"\n📊 What it shows: {what_it_shows}")
            print(f"\n📥 Given data: {what_was_given}")
            print(f"\n🔮 Predicted data: {what_was_predicted}")
            print(f"\n💡 Key insights: {key_insights}")
            print(f"\n{'=' * 80}")
            print(f"Executing plot code...")
            print(f"{'=' * 80}\n")

            # Execute the plotting code
            exec(code, globals_dict, globals_dict)

            # Close any open plots
            plt.close('all')

            print(f"\n✓ Plot {plot_number} created successfully")
    except Exception as e:
        plt.close('all')
        import traceback
        error_details = traceback.format_exc()
        return f"[ERROR] Plot {plot_number} failed: {e}\n\n{error_details}"

    output = buf.getvalue()

    # Also save the explanation as a text file
    explanation_path = STAGE5_OUT_DIR / f"plot_{plot_number}_explanation.txt"
    explanation = f"""
PLOT {plot_number}: {plot_title}
{'=' * 80}

What it shows:
{what_it_shows}

Given data (from original dataset):
{what_was_given}

Predicted data (from model output):
{what_was_predicted}

Key insights:
{key_insights}

{'=' * 80}
"""
    explanation_path.write_text(explanation)

    return output + f"\n\n✓ Explanation saved to: {explanation_path}"


@tool
def save_visualization_report(report_json: str) -> str:
    """Save the final visualization report.

    Args:
        report_json: JSON string containing:
            - plan_id: ID of the executed plan
            - task_category: descriptive/predictive/unsupervised
            - visualizations: list of created visualization paths
            - html_report: path to HTML report (if created)
            - summary: text summary of visualizations
            - insights: key insights from the visualizations

    Returns:
        Confirmation message with save path
    """
    from datetime import datetime

    try:
        report = json.loads(report_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")

    plan_id = report.get("plan_id", "UNKNOWN")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_path = STAGE5_OUT_DIR / f"visualization_report_{plan_id}_{timestamp}.json"
    output_path.write_text(json.dumps(report, indent=2))

    return f"✅ Visualization report saved to: {output_path}"


# Stage 5 tool list
STAGE5_TOOLS = [
    list_stage4_results,
    load_stage4_result,
    load_stage3_plan_viz,
    analyze_data_columns,  # NEW: ReAct - Analyze what we have
    plan_visualization,  # NEW: ReAct - Plan before plotting
    create_visualizations,  # Keep for backward compatibility
    create_plot_with_explanation,  # NEW: ReAct - Plot with full explanation
    save_visualization_report,
]

# ===========================
# Failsafe / Debugging Tools
# ===========================

FAILSAFE_TOOLS = [
    failsafe_python,
    search,
    list_data_files,
    inspect_data_file,
]


# ===========================
# Complete Tool Registry
# ===========================

ALL_TOOLS = {
    "stage2": STAGE2_TOOLS,
    "stage3": STAGE3_TOOLS,
    "stage3_5": STAGE3_5_TOOLS,
    "stage4": STAGE4_TOOLS,
    "stage5": STAGE5_TOOLS,
    "failsafe": FAILSAFE_TOOLS,
}


# ===========================
# Stage 0: Conversational Tools
# ===========================

@tool
def trigger_pipeline_stages(
    start_stage: int,
    end_stage: int,
    task_id: Optional[str] = None,
    user_query: Optional[str] = None,
) -> str:
    """Triggers execution of pipeline stages (1-5) based on the conversational query.
    
    Args:
        start_stage: Stage to start from (1-5)
        end_stage: Stage to end at (1-5)
        task_id: Optional task ID for Stages 3+
        user_query: Optional user request to guide proposal generation (Stage 2)
        
    Returns:
        Execution summary string
    """
    from .master_agent import run_partial_pipeline
    
    try:
        # Run the pipeline
        state = run_partial_pipeline(start_stage, end_stage, task_id, user_query=user_query)
        
        # Format summary based on what ran
        summary = []
        summary.append(f"✅ Pipeline stages {start_stage}-{end_stage} completed successfully.")
        
        if state.get("dataset_summaries"):
            summary.append(f"- Generated {len(state['dataset_summaries'])} dataset summaries")
            
        if state.get("task_proposals"):
            summary.append(f"- Generated {len(state['task_proposals'])} task proposals:")
            for p in state['task_proposals']:
                summary.append(f"  * [{p.id}] {p.category}: {p.title}")
            
        if state.get("stage3_plan"):
            summary.append(f"- Created execution plan: {state['stage3_plan'].plan_id}")
            
        if state.get("execution_result"):
            status = state['execution_result'].status
            summary.append(f"- Execution finished with status: {status}")
            
        if state.get("visualization_report"):
            viz_count = len(state['visualization_report'].visualizations)
            summary.append(f"- Created {viz_count} visualizations")
            
        return "\n".join(summary)
        
    except Exception as e:
        return f"❌ Pipeline execution failed: {str(e)}"


@tool
def query_data_capabilities() -> str:
    """Returns a summary of available datasets and what predictions/analyses are possible.
    
    Checks for existing Stage 1 summaries and Stage 2 task proposals.
    If not available, triggers Stage 1-2 automatically.
    
    Returns:
        Text summary of capabilities
    """
    from .master_agent import run_partial_pipeline
    
    # Check if we have summaries
    summaries = _list_summary_files()
    
    if not summaries:
        return "No data summaries found. Please run 'trigger_pipeline_stages(1, 2)' first to analyze the data."
        
    # Read summaries
    summary_texts = []
    for s in summaries:
        try:
            content = _read_summary_file(s)
            data = json.loads(content)
            summary_texts.append(f"- {data.get('filename', s)}: {data.get('description', 'No description')}")
        except:
            summary_texts.append(f"- {s}")
            
    # Check for proposals
    proposals_path = STAGE2_OUT_DIR / "task_proposals.json"
    proposals_text = ""
    
    if proposals_path.exists():
        try:
            data = json.loads(proposals_path.read_text())
            stage2 = Stage2Output.model_validate(data)
            proposals_text = "\n\nAvailable Analysis Tasks:\n"
            for p in stage2.proposals:
                proposals_text += f"- [{p.id}] {p.category}: {p.title}\n"
        except Exception as e:
            proposals_text = f"\n(Could not read existing proposals: {e})"
    else:
        proposals_text = "\n(No specific task proposals generated yet)"
        
    return f"Data Capabilities:\n\nDatasets:\n" + "\n".join(summary_texts) + proposals_text


@tool
def execute_dynamic_analysis(question: str, code: str, description: str) -> str:
    """General-purpose tool for running custom analysis code.
    
    The agent generates code based on the user's question.
    Has access to all data files and can create ad-hoc predictions.
    
    Args:
        question: The user's original question
        code: Python code to execute
        description: Brief description of the analysis
        
    Returns:
        Results as formatted text suitable for conversation
    """
    # Reuse the Stage 4 execution environment but with a focus on immediate text output
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    
    def load_dataframe_helper(filename: str):
        return load_dataframe(filename, base_dir=DATA_DIR)
    
    globals_dict = {
        "__name__": "__dynamic_analyzer__",
        "pd": pd,
        "np": np,
        "json": json,
        "Path": Path,
        "DATA_DIR": DATA_DIR,
        "load_dataframe": load_dataframe_helper,
        "plt": plt,
        "LinearRegression": LinearRegression,
        "RandomForestRegressor": RandomForestRegressor,
        "mean_squared_error": mean_squared_error,
        "train_test_split": train_test_split,
    }
    
    local_env = {}
    buf = io.StringIO()
    
    try:
        with contextlib.redirect_stdout(buf):
            print(f"=== Analysis: {description} ===")
            exec(code, globals_dict, local_env)
    except Exception as e:
        return f"❌ Analysis failed: {e}"
    
    return buf.getvalue() or "[Analysis finished with no output]"


@tool
def get_conversation_context() -> str:
    """Returns current conversation state (query history, completed analyses, cached results).
    
    Returns:
        JSON string of conversation state
    """
    state_path = OUTPUT_ROOT / "conversation_state.json"
    if not state_path.exists():
        return json.dumps({
            "history": [],
            "completed_tasks": [],
            "last_updated": "never"
        })
    return state_path.read_text()


@tool
def save_conversation_state(state_json: str) -> str:
    """Saves current conversation state for persistence.
    
    Args:
        state_json: JSON string of state
        
    Returns:
        Confirmation message
    """
    try:
        # Validate valid JSON
        json.loads(state_json)
        state_path = OUTPUT_ROOT / "conversation_state.json"
        state_path.write_text(state_json)
        return "✅ Conversation state saved"
    except Exception as e:
        return f"❌ Failed to save state: {e}"


# Stage 0 tool list
STAGE0_TOOLS = [
    trigger_pipeline_stages,
    query_data_capabilities,
    execute_dynamic_analysis,
    get_conversation_context,
    save_conversation_state,
]

# Update ALL_TOOLS
ALL_TOOLS["stage0"] = STAGE0_TOOLS
