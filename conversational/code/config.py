"""
Configuration for the Conversational AI Pipeline.

This module provides centralized configuration for:
- LLM settings and endpoints
- Directory paths for data and outputs
- Stage-specific parameters
- Robust data passing mechanisms
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import json
import hashlib
from datetime import datetime

# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", "/scratch/ziv_baretto/conversational_agent/Test_Capstone/conversational"))
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_ROOT = PROJECT_ROOT / "output"

# Stage-specific output directories
SUMMARIES_DIR = OUTPUT_ROOT / "summaries"
STAGE2_OUT_DIR = OUTPUT_ROOT / "stage2_out"
STAGE3_OUT_DIR = OUTPUT_ROOT / "stage3_out"
STAGE3B_OUT_DIR = OUTPUT_ROOT / "stage3b_data_prep"
STAGE3_5A_OUT_DIR = OUTPUT_ROOT / "stage3_5a_method_proposal"
STAGE3_5B_OUT_DIR = OUTPUT_ROOT / "stage3_5b_benchmarking"
STAGE4_OUT_DIR = OUTPUT_ROOT / "stage4_out"
STAGE4_WORKSPACE = OUTPUT_ROOT / "stage4_workspace"
STAGE5_OUT_DIR = OUTPUT_ROOT / "stage5_out"
STAGE5_WORKSPACE = OUTPUT_ROOT / "stage5_workspace"

# Conversation state directory
CONVERSATION_STATE_DIR = OUTPUT_ROOT / "conversation_state"

# Create all directories
for d in [SUMMARIES_DIR, STAGE2_OUT_DIR, STAGE3_OUT_DIR, STAGE3B_OUT_DIR,
          STAGE3_5A_OUT_DIR, STAGE3_5B_OUT_DIR, STAGE4_OUT_DIR, STAGE4_WORKSPACE,
          STAGE5_OUT_DIR, STAGE5_WORKSPACE, CONVERSATION_STATE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================================
# LLM CONFIGURATION
# ============================================================================

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:8001/v1")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "EMPTY")

# Primary LLM config (for complex reasoning tasks)
PRIMARY_LLM_CONFIG = {
    "base_url": LLM_BASE_URL,
    "api_key": LLM_API_KEY,
    "model": "Qwen/Qwen2.5-32B-Instruct",
    "temperature": 0.0,
    "max_tokens": 4096,
}

# Secondary LLM config (for tool-calling agents)
SECONDARY_LLM_CONFIG = {
    "base_url": LLM_BASE_URL,
    "api_key": LLM_API_KEY,
    "model": "Qwen/Qwen3-32B",
    "temperature": 0.0,
    "max_tokens": 8192,
}

# Conversation LLM config (for user interaction)
CONVERSATION_LLM_CONFIG = {
    "base_url": LLM_BASE_URL,
    "api_key": LLM_API_KEY,
    "model": "Qwen/Qwen3-32B",
    "temperature": 0.2,  # Slightly higher for more natural conversation
    "max_tokens": 2048,
}

# ============================================================================
# STAGE-SPECIFIC PARAMETERS
# ============================================================================

# Maximum iterations per stage
STAGE_MAX_ROUNDS = {
    "stage1": 1,      # Direct execution
    "stage2": 15,     # Exploration steps
    "stage3": 30,     # Planning rounds
    "stage3b": 100,   # Data preparation
    "stage3_5a": 35,  # Method proposal
    "stage3_5b": 120, # Benchmarking (3 methods x 3 iterations each)
    "stage4": 100,    # Execution
    "stage5": 60,     # Visualization
}

# Stage 1 sampling
STAGE1_SAMPLE_ROWS = 5000

# Data quality thresholds
MIN_NON_NULL_FRACTION = 0.65  # Minimum 65% non-null for columns
MIN_UNIQUE_FRACTION = 0.01   # Minimum 1% unique values

# Benchmarking parameters
BENCHMARK_ITERATIONS = 3
MAX_CV_THRESHOLD = 0.3  # Maximum coefficient of variation for valid results

# Global recursion limit for LangGraph
RECURSION_LIMIT = 200

# ============================================================================
# ROBUST DATA PASSING MECHANISMS
# ============================================================================

class DataPassingManager:
    """
    Manages robust data passing between pipeline stages.

    Features:
    - Atomic file operations with temp files
    - Checksum verification
    - Automatic retry on failure
    - Structured metadata for all artifacts
    """

    @staticmethod
    def generate_artifact_id(stage: str, task_id: str = None) -> str:
        """Generate unique artifact ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_part = f"_{task_id}" if task_id else ""
        return f"{stage}{task_part}_{timestamp}"

    @staticmethod
    def compute_checksum(data: Any) -> str:
        """Compute SHA256 checksum of data."""
        if isinstance(data, dict):
            data_str = json.dumps(data, sort_keys=True, default=str)
        else:
            data_str = str(data)
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]

    @staticmethod
    def save_artifact(
        data: Dict[str, Any],
        output_dir: Path,
        filename: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save artifact with atomic write and metadata.

        Uses temp file + rename for atomic operation to prevent
        partial writes on failure.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Prepare envelope with metadata
        envelope = {
            "_meta": {
                "timestamp": datetime.now().isoformat(),
                "checksum": DataPassingManager.compute_checksum(data),
                "version": "1.0",
                **(metadata or {})
            },
            "data": data
        }

        # Atomic write: temp file then rename
        output_path = output_dir / filename
        temp_path = output_dir / f".tmp_{filename}"

        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(envelope, f, indent=2, default=str, ensure_ascii=False)
            temp_path.rename(output_path)
            return output_path
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise RuntimeError(f"Failed to save artifact {filename}: {e}")

    @staticmethod
    def load_artifact(
        input_path: Path,
        verify_checksum: bool = True
    ) -> Dict[str, Any]:
        """
        Load artifact with optional checksum verification.

        Returns the data portion of the envelope.
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Artifact not found: {input_path}")

        with open(input_path, 'r', encoding='utf-8') as f:
            envelope = json.load(f)

        # Handle both wrapped and unwrapped formats
        if "_meta" in envelope and "data" in envelope:
            data = envelope["data"]
            if verify_checksum:
                expected_checksum = envelope["_meta"].get("checksum")
                actual_checksum = DataPassingManager.compute_checksum(data)
                if expected_checksum and expected_checksum != actual_checksum:
                    raise ValueError(f"Checksum mismatch for {input_path}")
            return data
        else:
            # Legacy format without envelope
            return envelope

    @staticmethod
    def save_parquet_with_metadata(
        df: "pd.DataFrame",
        output_path: Path,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Save parquet file with sidecar metadata JSON."""
        import pandas as pd

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save parquet
        df.to_parquet(output_path, index=False)

        # Save sidecar metadata
        meta_path = output_path.with_suffix('.meta.json')
        meta = {
            "timestamp": datetime.now().isoformat(),
            "shape": list(df.shape),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "null_counts": df.isnull().sum().to_dict(),
            **(metadata or {})
        }
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, default=str)

        return output_path


class StageTransition:
    """
    Manages transitions between pipeline stages with validation.

    Ensures data integrity and provides clear error messages
    when transitions fail.
    """

    # Define expected inputs and outputs for each stage
    STAGE_CONTRACTS = {
        "stage1": {
            "inputs": [],
            "outputs": ["summaries"],
            "output_dir": SUMMARIES_DIR,
        },
        "stage2": {
            "inputs": ["summaries"],
            "outputs": ["task_proposals"],
            "output_dir": STAGE2_OUT_DIR,
        },
        "stage3": {
            "inputs": ["task_proposals"],
            "outputs": ["execution_plan"],
            "output_dir": STAGE3_OUT_DIR,
        },
        "stage3b": {
            "inputs": ["execution_plan"],
            "outputs": ["prepared_data"],
            "output_dir": STAGE3B_OUT_DIR,
        },
        "stage3_5a": {
            "inputs": ["execution_plan", "prepared_data"],
            "outputs": ["method_proposals"],
            "output_dir": STAGE3_5A_OUT_DIR,
        },
        "stage3_5b": {
            "inputs": ["method_proposals", "prepared_data"],
            "outputs": ["benchmark_results"],
            "output_dir": STAGE3_5B_OUT_DIR,
        },
        "stage4": {
            "inputs": ["execution_plan", "prepared_data", "benchmark_results"],
            "outputs": ["execution_results"],
            "output_dir": STAGE4_OUT_DIR,
        },
        "stage5": {
            "inputs": ["execution_results", "execution_plan"],
            "outputs": ["visualizations"],
            "output_dir": STAGE5_OUT_DIR,
        },
    }

    @classmethod
    def validate_transition(cls, from_stage: str, to_stage: str, task_id: str) -> Dict[str, Any]:
        """
        Validate that a stage transition is possible.

        Returns dict with:
        - valid: bool
        - missing_inputs: list of missing required inputs
        - available_inputs: dict of available input paths
        """
        to_contract = cls.STAGE_CONTRACTS.get(to_stage)
        if not to_contract:
            return {"valid": False, "error": f"Unknown stage: {to_stage}"}

        available_inputs = {}
        missing_inputs = []

        for required_input in to_contract["inputs"]:
            input_path = cls._find_input(required_input, task_id)
            if input_path:
                available_inputs[required_input] = input_path
            else:
                missing_inputs.append(required_input)

        return {
            "valid": len(missing_inputs) == 0,
            "missing_inputs": missing_inputs,
            "available_inputs": available_inputs,
        }

    @classmethod
    def _find_input(cls, input_type: str, task_id: str) -> Optional[Path]:
        """Find input file for a given type."""
        patterns = {
            "summaries": (SUMMARIES_DIR, "*.summary.json"),
            "task_proposals": (STAGE2_OUT_DIR, "task_proposals.json"),
            "execution_plan": (STAGE3_OUT_DIR, f"PLAN-{task_id}.json"),
            "prepared_data": (STAGE3B_OUT_DIR, f"prepared_PLAN-{task_id}.parquet"),
            "method_proposals": (STAGE3_5A_OUT_DIR, f"method_proposal_PLAN-{task_id}.json"),
            "benchmark_results": (STAGE3_5B_OUT_DIR, f"tester_PLAN-{task_id}.json"),
            "execution_results": (STAGE4_OUT_DIR, f"execution_result_PLAN-{task_id}.json"),
        }

        if input_type not in patterns:
            return None

        directory, pattern = patterns[input_type]
        matches = list(Path(directory).glob(pattern))

        if matches:
            # Return most recent match
            return max(matches, key=lambda p: p.stat().st_mtime)
        return None


# ============================================================================
# JSON SANITIZATION FOR RELIABLE PARSING
# ============================================================================

class JSONSanitizer:
    """
    Sanitizes LLM output to extract valid JSON.

    Handles common LLM output issues:
    - Markdown code blocks
    - Trailing text after JSON
    - Unescaped characters
    - Python-style booleans/None
    """

    @staticmethod
    def extract_json(text: str) -> Optional[Dict[str, Any]]:
        """
        Extract first valid JSON object from text.

        Tries multiple strategies:
        1. Direct JSON parse
        2. Strip markdown code blocks
        3. Find JSON between braces
        4. Fix common issues and retry
        """
        if not text or not text.strip():
            return None

        strategies = [
            JSONSanitizer._try_direct_parse,
            JSONSanitizer._try_strip_markdown,
            JSONSanitizer._try_extract_braces,
            JSONSanitizer._try_fix_common_issues,
        ]

        for strategy in strategies:
            try:
                result = strategy(text)
                if result is not None:
                    return result
            except Exception:
                continue

        return None

    @staticmethod
    def _try_direct_parse(text: str) -> Optional[Dict]:
        """Try direct JSON parse."""
        return json.loads(text.strip())

    @staticmethod
    def _try_strip_markdown(text: str) -> Optional[Dict]:
        """Strip markdown code blocks and parse."""
        import re
        # Remove ```json ... ``` or ``` ... ```
        pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
        match = re.search(pattern, text)
        if match:
            return json.loads(match.group(1))
        return None

    @staticmethod
    def _try_extract_braces(text: str) -> Optional[Dict]:
        """Extract JSON between outermost braces."""
        # Find first { and last }
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end+1])
        return None

    @staticmethod
    def _try_fix_common_issues(text: str) -> Optional[Dict]:
        """Fix common LLM JSON issues."""
        import re

        # Extract between braces first
        start = text.find('{')
        end = text.rfind('}')
        if start == -1 or end == -1:
            return None

        json_str = text[start:end+1]

        # Fix Python-style booleans and None
        json_str = re.sub(r'\bTrue\b', 'true', json_str)
        json_str = re.sub(r'\bFalse\b', 'false', json_str)
        json_str = re.sub(r'\bNone\b', 'null', json_str)

        # Fix trailing commas
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)

        # Fix unquoted keys (simple cases)
        json_str = re.sub(r'(\s)([a-zA-Z_][a-zA-Z0-9_]*)(\s*:)', r'\1"\2"\3', json_str)

        return json.loads(json_str)

    @staticmethod
    def to_safe_json_string(obj: Any) -> str:
        """Convert object to JSON string with proper escaping."""
        return json.dumps(obj, indent=2, default=str, ensure_ascii=False)


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

import logging

# Debug configuration
DEBUG = os.environ.get("DEBUG", "True").lower() == "true"

def setup_logging(level: str = None) -> logging.Logger:
    """Setup logging for the pipeline."""
    if level is None:
        level = "DEBUG" if DEBUG else "INFO"
        
    logger = logging.getLogger("conversational_pipeline")
    logger.setLevel(getattr(logging, level.upper()))

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(handler)

    return logger


# Initialize logger
logger = setup_logging()


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Paths
    "PROJECT_ROOT", "DATA_DIR", "OUTPUT_ROOT",
    "SUMMARIES_DIR", "STAGE2_OUT_DIR", "STAGE3_OUT_DIR",
    "STAGE3B_OUT_DIR", "STAGE3_5A_OUT_DIR", "STAGE3_5B_OUT_DIR",
    "STAGE4_OUT_DIR", "STAGE4_WORKSPACE", "STAGE5_OUT_DIR", "STAGE5_WORKSPACE",
    "CONVERSATION_STATE_DIR",
    # LLM configs
    "PRIMARY_LLM_CONFIG", "SECONDARY_LLM_CONFIG", "CONVERSATION_LLM_CONFIG",
    # Parameters
    "STAGE_MAX_ROUNDS", "STAGE1_SAMPLE_ROWS",
    "MIN_NON_NULL_FRACTION", "MIN_UNIQUE_FRACTION",
    "BENCHMARK_ITERATIONS", "MAX_CV_THRESHOLD", "RECURSION_LIMIT",
    # Classes
    "DataPassingManager", "StageTransition", "JSONSanitizer",
    # Utilities
    "logger", "setup_logging", "DEBUG",
]
