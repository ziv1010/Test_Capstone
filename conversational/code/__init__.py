"""
Conversational AI Pipeline - Code Module

This module contains the core components of the conversational AI pipeline.
"""

from .config import (
    PROJECT_ROOT, DATA_DIR, OUTPUT_ROOT,
    SUMMARIES_DIR, STAGE2_OUT_DIR, STAGE3_OUT_DIR,
    STAGE3B_OUT_DIR, STAGE3_5A_OUT_DIR, STAGE3_5B_OUT_DIR,
    STAGE4_OUT_DIR, STAGE5_OUT_DIR,
    PRIMARY_LLM_CONFIG, SECONDARY_LLM_CONFIG,
    DataPassingManager, StageTransition, JSONSanitizer,
    logger
)

from .models import (
    TaskCategory, StageStatus, ExecutionStatus,
    ColumnSummary, DatasetSummary, Stage1Output,
    TaskProposal, Stage2Output,
    Stage3Plan, PreparedDataOutput,
    MethodProposalOutput, TesterOutput,
    ExecutionResult, VisualizationReport,
    PipelineState, ConversationContext
)

from .utils import (
    profile_csv, profile_all_datasets,
    list_data_files, list_summary_files, load_dataframe,
    execute_python_sandbox, extract_json_block,
    validate_dataframe, prepare_dataframe_for_modeling
)

__all__ = [
    # Config
    "PROJECT_ROOT", "DATA_DIR", "OUTPUT_ROOT",
    "PRIMARY_LLM_CONFIG", "SECONDARY_LLM_CONFIG",
    "DataPassingManager", "StageTransition", "JSONSanitizer",
    "logger",
    # Models
    "TaskCategory", "StageStatus", "ExecutionStatus",
    "DatasetSummary", "TaskProposal", "Stage3Plan",
    "PipelineState", "ConversationContext",
    # Utils
    "profile_csv", "list_data_files", "load_dataframe",
    "execute_python_sandbox",
]
