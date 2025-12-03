"""
Pydantic models for all stages of the unified agentic AI pipeline.

Contains data models for Stage 1-5 including dataset summaries, task proposals,
execution plans, results, and visualization reports.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Literal, Tuple, TypedDict
from datetime import datetime
from pydantic import BaseModel, Field

# ===========================
# Stage 1: Dataset Summarization
# ===========================

LogicalType = Literal[
    "numeric", "integer", "float", "categorical",
    "text", "datetime", "boolean", "unknown"
]


class ColumnSummary(BaseModel):
    """Summary of a single column in a dataset."""
    name: str
    physical_dtype: str
    logical_type: LogicalType
    description: str = Field(
        description="Short natural-language description of what the column represents."
    )
    nullable: bool
    null_fraction: float
    unique_fraction: float
    examples: List[str] = Field(default_factory=list)
    is_potential_key: bool = False


class DatasetSummary(BaseModel):
    """Summary of an entire dataset."""
    dataset_name: str
    path: str
    approx_n_rows: Optional[int] = None
    columns: List[ColumnSummary]
    candidate_primary_keys: List[List[str]] = Field(
        default_factory=list,
        description="Each entry is a list of column names that could form a primary key."
    )
    notes: Optional[str] = None


# ===========================
# Stage 2: Task Proposal
# ===========================

TaskCategory = Literal["predictive", "descriptive", "unsupervised"]


class JoinPlan(BaseModel):
    """Plan for joining datasets."""
    hypothesized_keys: List[List[str]] = Field(
        default_factory=list,
        description="Each inner list is a set of columns that might be join keys between files."
    )
    notes: Optional[str] = None


class TargetSpec(BaseModel):
    """Specification for prediction target."""
    name: Optional[str] = None
    granularity: Optional[List[str]] = None
    horizon: Optional[str] = None  # e.g. '1-year ahead'


class FeaturePlan(BaseModel):
    """Plan for feature engineering."""
    candidates: List[str] = Field(
        default_factory=list,
        description="Column names or wildcard patterns (e.g. 'Area-*')."
    )
    transform_ideas: List[str] = Field(
        default_factory=list,
        description="Free-text feature engineering ideas."
    )
    quality_checks: List[str] = Field(
        default_factory=list,
        description="Checks for avoiding leakage, broken joins, etc."
    )
    excluded_columns: List[Dict[str, str]] = Field(
        default_factory=list,
        description=(
            "Columns that were considered but excluded due to data quality issues. "
            "Each entry has 'column_name', 'file', and 'reason' (e.g., 'Only 45% non-NaN data, below 65% threshold')"
        )
    )
    expected_outputs: List[str] = Field(
        default_factory=list,
        description="Expected output types like tables, plots, metrics, etc."
    )
    handling_missingness: Optional[str] = None


class TaskProposal(BaseModel):
    """Proposed analytical task."""
    id: str
    category: TaskCategory
    title: str
    problem_statement: str
    required_files: List[str] = Field(
        default_factory=list,
        description="Dataset filenames needed for this task."
    )
    join_plan: JoinPlan = Field(default_factory=JoinPlan)
    target: Optional[TargetSpec] = None
    feature_plan: FeaturePlan = Field(default_factory=FeaturePlan)
    validation_plan: Optional[str] = None
    quality_checks: List[str] = Field(default_factory=list)
    expected_outputs: List[str] = Field(default_factory=list)


class Stage2Output(BaseModel):
    """Output from Stage 2 containing all task proposals."""
    proposals: List[TaskProposal]


# ===========================
# Stage 3: Execution Planning
# ===========================

class ArtifactSpec(BaseModel):
    """Specification for output artifacts."""
    intermediate_table: str = Field(description="Filename for the output table")
    intermediate_format: Literal["parquet", "csv", "feather"] = "parquet"
    expected_columns: List[str] = Field(
        default_factory=list,
        description="List of all columns expected in final table"
    )
    expected_row_count_range: Optional[Tuple[int, int]] = Field(
        default=None,
        description="Expected min/max row count"
    )


class KeyNormalization(BaseModel):
    """Normalization rules for join keys."""
    column_name: str
    mapping: Dict[str, str] = Field(default_factory=dict)
    format_type: Optional[str] = None
    valid_range: Optional[Tuple[Any, Any]] = None


class FileInstruction(BaseModel):
    """Instructions for loading and preprocessing a file."""
    file_id: str
    original_name: str
    alias: str
    filters: List[str] = Field(default_factory=list)
    rename_columns: Dict[str, str] = Field(default_factory=dict)
    join_keys: List[str] = Field(default_factory=list)
    keep_columns: List[str] = Field(default_factory=list)
    notes: Optional[str] = None


class JoinValidation(BaseModel):
    """Validation rules for joins."""
    check_duplicates_on_keys: List[str] = Field(default_factory=list)
    expected_unique: bool = False
    check_row_count_stable: bool = False
    check_no_duplicates_introduced: bool = False
    acceptable_coverage: Optional[float] = None
    max_cardinality_ratio: Optional[float] = None


class JoinStep(BaseModel):
    """A single join operation."""
    step: int
    description: str
    left_table: str
    right_table: Optional[str] = None
    join_type: Literal["base", "inner", "left", "right", "outer"] = "base"
    join_keys: List[str] = Field(default_factory=list)
    left_on: List[str] = Field(default_factory=list, description="Keys for left table if different from right")
    right_on: List[str] = Field(default_factory=list, description="Keys for right table if different from left")
    expected_cardinality: str = "base"
    validation: JoinValidation = Field(default_factory=JoinValidation)


class FeatureEngineering(BaseModel):
    """Feature engineering specification."""
    feature_name: str
    description: str
    transform: str
    depends_on: List[str] = Field(default_factory=list)
    implementation: Optional[str] = None


class TimeSplit(BaseModel):
    """Time-based train/test split specification."""
    method: Literal["year-based", "date-based", "rolling-window", "none"] = "none"
    train_years: Optional[str] = None
    test_years: Optional[str] = None
    validation_years: Optional[str] = None
    leakage_check: str = "Not applicable"


class CoverageCheck(BaseModel):
    """Data coverage validation check."""
    check: str
    threshold: float
    description: str
    action_if_violation: Optional[str] = None


class CardinalityCheck(BaseModel):
    """Cardinality validation check."""
    check: str
    expected: str
    action_if_violation: str


class ValidationSpec(BaseModel):
    """Validation strategy for the pipeline."""
    time_split: Optional[TimeSplit] = Field(default_factory=lambda: TimeSplit())
    coverage_checks: List[CoverageCheck] = Field(default_factory=list)
    cardinality_checks: List[CardinalityCheck] = Field(default_factory=list)
    additional_checks: List[str] = Field(default_factory=list)


class Stage3Plan(BaseModel):
    """Complete execution plan from Stage 3."""
    # Metadata
    plan_id: str
    selected_task_id: str
    goal: str
    task_category: TaskCategory
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    created_by: str = Field(default="stage3_agent")
    
    # Output specification
    artifacts: ArtifactSpec
    
    # Key normalization
    key_normalization: List[KeyNormalization] = Field(default_factory=list)
    
    # File loading instructions
    file_instructions: List[FileInstruction] = Field(default_factory=list)
    
    # Join strategy
    join_steps: List[JoinStep] = Field(default_factory=list)
    
    # Feature engineering
    feature_engineering: List[FeatureEngineering] = Field(default_factory=list)
    
    # Validation strategy
    validation: ValidationSpec = Field(default_factory=ValidationSpec)
    
    # Expected models and metrics
    expected_model_types: List[str] = Field(default_factory=list)
    evaluation_metrics: List[str] = Field(default_factory=list)
    
    # Documentation
    notes: List[str] = Field(default_factory=list)


# ===========================
# Stage 3B: Data Preparation
# ===========================

class PreparedDataOutput(BaseModel):
    """Output from Stage 3B data preparation agent."""
    plan_id: str = Field(description="Links back to Stage 3 plan")
    prepared_file_path: str = Field(description="Path to saved prepared data file (parquet)")
    original_row_count: int = Field(description="Number of rows before preparation")
    prepared_row_count: int = Field(description="Number of rows after preparation")
    columns_created: List[str] = Field(
        default_factory=list,
        description="Names of feature engineering columns created"
    )
    transformations_applied: List[str] = Field(
        default_factory=list,
        description="List of transformations applied (filters, joins, feature engineering)"
    )
    data_quality_report: Dict[str, Any] = Field(
        default_factory=dict,
        description="Data quality metrics (null counts, duplicates, etc.)"
    )
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


# ===========================
# Stage 3.5: Method Testing & Benchmarking
# ===========================

class Stage3_5Checkpoint(BaseModel):
    """Checkpoint for Stage 3.5 to maintain memory across conversation truncation.
    
    Simplified strategy: Save only when a method completes all 3 iterations
    with consistent results (CV < 0.3).
    """
    plan_id: str = Field(description="Plan ID being tested")

    # Data split information (to maintain consistency across all methods)
    data_split_strategy: str = Field(
        description="How data was split for benchmarking, e.g., '2020-2023 train, 2024 validation'"
    )
    date_column: Optional[str] = Field(
        default=None,
        description="Name of the date/time column identified in the data"
    )
    target_column: Optional[str] = Field(
        default=None,
        description="Name of the target variable column"
    )
    train_period: str = Field(description="Training period, e.g., '2020-2023'")
    validation_period: str = Field(description="Validation period, e.g., '2024'")
    test_period: Optional[str] = Field(default=None, description="Test period if applicable")

    # Methods to test
    methods_to_test: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of methods that need to be tested (serialized ForecastingMethod)"
    )

    # Progress tracking - SIMPLIFIED
    methods_completed: List[str] = Field(
        default_factory=list,
        description="List of method_ids that completed all 3 iterations with consistent results"
    )

    # SIMPLIFIED: Only store final averaged results for completed methods
    completed_results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Final averaged benchmark results for completed methods (serialized BenchmarkResult)"
    )

    # Metadata
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class ForecastingMethod(BaseModel):
    """Specification for a forecasting method to benchmark."""
    method_id: str = Field(description="Unique identifier, e.g., 'METHOD-1'")
    name: str = Field(description="Method name, e.g., 'ARIMA', 'Prophet', 'Linear Regression'")
    description: str = Field(description="Why this method is suitable for the task")
    implementation_code: str = Field(description="Python code snippet to implement this method")
    libraries_required: List[str] = Field(
        default_factory=list,
        description="Required libraries, e.g., ['statsmodels', 'pandas']"
    )


class BenchmarkResult(BaseModel):
    """Results from benchmarking a single forecasting method."""
    method_id: str
    method_name: str
    metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Performance metrics, e.g., {'MAE': 123.45, 'RMSE': 234.56, 'MAPE': 0.12}"
    )
    train_period: str = Field(description="Training period, e.g., '2020-2023'")
    validation_period: str = Field(description="Validation period, e.g., '2024'")
    test_period: Optional[str] = Field(default=None, description="Test period if applicable")
    execution_time_seconds: float
    status: Literal["success", "failure"]
    error_message: Optional[str] = None
    predictions_sample: Optional[List[float]] = Field(
        default=None,
        description="Sample of predictions for inspection"
    )


class MethodProposalOutput(BaseModel):
    """Output from Stage 3.5a Method Proposal."""
    plan_id: str = Field(description="Links back to Stage 3 plan")
    task_category: TaskCategory

    # Methods proposed
    methods_proposed: List[ForecastingMethod] = Field(
        description="All methods proposed for benchmarking (exactly 3)"
    )

    # Data split strategy used
    data_split_strategy: str = Field(
        description="How data will be split for benchmarking, e.g., '2020-2023 train, 2024 validation'"
    )
    date_column: Optional[str] = Field(
        default=None,
        description="Name of the date/time column identified in the data"
    )
    target_column: Optional[str] = Field(
        default=None,
        description="Name of the target variable column"
    )
    train_period: str = Field(description="Training period, e.g., '2020-2023'")
    validation_period: str = Field(description="Validation period, e.g., '2024'")
    test_period: Optional[str] = Field(default=None, description="Test period if applicable")

    # Data preprocessing guide
    data_preprocessing_steps: List[str] = Field(
        default_factory=list,
        description="Ordered list of data preprocessing steps to apply before benchmarking"
    )

    # Metadata
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class TesterOutput(BaseModel):
    """Output from Stage 3.5b Tester (final benchmarking result)."""
    plan_id: str = Field(description="Links back to Stage 3 plan")
    task_category: TaskCategory

    # Methods evaluated
    methods_proposed: List[ForecastingMethod] = Field(
        description="All methods proposed for benchmarking"
    )
    benchmark_results: List[BenchmarkResult] = Field(
        description="Results from benchmarking each method"
    )

    # Selection
    selected_method_id: str = Field(description="ID of the best performing method")
    selected_method: ForecastingMethod = Field(
        description="The winning method to use in final execution"
    )
    selection_rationale: str = Field(
        description="Why this method was selected over others"
    )

    # Data split strategy used
    data_split_strategy: str = Field(
        description="How data was split for benchmarking, e.g., '2020-2023 train, 2024 validation'"
    )

    # Detailed replication guide
    detailed_procedure: str = Field(
        default="",
        description="Step-by-step procedure to replicate the benchmarking and use the selected method"
    )
    data_preprocessing_steps: List[str] = Field(
        default_factory=list,
        description="Ordered list of data preprocessing steps applied before benchmarking"
    )
    method_comparison_summary: str = Field(
        default="",
        description="Summary table or text comparing all methods tested with their metrics"
    )

    # Metadata
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())



class ExecutionResult(BaseModel):
    """Results from Stage 4 execution."""
    plan_id: str
    task_category: TaskCategory
    status: Literal["success", "failure", "partial"]
    outputs: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of output type to file path (e.g., 'predictions': 'path/to/results.parquet')"
    )
    metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Performance metrics (if applicable)"
    )
    summary: str
    errors: List[str] = Field(default_factory=list)
    
    # Enhanced detailed outputs
    output_parquet_path: Optional[str] = Field(
        default=None,
        description="Path to main parquet file containing all table data + predictions for visualization"
    )
    method_used: Optional[str] = Field(
        default=None,
        description="Name/ID of the forecasting method actually used"
    )
    detailed_log: List[str] = Field(
        default_factory=list,
        description="Step-by-step execution log for transparency and debugging"
    )
    data_shape: Optional[Dict[str, int]] = Field(
        default=None,
        description="Final data shape, e.g., {'rows': 1000, 'columns': 25}"
    )
    
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


# ===========================
# Stage 5: Visualization Reports
# ===========================

class VisualizationReport(BaseModel):
    """Report from Stage 5 visualization."""
    plan_id: str
    task_category: TaskCategory
    visualizations: List[str] = Field(
        default_factory=list,
        description="Paths to created visualization files"
    )
    html_report: Optional[str] = Field(
        default=None,
        description="Path to HTML report (if created)"
    )
    expected_columns: List[str] = Field(
        default_factory=list,
        description="Expected columns in the final merged dataset"
    )
    excluded_columns: List[Dict[str, str]] = Field(
        default_factory=list,
        description=(
            "Columns excluded during planning due to data quality. "
            "Each entry has 'column_name', 'file', and 'reason' (e.g., 'Only 50% complete, below 65% threshold')"
        )
    )
    notes: Optional[str] = Field(
        default=None,
        description="Any additional context, warnings, or clarifications"
    )
    summary: str
    insights: List[str] = Field(
        default_factory=list,
        description="Key insights from the visualizations"
    )
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


# ===========================
# Failsafe Agent
# ===========================

class FailsafeRecommendation(BaseModel):
    """Recommendation generated by the failsafe agent when a stage gets stuck."""
    stage: str = Field(description="Name of the stage that requested help")
    error: str = Field(description="Error message or symptom the stage encountered")
    analysis: str = Field(description="Short diagnosis of probable causes")
    next_steps: List[str] = Field(default_factory=list, description="Ordered steps to attempt")
    tool_evidence: Optional[str] = Field(
        default=None,
        description="Key evidence pulled via search or inspection tools"
    )
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


# ===========================
# Master Pipeline State
# ===========================

class PipelineState(BaseModel):
    """Unified state for the entire pipeline."""
    current_stage: int = 1  # Current stage (1-5)
    
    # Stage outputs
    dataset_summaries: List[DatasetSummary] = Field(default_factory=list)
    task_proposals: List[TaskProposal] = Field(default_factory=list)
    selected_task_id: Optional[str] = None
    stage3_plan: Optional[Stage3Plan] = None
    execution_result: Optional[ExecutionResult] = None
    visualization_report: Optional[VisualizationReport] = None
    failsafe_history: List[FailsafeRecommendation] = Field(default_factory=list)
    
    # Tracking
    errors: List[str] = Field(default_factory=list)
    started_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    completed_stages: List[int] = Field(default_factory=list)
    
    class Config:
        arbitrary_types_allowed = True


# ===========================
# Stage 0: Conversation State
# ===========================

class ConversationState(TypedDict):
    """State for conversational interactions."""
    query: str  # Current user question
    conversation_history: List[Dict[str, str]]  # Past Q&A
    available_datasets: List[str]  # Known datasets
    completed_tasks: List[str]  # Tasks already executed
    current_plan: Optional[str]  # Current execution plan
    response: Optional[str]  # Agent's response
