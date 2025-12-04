"""
Pydantic Data Models for the Conversational AI Pipeline.

This module defines all data structures used throughout the pipeline,
with enhanced validation and serialization support.
"""

from typing import Dict, List, Optional, Any, Union, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import datetime
from enum import Enum


# ============================================================================
# ENUMS
# ============================================================================

class TaskCategory(str, Enum):
    """Categories of analytical tasks."""
    FORECASTING = "forecasting"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"
    DESCRIPTIVE = "descriptive"
    OTHER = "other"


class StageStatus(str, Enum):
    """Status of a pipeline stage."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ExecutionStatus(str, Enum):
    """Status of task execution."""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"


class LogicalDataType(str, Enum):
    """Logical data types for columns."""
    NUMERIC = "numeric"
    INTEGER = "integer"
    FLOAT = "float"
    CATEGORICAL = "categorical"
    TEXT = "text"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    UNKNOWN = "unknown"


# ============================================================================
# STAGE 1: DATASET SUMMARIZATION
# ============================================================================

class ColumnSummary(BaseModel):
    """Summary of a single column in a dataset."""
    name: str = Field(..., description="Column name")
    dtype: str = Field(..., description="Pandas dtype")
    logical_type: LogicalDataType = Field(..., description="Inferred logical type")
    null_fraction: float = Field(..., ge=0, le=1, description="Fraction of null values")
    unique_fraction: float = Field(..., ge=0, le=1, description="Fraction of unique values")
    n_unique: int = Field(..., ge=0, description="Number of unique values")
    examples: List[Any] = Field(default_factory=list, description="Sample values")

    # Optional statistics for numeric columns
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean_value: Optional[float] = None
    std_value: Optional[float] = None

    # Optional info for datetime columns
    min_date: Optional[str] = None
    max_date: Optional[str] = None
    date_frequency: Optional[str] = None


class DatasetSummary(BaseModel):
    """Complete summary of a dataset."""
    filename: str = Field(..., description="Dataset filename")
    filepath: str = Field(..., description="Full path to dataset")
    n_rows: int = Field(..., ge=0, description="Number of rows")
    n_cols: int = Field(..., ge=0, description="Number of columns")
    columns: List[ColumnSummary] = Field(default_factory=list)
    candidate_keys: List[str] = Field(default_factory=list, description="Potential primary key columns")
    file_size_mb: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.now)

    # Enhanced metadata
    has_datetime_column: bool = False
    has_target_candidates: List[str] = Field(default_factory=list)
    data_quality_score: Optional[float] = None

    @property
    def column_names(self) -> List[str]:
        """Get list of all column names."""
        return [c.name for c in self.columns]

    @property
    def numeric_columns(self) -> List[str]:
        """Get list of numeric column names."""
        return [c.name for c in self.columns
                if c.logical_type in [LogicalDataType.NUMERIC, LogicalDataType.INTEGER, LogicalDataType.FLOAT]]

    @property
    def datetime_columns(self) -> List[str]:
        """Get list of datetime column names."""
        return [c.name for c in self.columns if c.logical_type == LogicalDataType.DATETIME]


class Stage1Output(BaseModel):
    """Output from Stage 1: Dataset Summarization."""
    summaries: List[DatasetSummary] = Field(default_factory=list)
    total_files_processed: int = 0
    errors: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)


# ============================================================================
# STAGE 2: TASK PROPOSAL
# ============================================================================

class JoinPlan(BaseModel):
    """Plan for joining multiple datasets."""
    datasets: List[str] = Field(..., description="Datasets to join")
    join_keys: Dict[str, str] = Field(..., description="Join keys mapping")
    join_type: str = Field(default="inner", description="Join type (inner, left, right, outer)")
    expected_rows: Optional[int] = None


class FeaturePlan(BaseModel):
    """Plan for feature engineering."""
    name: str = Field(..., description="Feature name")
    source_columns: List[str] = Field(..., description="Columns used to create feature")
    transformation: str = Field(..., description="Transformation description")
    implementation_hint: Optional[str] = None


class ValidationPlan(BaseModel):
    """Plan for data validation and splitting."""
    train_fraction: float = Field(default=0.7, ge=0, le=1)
    validation_fraction: float = Field(default=0.15, ge=0, le=1)
    test_fraction: float = Field(default=0.15, ge=0, le=1)
    split_strategy: str = Field(default="temporal", description="temporal or random")
    date_column: Optional[str] = None

    @model_validator(mode='after')
    def validate_fractions(self):
        total = self.train_fraction + self.validation_fraction + self.test_fraction
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Fractions must sum to 1.0, got {total}")
        return self


class TaskProposal(BaseModel):
    """A proposed analytical task."""
    id: str = Field(..., description="Unique task ID (e.g., TSK-001)")
    category: TaskCategory = Field(..., description="Task category")
    title: str = Field(..., description="Short descriptive title")
    problem_statement: str = Field(..., description="Detailed problem description")

    # Data requirements
    required_datasets: List[str] = Field(default_factory=list)
    target_column: Optional[str] = None
    target_dataset: Optional[str] = None
    feature_columns: List[str] = Field(default_factory=list)

    # Planning
    join_plan: Optional[JoinPlan] = None
    feature_plan: List[FeaturePlan] = Field(default_factory=list)
    validation_plan: Optional[ValidationPlan] = None

    # Feasibility
    feasibility_score: float = Field(default=0.5, ge=0, le=1)
    feasibility_notes: Optional[str] = None

    # For forecasting tasks
    forecast_horizon: Optional[int] = None
    forecast_granularity: Optional[str] = None


class Stage2Output(BaseModel):
    """Output from Stage 2: Task Proposal Generation."""
    proposals: List[TaskProposal] = Field(default_factory=list)
    selected_proposal_id: Optional[str] = None
    exploration_notes: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)


# ============================================================================
# STAGE 3: EXECUTION PLANNING
# ============================================================================

class FileInstruction(BaseModel):
    """Instructions for loading a data file."""
    filename: str
    filepath: str
    columns_to_use: List[str] = Field(default_factory=list)
    filters: List[str] = Field(default_factory=list, description="Filter conditions as strings")
    date_column: Optional[str] = None
    parse_dates: List[str] = Field(default_factory=list)


class JoinStep(BaseModel):
    """A single join operation."""
    left_dataset: str
    right_dataset: str
    left_on: str
    right_on: str
    how: str = "inner"
    validation_rule: Optional[str] = None


class FeatureSpec(BaseModel):
    """Specification for a feature to be engineered."""
    name: str
    description: str
    source_columns: List[str]
    implementation_code: str = Field(..., description="Python code to create feature")
    dtype: Optional[str] = None


class Stage3Plan(BaseModel):
    """Detailed execution plan from Stage 3."""
    plan_id: str = Field(..., description="Plan ID (e.g., PLAN-TSK-001)")
    selected_task_id: str
    goal: str
    task_category: TaskCategory

    # Data loading
    file_instructions: List[FileInstruction] = Field(default_factory=list)

    # Data transformation
    join_steps: List[JoinStep] = Field(default_factory=list)
    feature_engineering: List[FeatureSpec] = Field(default_factory=list)

    # Target and validation
    target_column: str
    date_column: Optional[str] = None
    validation_strategy: str = "temporal"
    train_end_date: Optional[str] = None
    validation_end_date: Optional[str] = None

    # Model expectations
    expected_model_types: List[str] = Field(default_factory=list)
    evaluation_metrics: List[str] = Field(default_factory=list)

    # Output specification
    output_columns: List[str] = Field(default_factory=list)
    artifacts_to_save: List[str] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=datetime.now)


# ============================================================================
# STAGE 3B: DATA PREPARATION
# ============================================================================

class DataQualityReport(BaseModel):
    """Report on data quality after preparation."""
    total_rows: int
    total_columns: int
    null_counts: Dict[str, int] = Field(default_factory=dict)
    duplicate_rows: int = 0
    columns_with_nulls: List[str] = Field(default_factory=list)
    transformations_applied: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class PreparedDataOutput(BaseModel):
    """Output from Stage 3B: Data Preparation."""
    plan_id: str
    prepared_file_path: str
    original_row_count: int
    final_row_count: int
    columns_created: List[str] = Field(default_factory=list)
    columns_dropped: List[str] = Field(default_factory=list)
    data_quality_report: DataQualityReport
    created_at: datetime = Field(default_factory=datetime.now)

    # Validation
    has_no_nulls: bool = True
    ready_for_modeling: bool = True


# ============================================================================
# STAGE 3.5A: METHOD PROPOSAL
# ============================================================================

class ForecastingMethod(BaseModel):
    """A proposed forecasting method."""
    method_id: str = Field(..., description="Unique method ID (e.g., M1)")
    name: str = Field(..., description="Method name")
    category: str = Field(..., description="baseline, statistical, or ml")
    description: str
    implementation_code: str = Field(..., description="Full implementation code")
    required_libraries: List[str] = Field(default_factory=list)
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    expected_strengths: List[str] = Field(default_factory=list)
    expected_weaknesses: List[str] = Field(default_factory=list)


class DataSplitStrategy(BaseModel):
    """Strategy for splitting data for training/validation/testing."""
    strategy_type: str = "temporal"  # temporal or random
    date_column: Optional[str] = None
    target_column: str
    train_period: Optional[str] = None
    validation_period: Optional[str] = None
    test_period: Optional[str] = None
    train_size: Optional[float] = None
    validation_size: Optional[float] = None
    test_size: Optional[float] = None


class MethodProposalOutput(BaseModel):
    """Output from Stage 3.5A: Method Proposal."""
    plan_id: str
    methods_proposed: List[ForecastingMethod] = Field(..., min_length=3, max_length=3)
    data_split_strategy: DataSplitStrategy
    date_column: Optional[str] = None
    target_column: str
    feature_columns: List[str] = Field(default_factory=list)
    data_preprocessing_steps: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)

    @field_validator('methods_proposed')
    @classmethod
    def validate_method_count(cls, v):
        if len(v) != 3:
            raise ValueError("Exactly 3 methods must be proposed")
        return v


# ============================================================================
# STAGE 3.5B: BENCHMARKING
# ============================================================================

class BenchmarkMetrics(BaseModel):
    """Metrics from a single benchmark run."""
    mae: Optional[float] = None
    rmse: Optional[float] = None
    mape: Optional[float] = None
    r2: Optional[float] = None
    execution_time_seconds: Optional[float] = None
    additional_metrics: Dict[str, float] = Field(default_factory=dict)


class MethodBenchmarkResult(BaseModel):
    """Benchmark results for a single method."""
    method_id: str
    method_name: Optional[str] = None  # Make optional to support simplified format
    iterations: List[BenchmarkMetrics] = Field(default_factory=list)
    average_metrics: Optional[BenchmarkMetrics] = None
    coefficient_of_variation: Optional[float] = None
    is_valid: bool = True
    failure_reason: Optional[str] = None
    
    # Allow direct metrics for simplified format
    mae: Optional[float] = None
    rmse: Optional[float] = None
    mape: Optional[float] = None
    status: Optional[str] = None


class TesterOutput(BaseModel):
    """Output from Stage 3.5B: Method Benchmarking."""
    plan_id: str
    methods_tested: List[MethodBenchmarkResult] = Field(default_factory=list)
    selected_method_id: str
    selected_method_name: str
    selection_rationale: str
    method_comparison_summary: Optional[str] = None  # Make optional
    detailed_procedure: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)


# ============================================================================
# STAGE 4: EXECUTION
# ============================================================================

class ExecutionResult(BaseModel):
    """Output from Stage 4: Execution."""
    plan_id: str
    status: ExecutionStatus
    outputs: Dict[str, str] = Field(default_factory=dict, description="Output type to file path mapping")
    metrics: Dict[str, float] = Field(default_factory=dict)
    summary: str
    errors: List[str] = Field(default_factory=list)
    detailed_log: Optional[str] = None
    data_shape: Optional[List[int]] = None
    execution_time_seconds: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.now)


# ============================================================================
# STAGE 5: VISUALIZATION
# ============================================================================

class Visualization(BaseModel):
    """A single visualization."""
    filename: Optional[str] = None  # Optional - can be derived from filepath
    filepath: str
    plot_type: str
    title: str
    description: str
    columns_used: List[str] = Field(default_factory=list)


class VisualizationReport(BaseModel):
    """Output from Stage 5: Visualization."""
    plan_id: str
    visualizations: List[Visualization] = Field(default_factory=list)
    html_report_path: Optional[str] = None
    insights: List[str] = Field(default_factory=list)
    summary: str
    task_answer: Optional[str] = None  # Answer to the original task question
    created_at: datetime = Field(default_factory=datetime.now)


# ============================================================================
# PIPELINE STATE
# ============================================================================

class StageState(BaseModel):
    """State of a single pipeline stage."""
    stage_name: str
    status: StageStatus = StageStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    output: Optional[Any] = None
    errors: List[str] = Field(default_factory=list)


class PipelineState(BaseModel):
    """Overall pipeline state."""
    session_id: str
    selected_task_id: Optional[str] = None
    user_query: Optional[str] = None

    # Stage states
    stages: Dict[str, StageState] = Field(default_factory=dict)

    # Outputs from each stage (for easy access)
    stage1_output: Optional[Stage1Output] = None
    stage2_output: Optional[Stage2Output] = None
    stage3_output: Optional[Stage3Plan] = None
    stage3b_output: Optional[PreparedDataOutput] = None
    stage3_5a_output: Optional[MethodProposalOutput] = None
    stage3_5b_output: Optional[TesterOutput] = None
    stage4_output: Optional[ExecutionResult] = None
    stage5_output: Optional[VisualizationReport] = None

    # Tracking
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    errors: List[str] = Field(default_factory=list)

    def mark_stage_started(self, stage_name: str):
        """Mark a stage as started."""
        if stage_name not in self.stages:
            self.stages[stage_name] = StageState(stage_name=stage_name)
        self.stages[stage_name].status = StageStatus.RUNNING
        self.stages[stage_name].started_at = datetime.now()

    def mark_stage_completed(self, stage_name: str, output: Any = None):
        """Mark a stage as completed."""
        if stage_name not in self.stages:
            self.stages[stage_name] = StageState(stage_name=stage_name)
        self.stages[stage_name].status = StageStatus.COMPLETED
        self.stages[stage_name].completed_at = datetime.now()
        self.stages[stage_name].output = output

    def mark_stage_failed(self, stage_name: str, error: str):
        """Mark a stage as failed."""
        if stage_name not in self.stages:
            self.stages[stage_name] = StageState(stage_name=stage_name)
        self.stages[stage_name].status = StageStatus.FAILED
        self.stages[stage_name].errors.append(error)
        self.errors.append(f"{stage_name}: {error}")


# ============================================================================
# CONVERSATION STATE
# ============================================================================

class ConversationMessage(BaseModel):
    """A single message in the conversation."""
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ConversationContext(BaseModel):
    """Context for the conversational agent."""
    session_id: str
    messages: List[ConversationMessage] = Field(default_factory=list)
    current_intent: Optional[str] = None
    pipeline_state: Optional[PipelineState] = None

    # Quick access to current results
    available_summaries: List[str] = Field(default_factory=list)
    available_proposals: List[str] = Field(default_factory=list)
    current_task_id: Optional[str] = None

    # User preferences
    preferred_task_category: Optional[TaskCategory] = None
    custom_query: Optional[str] = None

    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)

    def add_message(self, role: str, content: str, **metadata):
        """Add a message to the conversation."""
        self.messages.append(ConversationMessage(
            role=role,
            content=content,
            metadata=metadata
        ))
        self.last_updated = datetime.now()


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Enums
    "TaskCategory", "StageStatus", "ExecutionStatus", "LogicalDataType",
    # Stage 1
    "ColumnSummary", "DatasetSummary", "Stage1Output",
    # Stage 2
    "JoinPlan", "FeaturePlan", "ValidationPlan", "TaskProposal", "Stage2Output",
    # Stage 3
    "FileInstruction", "JoinStep", "FeatureSpec", "Stage3Plan",
    # Stage 3B
    "DataQualityReport", "PreparedDataOutput",
    # Stage 3.5A
    "ForecastingMethod", "DataSplitStrategy", "MethodProposalOutput",
    # Stage 3.5B
    "BenchmarkMetrics", "MethodBenchmarkResult", "TesterOutput",
    # Stage 4
    "ExecutionResult",
    # Stage 5
    "Visualization", "VisualizationReport",
    # Pipeline
    "StageState", "PipelineState",
    # Conversation
    "ConversationMessage", "ConversationContext",
]
