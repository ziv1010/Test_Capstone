"""
Master Agent: Unified Agentic AI Pipeline

Orchestrates all 5 stages using a single LangGraph workflow:
- Stage 1: Dataset Summarization
- Stage 2: Task Proposal Generation
- Stage 3: Execution Planning
- Stage 4: Execution
- Stage 5: Visualization

Each stage is a state node that processes data and updates the shared pipeline state.
"""

from __future__ import annotations

from typing import TypedDict, List, Optional, Tuple
from datetime import datetime
import json

from langgraph.graph import StateGraph, END

from .config import (
    SUMMARIES_DIR,
    STAGE2_OUT_DIR,
    STAGE3_OUT_DIR,
    STAGE3B_OUT_DIR,
    STAGE3_5_OUT_DIR,
    STAGE4_OUT_DIR,
    STAGE5_OUT_DIR,
)
from .models import (
    DatasetSummary,
    TaskProposal,
    Stage2Output,
    Stage3Plan,
    PreparedDataOutput,
    TesterOutput,
    ExecutionResult,
    VisualizationReport,
    FailsafeRecommendation,
)
from .stage1_agent import stage1_node
from .stage2_agent import stage2_node
from .stage3_agent import stage3_node
from .stage3b_agent import stage3b_node
from .stage3_5_agent import stage3_5_node
from .stage4_agent import stage4_node
from .stage5_agent import stage5_node


# ===========================
# Unified Pipeline State
# ===========================

class PipelineState(TypedDict):
    """Unified state for the entire agentic pipeline."""
    # Current progress
    current_stage: int  # Current stage (1-5)
    completed_stages: List[float]  # List of completed stages (may include 3.2, 3.5)
    
    # Stage outputs
    dataset_summaries: List[DatasetSummary]  # From Stage 1
    task_proposals: List[TaskProposal]  # From Stage 2
    selected_task_id: Optional[str]  # Which task to execute
    stage3_plan: Optional[Stage3Plan]  # From Stage 3
    prepared_data: Optional[PreparedDataOutput]  # From Stage 3B
    tester_output: Optional[TesterOutput]  # From Stage 3.5
    execution_result: Optional[ExecutionResult]  # From Stage 4
    visualization_report: Optional[VisualizationReport]  # From Stage 5
    failsafe_history: List[FailsafeRecommendation]  # Failsafe recommendations
    user_query: Optional[str]  # Original user intent to guide proposals
    
    # Tracking
    errors: List[str]  # Track any errors
    started_at: str  # When the pipeline started


# ===========================
# Build Master Graph
# ===========================

def load_cached_state(end_stage: float, selected_task_id: Optional[str] = None) -> Tuple[PipelineState, float]:
    """Preload artifacts from disk so we can skip completed stages."""
    state: PipelineState = {
        "current_stage": 1,
        "completed_stages": [],
        "dataset_summaries": [],
        "task_proposals": [],
        "selected_task_id": selected_task_id,
        "stage3_plan": None,
        "prepared_data": None,
        "tester_output": None,
        "execution_result": None,
        "visualization_report": None,
        "failsafe_history": [],
        "user_query": None,
        "errors": [],
        "started_at": datetime.now().isoformat(),
    }

    completed: List[float] = []

    # Stage 1 cache
    summary_files = sorted(SUMMARIES_DIR.glob("*.summary.json"))
    if summary_files and end_stage >= 1:
        summaries = []
        for path in summary_files:
            try:
                data = json.loads(path.read_text())
                summaries.append(DatasetSummary.model_validate(data))
            except Exception:
                continue
        if summaries:
            state["dataset_summaries"] = summaries
            completed.append(1)

    # Stage 2 cache
    proposals_path = STAGE2_OUT_DIR / "task_proposals.json"
    if proposals_path.exists() and end_stage >= 2:
        try:
            data = json.loads(proposals_path.read_text())
            stage2 = Stage2Output.model_validate(data)
            state["task_proposals"] = stage2.proposals
            completed.append(2)
            if not state["selected_task_id"] and stage2.proposals:
                state["selected_task_id"] = stage2.proposals[0].id
        except Exception:
            pass

    plan_id = f"PLAN-{state['selected_task_id']}" if state.get("selected_task_id") else None

    # Stage 3 cache
    if end_stage >= 3 and plan_id:
        plan_path = STAGE3_OUT_DIR / f"{plan_id}.json"
        if not plan_path.exists():
            matches = sorted(STAGE3_OUT_DIR.glob(f"*{plan_id}*.json"))
            if matches:
                plan_path = matches[-1]
        if plan_path.exists():
            try:
                plan_data = json.loads(plan_path.read_text())
                state["stage3_plan"] = Stage3Plan.model_validate(plan_data)
                completed.append(3)
            except Exception:
                pass

    # Stage 3B cache (data preparation)
    if end_stage >= 3.2 and plan_id:
        prep_files = sorted(STAGE3B_OUT_DIR.glob(f"prep_{plan_id}*.json"))
        if prep_files:
            try:
                prep_data = json.loads(prep_files[-1].read_text())
                state["prepared_data"] = PreparedDataOutput.model_validate(prep_data)
                completed.append(3.2)
            except Exception:
                pass

    # Stage 3.5 cache
    if end_stage >= 3.5 and plan_id:
        tester_files = sorted(STAGE3_5_OUT_DIR.glob(f"tester_{plan_id}*.json"))
        if tester_files:
            try:
                tester_data = json.loads(tester_files[-1].read_text())
                state["tester_output"] = TesterOutput.model_validate(tester_data)
                completed.append(3.5)
            except Exception:
                pass

    # Stage 4 cache
    if end_stage >= 4 and plan_id:
        exec_files = sorted(STAGE4_OUT_DIR.glob(f"execution_{plan_id}*.json"))
        if exec_files:
            try:
                exec_data = json.loads(exec_files[-1].read_text())
                state["execution_result"] = ExecutionResult.model_validate(exec_data)
                completed.append(4)
            except Exception:
                pass

    # Stage 5 cache
    if end_stage >= 5 and plan_id:
        viz_files = sorted(STAGE5_OUT_DIR.glob(f"visualization_report_{plan_id}*.json"))
        if viz_files:
            try:
                viz_data = json.loads(viz_files[-1].read_text())
                state["visualization_report"] = VisualizationReport.model_validate(viz_data)
                completed.append(5)
            except Exception:
                pass

    completed_sorted = sorted(set(completed))
    if completed_sorted:
        last_done = completed_sorted[-1]
        # Advance to the next unfinished stage; do not force reruns of completed stages
        start_stage = last_done + 1
    else:
        start_stage = 1

    state["current_stage"] = int(start_stage if start_stage <= end_stage else end_stage)
    state["completed_stages"] = completed_sorted
    return state, start_stage


def build_master_graph():
    """Build the master pipeline graph with all stages including Stage 3B and 3.5.
    
    Returns:
        Compiled LangGraph application
    """
    builder = StateGraph(PipelineState)

    # Add all stage nodes
    builder.add_node("stage1", stage1_node)
    builder.add_node("stage2", stage2_node)  
    builder.add_node("stage3", stage3_node)
    builder.add_node("stage3b", stage3b_node)  # Data preparation
    builder.add_node("stage3_5", stage3_5_node)  # Method testing & benchmarking
    builder.add_node("stage4", stage4_node)
    builder.add_node("stage5", stage5_node)

    # Set up linear progression through stages
    builder.set_entry_point("stage1")
    builder.add_edge("stage1", "stage2")
    builder.add_edge("stage2", "stage3")  
    builder.add_edge("stage3", "stage3b")  # Prepare data after planning
    builder.add_edge("stage3b", "stage3_5")  # Test methods on prepared data
    builder.add_edge("stage3_5", "stage4")
    builder.add_edge("stage4", "stage5")
    builder.add_edge("stage5", END)

    # Compile
    master_app = builder.compile()
    
    return master_app


# ===========================
# Pipeline Runner
# ===========================

def run_full_pipeline(selected_task_id: Optional[str] = None) -> PipelineState:
    """Run the complete 5-stage pipeline.
    
    Args:
        selected_task_id: Task ID to execute (e.g., 'TSK-001').
                         If None, will use the first task from Stage 2.
        
    Returns:
        Final pipeline state with all results
    """
    print("\n" + "=" * 80)
    print("🚀 UNIFIED AGENTIC AI PIPELINE")
    print("=" * 80)
    print(f"Started at: {datetime.now().isoformat()}")
    print("=" * 80)
    
    final_state = run_up_to_stage(5, selected_task_id)
    
    # Print summary
    print("\n" + "=" * 80)
    print("🎉 PIPELINE COMPLETE!")
    print("=" * 80)
    print(f"Completed stages: {final_state['completed_stages']}")
    print(f"Dataset summaries: {len(final_state.get('dataset_summaries', []))}")
    print(f"Task proposals: {len(final_state.get('task_proposals', []))}")
    if final_state.get('selected_task_id'):
        print(f"Executed task: {final_state['selected_task_id']}")
    if final_state.get('prepared_data'):
        prep = final_state['prepared_data']
        print(f"Data preparation: {prep.prepared_row_count} rows prepared")
        print(f"                  {len(prep.columns_created)} features created")
    if final_state.get('tester_output'):
        tester = final_state['tester_output']
        print(f"Method testing: {len(tester.methods_proposed)} methods benchmarked")
        print(f"Selected method: {tester.selected_method.name}")
    if final_state.get('execution_result'):
        print(f"Execution status: {final_state['execution_result'].status}")
    if final_state.get('visualization_report'):
        print(f"Visualizations: {len(final_state['visualization_report'].visualizations)}")
    if final_state.get('errors'):
        print(f"\n⚠️  Errors encountered:")
        for error in final_state['errors']:
            print(f"  - {error}")
    print("=" * 80)
    
    return final_state


def run_partial_pipeline(
    start_stage: int = 1,
    end_stage: int = 5,
    selected_task_id: Optional[str] = None,
    user_query: Optional[str] = None,
) -> PipelineState:
    """Run a subset of the pipeline stages.
    
    Args:
        start_stage: Stage to start from (1-5)
        end_stage: Stage to end at (1-5)
        selected_task_id: Task ID to execute (required for stages 3+)
        user_query: Optional user request to guide proposal generation (Stage 2)
        
    Returns:
        Final pipeline state
        
    Raises:
        ValueError: If stage range is invalid
    """
    if not (1 <= start_stage <= end_stage <= 5):
        raise ValueError(f"Invalid stage range: {start_stage}-{end_stage}")
    
    print(f"\n🎯 Running pipeline stages {start_stage}-{end_stage}")

    # Respect end_stage; cached artifacts will be reused automatically unless
    # a single-stage run is explicitly requested (start_stage == end_stage),
    # in which case we rerun that stage even if cached.
    start_override = start_stage if start_stage == end_stage else None
    return run_up_to_stage(
        end_stage,
        selected_task_id,
        start_stage_override=start_override,
        user_query=user_query,
    )


def run_up_to_stage(
    end_stage: int,
    selected_task_id: Optional[str] = None,
    start_stage_override: Optional[int] = None,
    user_query: Optional[str] = None,
) -> PipelineState:
    """Run pipeline up to specified stage.
    
    Args:
        end_stage: Stage to end at (1-5)
        selected_task_id: Task ID to execute (required for stages 3+)
        
    Returns:
        Pipeline state after reaching end_stage
    """
    # Load cached artifacts to skip completed stages
    initial_state, cached_start_stage = load_cached_state(end_stage, selected_task_id)

    # Persist user query in state so downstream stages (e.g., Stage 2) can tailor proposals
    if user_query:
        initial_state["user_query"] = user_query

    # Allow caller to force a specific start stage (used for single-stage reruns).
    start_stage = start_stage_override if start_stage_override is not None else cached_start_stage

    if start_stage > end_stage:
        print(f"\nℹ️  Cached artifacts found up to stage {end_stage}. Skipping execution.")
        return initial_state

    builder = StateGraph(PipelineState)

    # Create stage name to node function mapping
    stage_nodes = {
        "stage1": stage1_node,
        "stage2": stage2_node,
        "stage3": stage3_node,
        "stage3b": stage3b_node,
        "stage3_5": stage3_5_node,
        "stage4": stage4_node,
        "stage5": stage5_node,
    }

    stage_order: List[Tuple[float, str]] = [
        (1, "stage1"),
        (2, "stage2"),
        (3, "stage3"),
        (3.2, "stage3b"),  # Data preparation
        (3.5, "stage3_5"),  # Method testing
        (4, "stage4"),
        (5, "stage5"),
    ]

    included = [(num, name) for num, name in stage_order if start_stage <= num <= end_stage]
    if not included:
        return initial_state

    for _, name in included:
        builder.add_node(name, stage_nodes[name])

    entry_name = included[0][1]
    builder.set_entry_point(entry_name)

    for idx in range(len(included) - 1):
        builder.add_edge(included[idx][1], included[idx + 1][1])

    builder.add_edge(included[-1][1], END)

    partial_app = builder.compile()
    return partial_app.invoke(initial_state)


# ===========================
# Export Master App
# ===========================

# Pre-built master application
master_app = build_master_graph()


if __name__ == "__main__":
    # Run the full pipeline
    import sys
    
    # Check if task ID provided
    task_id = sys.argv[1] if len(sys.argv) > 1 else None
    
    if task_id:
        print(f"Running pipeline with task ID: {task_id}")
    else:
        print("Running pipeline (will auto-select first task from Stage 2)")
    
    run_full_pipeline(selected_task_id=task_id)
