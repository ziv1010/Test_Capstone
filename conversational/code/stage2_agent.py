"""
Stage 2 Agent: Task Proposal Generation

This agent explores dataset summaries and proposes analytical tasks,
with a focus on forecasting when possible.
"""

import json
from typing import Dict, Any, Optional, Annotated
from datetime import datetime
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from code.config import (
    SUMMARIES_DIR, STAGE2_OUT_DIR, SECONDARY_LLM_CONFIG,
    STAGE_MAX_ROUNDS, DataPassingManager, logger
)
from code.models import TaskProposal, Stage2Output, PipelineState
from tools.stage2_tools import STAGE2_TOOLS


# ============================================================================
# STATE DEFINITION
# ============================================================================

class Stage2State(BaseModel):
    """State for Stage 2 agent."""
    messages: Annotated[list, add_messages] = []
    summaries_explored: list = []
    proposals: list = []
    exploration_notes: str = ""
    iteration: int = 0
    complete: bool = False


# ============================================================================
# SYSTEM PROMPT
# ============================================================================

STAGE2_SYSTEM_PROMPT = """You are a Task Proposal Agent. Analyze datasets and propose 5 analytical tasks.

## CRITICAL RULES
1. ONLY propose tasks using EXISTING datasets (from list_dataset_summaries)
2. DO NOT invent datasets - use what exists
3. STOP after calling save_task_proposals - do not continue
4. READ dataset summaries carefully - pay attention to column semantics
5. **AT LEAST 2 proposals MUST use MULTIPLE DATASETS (cross-dataset analysis)**

## Workflow (Follow Exactly)
1. Call list_dataset_summaries() to see available data
2. Call read_dataset_summary() for each dataset
3. **ALWAYS call explore_data_relationships() to find join keys** (REQUIRED - not optional)
4. Create 5 task proposals grounded in the data you observed:
   - 3 primary analytical tasks (forecasting, regression, classification)
   - 2 factor-based tasks (using categorical columns as analysis factors)
   - **AT LEAST 2 tasks must combine 2+ datasets via joins**
5. Call save_task_proposals() with your JSON
6. STOP - you are done

## MULTI-DATASET TASKS (CRITICAL)
You MUST create at least 2 proposals that use MORE THAN ONE dataset:
- Look for common columns (Year, Season, Crop, Region) across datasets
- Create tasks that JOIN datasets to enrich the analysis
- Example: "Join production data with export data to predict export potential"
- The join_plan field should specify how datasets are joined

## Understanding Column Values
When reading summaries, pay attention to:
- "value_interpretation": Explains what the column values mean
- "unique_values": All values in categorical columns
- Some categorical columns may have aggregate/summary rows (like "Total" in a Season column)
- When creating tasks, decide intelligently which values to use based on context

## Task Categories (Priority Order)
1. FORECASTING - if datetime/year columns exist
2. REGRESSION - predicting continuous values
3. CLASSIFICATION - predicting categories
4. CLUSTERING - finding groupings

## Factor-Based Tasks (2 Required)
These use categorical columns as analysis dimensions:
- "Predict X by Season" - use Season column as a factor
- "Forecast Y grouped by Region" - stratify analysis by region
- Consider which categorical values are meaningful (individual items vs aggregates)

## Each Proposal Must Include
- id: TSK-001, TSK-002, etc.
- category: forecasting/regression/classification/clustering
- title: Clear name
- problem_statement: What will be predicted, which factors/groupings used
- required_datasets: List of dataset filenames (MUST exist) - USE 2+ DATASETS WHEN POSSIBLE
- target_column: Column to predict (MUST exist in data)
- target_dataset: Dataset containing target
- feature_columns: Include categorical columns used as factors
- join_plan: If using multiple datasets, specify join keys and type
- feasibility_score: 0-1 based on data quality

## Available Tools
- list_dataset_summaries: See what datasets exist
- read_dataset_summary: Get dataset details (includes semantic info)
- explore_data_relationships: Find join possibilities (MUST CALL THIS)
- evaluate_forecasting_feasibility: Check if forecasting works
- save_task_proposals: Save your proposals (CALL THIS TO FINISH)
- get_proposal_template: Get JSON format

IMPORTANT: After save_task_proposals succeeds, STOP. Do not call more tools.
"""



# ============================================================================
# AGENT LOGIC
# ============================================================================

def create_stage2_agent():
    """Create the Stage 2 agent graph."""

    llm = ChatOpenAI(**SECONDARY_LLM_CONFIG)
    llm_with_tools = llm.bind_tools(STAGE2_TOOLS, parallel_tool_calls=False)

    def agent_node(state: Stage2State) -> Dict[str, Any]:
        """Main agent reasoning node."""
        messages = state.messages

        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=STAGE2_SYSTEM_PROMPT)] + list(messages)

        if state.iteration >= STAGE_MAX_ROUNDS.get("stage2", 15):
            return {
                "messages": [AIMessage(content="Maximum iterations reached. Finalizing proposals.")],
                "complete": True
            }

        response = llm_with_tools.invoke(messages)

        return {
            "messages": [response],
            "iteration": state.iteration + 1
        }

    def should_continue(state: Stage2State) -> str:
        """Determine if we should continue or end."""
        if state.complete:
            return "end"

        if state.messages:
            last_message = state.messages[-1]
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "tools"

        return "end"

    builder = StateGraph(Stage2State)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(STAGE2_TOOLS))
    builder.set_entry_point("agent")
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END}
    )
    builder.add_edge("tools", "agent")

    return builder.compile(checkpointer=MemorySaver())


def run_stage2(pipeline_state: PipelineState = None) -> Stage2Output:
    """
    Run Stage 2: Task Proposal Generation.

    Explores dataset summaries and proposes analytical tasks.
    """
    import uuid
    import random

    logger.info("Starting Stage 2: Task Proposal Generation")

    graph = create_stage2_agent()

    # Generate unique session to avoid cached results
    session_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Randomize task emphasis to get different proposals
    task_emphasis_options = [
        "Focus on tasks that combine MULTIPLE datasets through joins - cross-dataset analysis is preferred.",
        "Prioritize tasks that leverage relationships between different datasets.",
        "Look for opportunities to merge datasets and create richer analytical tasks.",
        "Emphasize analytical tasks that require data from 2 or more datasets.",
    ]
    task_emphasis = random.choice(task_emphasis_options)

    initial_message = HumanMessage(content=f"""
Session: {session_id} | Time: {timestamp}

Please analyze the available dataset summaries and propose analytical tasks:

1. First, list all available dataset summaries
2. Read each summary to understand the data - pay attention to column semantics
3. **IMPORTANT**: Call explore_data_relationships() to find join possibilities between datasets
4. For datasets with datetime columns, evaluate forecasting feasibility
5. Propose 5 tasks total:
   - 3 primary tasks (forecasting, regression, classification)
   - 2 factor-based tasks (using categorical columns as grouping/factors)
6. Save your proposals

## CRITICAL REQUIREMENTS:
- **{task_emphasis}**
- At least 2 of your 5 proposals MUST use MORE THAN ONE DATASET (multi-dataset tasks)
- Explore join keys between datasets and create tasks that leverage combined data
- Think creatively about how different datasets can complement each other

When reading summaries:
- Look at value_interpretation to understand what each column means
- For categorical columns, understand which values are individual items vs aggregates
- Identify common columns across datasets that could be used for joining

When complete, save proposals and summarize what you proposed.
""")

    # Use unique thread_id to ensure fresh conversation each run
    config = {"configurable": {"thread_id": f"stage2_{session_id}_{timestamp}"}}
    initial_state = Stage2State(messages=[initial_message])

    try:
        final_state = graph.invoke(initial_state, config)

        # Load proposals from disk
        proposals_path = STAGE2_OUT_DIR / "task_proposals.json"
        if proposals_path.exists():
            data = DataPassingManager.load_artifact(proposals_path)
            proposals_data = data.get('proposals', [])

            proposals = []
            for p in proposals_data:
                try:
                    proposals.append(TaskProposal(**p))
                except Exception as e:
                    logger.warning(f"Could not parse proposal: {e}")

            output = Stage2Output(
                proposals=proposals,
                exploration_notes="Stage 2 completed successfully"
            )
        else:
            output = Stage2Output(proposals=[], exploration_notes="No proposals generated")

        logger.info(f"Stage 2 complete: {len(output.proposals)} proposals generated")
        return output

    except Exception as e:
        logger.error(f"Stage 2 failed: {e}")
        return Stage2Output(proposals=[], exploration_notes=f"Error: {e}")


def run_stage2_for_query(user_query: str) -> Stage2Output:
    """
    Run Stage 2 with a specific user query in mind.

    Tailors proposals to match the user's intent.
    """
    import uuid

    logger.info(f"Running Stage 2 for query: {user_query}")

    graph = create_stage2_agent()

    # Generate unique session to avoid cached results
    session_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    initial_message = HumanMessage(content=f"""
Session: {session_id} | Time: {timestamp}

A user has a specific analytical goal:

"{user_query}"

Please:
1. List and read available dataset summaries
2. **ALWAYS call explore_data_relationships()** to find join possibilities
3. Evaluate if this specific goal is achievable with the data
4. If feasible, create a task proposal for this goal
5. Also propose 2-3 alternative tasks in case the primary goal isn't possible
6. PRIORITIZE FORECASTING tasks when datetime columns exist
7. **Include at least 1 multi-dataset task** that combines data through joins
8. Save all proposals

Focus on making the user's request work if at all possible.
When creating alternative tasks, consider how multiple datasets could be combined.
""")

    # Use unique thread_id to ensure fresh conversation each run
    config = {"configurable": {"thread_id": f"stage2_query_{session_id}_{timestamp}"}}
    initial_state = Stage2State(messages=[initial_message])

    try:
        final_state = graph.invoke(initial_state, config)

        proposals_path = STAGE2_OUT_DIR / "task_proposals.json"
        if proposals_path.exists():
            data = DataPassingManager.load_artifact(proposals_path)
            proposals_data = data.get('proposals', [])

            proposals = []
            for p in proposals_data:
                try:
                    proposals.append(TaskProposal(**p))
                except Exception as e:
                    logger.warning(f"Could not parse proposal: {e}")

            return Stage2Output(
                proposals=proposals,
                exploration_notes=f"Generated proposals for: {user_query}"
            )

        return Stage2Output(proposals=[])

    except Exception as e:
        logger.error(f"Stage 2 query failed: {e}")
        return Stage2Output(proposals=[])


# ============================================================================
# PIPELINE NODE FUNCTION
# ============================================================================

def stage2_node(state: PipelineState) -> PipelineState:
    """
    Stage 2 node for the master pipeline graph.
    """
    state.mark_stage_started("stage2")

    try:
        if state.user_query:
            output = run_stage2_for_query(state.user_query)
        else:
            output = run_stage2(state)

        state.stage2_output = output
        state.mark_stage_completed("stage2", output)
    except Exception as e:
        state.mark_stage_failed("stage2", str(e))

    return state


if __name__ == "__main__":
    output = run_stage2()
    print(f"Generated {len(output.proposals)} proposals")
    for p in output.proposals:
        print(f"  - {p.id}: {p.title} ({p.category})")
