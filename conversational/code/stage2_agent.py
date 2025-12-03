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

STAGE2_SYSTEM_PROMPT = """You are a Task Proposal Agent responsible for analyzing dataset summaries and proposing meaningful analytical tasks.

## Your Role
You analyze dataset summaries from Stage 1 to understand what data is available, then propose 3-4 concrete analytical tasks that can be performed on this data.

## Priority Order for Task Types
1. **FORECASTING** (highest priority) - Time series prediction if datetime columns exist
2. **REGRESSION** - Predicting continuous numeric values
3. **CLASSIFICATION** - Predicting categories
4. **CLUSTERING** - Finding natural groupings
5. **DESCRIPTIVE** - Summarizing and exploring patterns

## Your Goals
1. Explore all available dataset summaries
2. Identify relationships between datasets (possible joins)
3. Propose 3-4 diverse, feasible analytical tasks
4. Prioritize forecasting tasks when datetime columns exist
5. Assess feasibility of each proposal

## Available Tools
- list_dataset_summaries: List all Stage 1 summaries
- read_dataset_summary: Read details of a specific summary
- explore_data_relationships: Analyze join possibilities between datasets
- evaluate_forecasting_feasibility: Check if forecasting is viable
- python_sandbox_stage2: Execute Python for custom analysis
- save_task_proposals: Save your final proposals
- get_proposal_template: Get the required JSON structure

## Task Proposal Requirements
Each proposal must include:
- id: Unique ID (TSK-001, TSK-002, etc.)
- category: Task type (forecasting, regression, classification, etc.)
- title: Clear, descriptive title
- problem_statement: Detailed description of what will be predicted/analyzed
- required_datasets: Which datasets are needed
- target_column: The column to predict
- target_dataset: Which dataset contains the target
- feasibility_score: 0-1 score based on data quality and completeness
- For forecasting: forecast_horizon and forecast_granularity

## Workflow
1. List and read all dataset summaries
2. Identify datetime columns and potential targets
3. Explore relationships between datasets
4. Evaluate forecasting feasibility for each potential target
5. Propose 3-4 tasks, prioritizing forecasting
6. Save proposals using the save_task_proposals tool

## Important Guidelines
- ALWAYS prioritize forecasting if datetime columns exist
- Consider data quality when assessing feasibility
- Be specific about which columns from which datasets
- Explain WHY each task is valuable
- Consider if joins are needed and if join keys exist

When done, save your proposals and provide a summary of what you proposed.
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
    logger.info("Starting Stage 2: Task Proposal Generation")

    graph = create_stage2_agent()

    initial_message = HumanMessage(content="""
Please analyze the available dataset summaries and propose analytical tasks:

1. First, list all available dataset summaries
2. Read each summary to understand the data
3. Explore relationships between datasets (possible joins)
4. For datasets with datetime columns, evaluate forecasting feasibility
5. Propose 3-4 diverse tasks, PRIORITIZING FORECASTING
6. Save your proposals

Focus on:
- Forecasting tasks if datetime columns exist
- Clear problem statements
- Realistic feasibility assessments

When complete, save proposals and summarize what you proposed.
""")

    config = {"configurable": {"thread_id": "stage2_main"}}
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
    logger.info(f"Running Stage 2 for query: {user_query}")

    graph = create_stage2_agent()

    initial_message = HumanMessage(content=f"""
A user has a specific analytical goal:

"{user_query}"

Please:
1. List and read available dataset summaries
2. Evaluate if this specific goal is achievable with the data
3. If feasible, create a task proposal for this goal
4. Also propose 2-3 alternative tasks in case the primary goal isn't possible
5. PRIORITIZE FORECASTING tasks when datetime columns exist
6. Save all proposals

Focus on making the user's request work if at all possible.
""")

    config = {"configurable": {"thread_id": f"stage2_query_{hash(user_query)}"}}
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
