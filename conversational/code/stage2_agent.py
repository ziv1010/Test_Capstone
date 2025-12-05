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
    STAGE_MAX_ROUNDS, RECURSION_LIMIT, DataPassingManager, logger, DEBUG
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
# PROPOSAL SANITIZATION
# ============================================================================

def sanitize_proposal(proposal: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize a proposal dict to fix common LLM output issues.
    
    Fixes:
    1. join_plan structure (type/on -> join_type/datasets/join_keys)
    2. Invalid category values (factor-based -> regression)
    3. Empty join_plan dicts
    """
    p = proposal.copy()
    
    # Fix category - map invalid values to valid ones
    category_mapping = {
        "factor-based": "regression",  # Factor analysis is typically regression
        "factor_based": "regression",
        "trend_analysis": "forecasting",
        "trend-analysis": "forecasting",
        "time_series": "forecasting",
        "time-series": "forecasting",
        "prediction": "regression",
    }
    
    if 'category' in p:
        cat = str(p['category']).lower().strip()
        if cat in category_mapping:
            logger.debug(f"Sanitizing category: '{p['category']}' -> '{category_mapping[cat]}'")
            p['category'] = category_mapping[cat]
    
    # Fix join_plan
    if 'join_plan' in p and p['join_plan']:
        jp = p['join_plan']
        
        # If it's empty dict, remove it
        if jp == {} or jp is None:
            p['join_plan'] = None
            logger.debug(f"Removing empty join_plan")
        elif isinstance(jp, dict):
            # Check if it has the wrong structure
            if 'datasets' not in jp or 'join_keys' not in jp:
                # Try to fix it
                fixed_jp = {}
                
                # Get join type
                fixed_jp['join_type'] = jp.get('type', jp.get('join_type', 'inner'))
                
                # Try to extract datasets and join_keys from various formats
                if 'on' in jp and isinstance(jp['on'], list):
                    # Format: {"type": "inner", "on": ["Year", "Crop"]}
                    # This means joining on same-named columns across datasets
                    datasets = p.get('required_datasets', [])
                    fixed_jp['datasets'] = datasets if datasets else ['dataset1.csv', 'dataset2.csv']
                    # Create join_keys mapping same column to all datasets
                    if jp['on'] and datasets:
                        # Use first join column for all datasets
                        join_col = jp['on'][0]
                        fixed_jp['join_keys'] = {ds: join_col for ds in datasets}
                    else:
                        fixed_jp['join_keys'] = {}
                elif 'left' in jp and 'right' in jp:
                    # Format: {"type": "inner", "left": "col1", "right": "col2"}
                    datasets = p.get('required_datasets', [])
                    fixed_jp['datasets'] = datasets
                    if len(datasets) >= 2:
                        fixed_jp['join_keys'] = {
                            datasets[0]: jp.get('left', ''),
                            datasets[1]: jp.get('right', '')
                        }
                    else:
                        fixed_jp['join_keys'] = {}
                else:
                    # Can't fix, remove join_plan
                    logger.debug(f"Cannot fix join_plan structure: {jp}")
                    p['join_plan'] = None
                    return p
                
                if fixed_jp.get('datasets') and fixed_jp.get('join_keys'):
                    logger.debug(f"Fixed join_plan: {jp} -> {fixed_jp}")
                    p['join_plan'] = fixed_jp
                else:
                    p['join_plan'] = None
    
    return p


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
3. **Call explore_data_relationships() with NO ARGUMENTS** - it will automatically use all datasets
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
- Use "regression" as the category for factor-based tasks

## Each Proposal Must Include
- id: TSK-001, TSK-002, etc.
- category: forecasting/regression/classification/clustering/descriptive (NOT "factor-based")
- title: Clear name
- problem_statement: What will be predicted, which factors/groupings used
- required_datasets: List of dataset filenames (MUST exist) - USE 2+ DATASETS WHEN POSSIBLE
- target_column: Column to predict (MUST exist in data)
- target_dataset: Dataset containing target
- feature_columns: Include categorical columns used as factors
- join_plan: If using multiple datasets, use format: {"datasets": [...], "join_keys": {"dataset1": "col", "dataset2": "col"}, "join_type": "inner"}
- feasibility_score: 0-1 based on data quality

## Available Tools
- list_dataset_summaries: See what datasets exist
- read_dataset_summary: Get dataset details (includes semantic info)
- explore_data_relationships: Find join possibilities - **CALL WITH NO ARGUMENTS**: explore_data_relationships()
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
            logger.warning(f"Stage 2: Maximum iterations ({STAGE_MAX_ROUNDS.get('stage2', 15)}) reached")
            return {
                "messages": [AIMessage(content="Maximum iterations reached. Finalizing proposals.")],
                "complete": True
            }

        if DEBUG:
            logger.debug(f"Stage 2 iteration {state.iteration}: Invoking LLM...")

        response = llm_with_tools.invoke(messages)

        # Debug logging for LLM response and tool calls
        if DEBUG:
            logger.debug(f"Stage 2 Agent Response: {response.content[:500] if response.content else 'No content'}...")
            if hasattr(response, 'tool_calls') and response.tool_calls:
                for tc in response.tool_calls:
                    logger.debug(f"Tool Call: {tc['name']} with args: {str(tc['args'])[:500]}...")

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

    logger.info("=" * 60)
    logger.info("Starting Stage 2: Task Proposal Generation")
    logger.info("=" * 60)

    # Delete old proposals to ensure fresh generation
    proposals_path = STAGE2_OUT_DIR / "task_proposals.json"
    if proposals_path.exists():
        logger.info(f"Removing old proposals file: {proposals_path}")
        proposals_path.unlink()

    graph = create_stage2_agent()

    # Generate unique session to avoid cached results
    session_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


    if DEBUG:
        logger.debug(f"Session ID: {session_id}")
        logger.debug(f"Timestamp: {timestamp}")

    # Randomize task emphasis to get different proposals
    task_emphasis_options = [
        "Focus on tasks that combine MULTIPLE datasets through joins - cross-dataset analysis is preferred.",
        "Prioritize tasks that leverage relationships between different datasets.",
        "Look for opportunities to merge datasets and create richer analytical tasks.",
        "Emphasize analytical tasks that require data from 2 or more datasets.",
    ]
    task_emphasis = random.choice(task_emphasis_options)

    if DEBUG:
        logger.debug(f"Task emphasis: {task_emphasis}")

    initial_message = HumanMessage(content=f"""
Session: {session_id} | Time: {timestamp}

Please analyze the available dataset summaries and propose analytical tasks:

1. First, list all available dataset summaries
2. Read each summary to understand the data - pay attention to column semantics
3. **IMPORTANT**: Call explore_data_relationships() with NO ARGUMENTS - it auto-uses all datasets
4. For datasets with datetime columns, evaluate forecasting feasibility
5. Propose 5 tasks total:
   - 3 primary tasks (forecasting, regression, classification)
   - 2 factor-based tasks (use "regression" category for these)
6. Save your proposals using save_task_proposals tool

## CRITICAL REQUIREMENTS:
- **{task_emphasis}**
- At least 2 of your 5 proposals MUST use MORE THAN ONE DATASET (multi-dataset tasks)
- For multi-dataset tasks, use join_plan format: {{"datasets": [...], "join_keys": {{"ds1": "col", "ds2": "col"}}, "join_type": "inner"}}
- Valid categories are: forecasting, regression, classification, clustering, descriptive (NOT "factor-based")

When reading summaries:
- Look at value_interpretation to understand what each column means
- For categorical columns, understand which values are individual items vs aggregates
- Identify common columns across datasets that could be used for joining

When complete, YOU MUST call save_task_proposals with your proposals JSON and summarize what you proposed.
""")

    # Use unique thread_id to ensure fresh conversation each run
    # Also set recursion_limit to allow enough iterations for the agent to complete
    config = {
        "configurable": {"thread_id": f"stage2_{session_id}_{timestamp}"},
        "recursion_limit": RECURSION_LIMIT
    }
    initial_state = Stage2State(messages=[initial_message])

    if DEBUG:
        logger.debug(f"Initial state: messages={len(initial_state.messages)}, iteration={initial_state.iteration}")

    try:
        logger.info("Invoking Stage 2 graph...")
        final_state = graph.invoke(initial_state, config)

        if DEBUG:
            logger.debug(f"Final state: iteration={final_state.get('iteration', 'N/A')}, complete={final_state.get('complete', 'N/A')}")
            # Log tool calls made during execution
            messages = final_state.get('messages', [])
            tool_calls_made = []
            for msg in messages:
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_calls_made.append(tc.get('name', 'unknown'))
            logger.debug(f"Tools called during execution: {tool_calls_made}")

        # Load proposals from disk
        proposals_path = STAGE2_OUT_DIR / "task_proposals.json"
        if DEBUG:
            logger.debug(f"Checking for proposals at: {proposals_path}")
            logger.debug(f"File exists: {proposals_path.exists()}")

        if proposals_path.exists():
            logger.info(f"✅ Found proposals file at: {proposals_path}")
            data = DataPassingManager.load_artifact(proposals_path)
            proposals_data = data.get('proposals', [])

            if DEBUG:
                logger.debug(f"Loaded {len(proposals_data)} proposals from file")

            proposals = []
            for i, p in enumerate(proposals_data):
                try:
                    # Sanitize proposal to fix common LLM output issues
                    sanitized = sanitize_proposal(p)
                    proposals.append(TaskProposal(**sanitized))
                    if DEBUG:
                        logger.debug(f"  ✅ Proposal {i+1}: {p.get('id', 'N/A')} - {p.get('title', 'N/A')}")
                except Exception as e:
                    logger.warning(f"Could not parse proposal {i+1} (id={p.get('id', 'N/A')}): {e}")

            output = Stage2Output(
                proposals=proposals,
                exploration_notes="Stage 2 completed successfully"
            )
        else:
            logger.warning("❌ No proposals file found - agent did not call save_task_proposals!")
            logger.warning("Check debug logs above to see what the agent did instead.")
            output = Stage2Output(proposals=[], exploration_notes="No proposals generated - save_task_proposals was not called")

        logger.info(f"Stage 2 complete: {len(output.proposals)} proposals generated")
        return output

    except Exception as e:
        import traceback
        logger.error(f"Stage 2 failed: {e}")
        if DEBUG:
            logger.debug(f"Traceback: {traceback.format_exc()}")
        return Stage2Output(proposals=[], exploration_notes=f"Error: {e}")


def run_stage2_for_query(user_query: str) -> Stage2Output:
    """
    Run Stage 2 with a specific user query in mind.

    Tailors proposals to match the user's intent.
    """
    import uuid

    logger.info("=" * 60)
    logger.info(f"Running Stage 2 for query: {user_query}")
    logger.info("=" * 60)

    # Delete old proposals to ensure fresh generation
    proposals_path = STAGE2_OUT_DIR / "task_proposals.json"
    if proposals_path.exists():
        logger.info(f"Removing old proposals file: {proposals_path}")
        proposals_path.unlink()

    graph = create_stage2_agent()

    # Generate unique session to avoid cached results
    session_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if DEBUG:
        logger.debug(f"Session ID: {session_id}, Timestamp: {timestamp}")

    initial_message = HumanMessage(content=f"""
Session: {session_id} | Time: {timestamp}

A user has a specific analytical goal:

"{user_query}"

Please:
1. List and read available dataset summaries
2. **Call explore_data_relationships() with NO ARGUMENTS** - it auto-uses all datasets
3. Evaluate if this specific goal is achievable with the data
4. If feasible, create a task proposal for this goal
5. Also propose 2-3 alternative tasks in case the primary goal isn't possible
6. PRIORITIZE FORECASTING tasks when datetime columns exist
7. **Include at least 1 multi-dataset task** that combines data through joins
8. YOU MUST call save_task_proposals with your proposals JSON

Valid categories are: forecasting, regression, classification, clustering, descriptive (NOT "factor-based")
For join_plan, use format: {{"datasets": [...], "join_keys": {{"ds1": "col", "ds2": "col"}}, "join_type": "inner"}}
""")

    # Use unique thread_id to ensure fresh conversation each run
    # Also set recursion_limit to allow enough iterations for the agent to complete
    config = {
        "configurable": {"thread_id": f"stage2_query_{session_id}_{timestamp}"},
        "recursion_limit": RECURSION_LIMIT
    }
    initial_state = Stage2State(messages=[initial_message])

    if DEBUG:
        logger.debug(f"Initial state: messages={len(initial_state.messages)}, iteration={initial_state.iteration}")

    try:
        logger.info("Invoking Stage 2 graph for query...")
        final_state = graph.invoke(initial_state, config)

        if DEBUG:
            logger.debug(f"Final state: iteration={final_state.get('iteration', 'N/A')}, complete={final_state.get('complete', 'N/A')}")
            messages = final_state.get('messages', [])
            tool_calls_made = []
            for msg in messages:
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_calls_made.append(tc.get('name', 'unknown'))
            logger.debug(f"Tools called during execution: {tool_calls_made}")

        proposals_path = STAGE2_OUT_DIR / "task_proposals.json"
        if DEBUG:
            logger.debug(f"Checking for proposals at: {proposals_path}")
            logger.debug(f"File exists: {proposals_path.exists()}")

        if proposals_path.exists():
            logger.info(f"✅ Found proposals file at: {proposals_path}")
            data = DataPassingManager.load_artifact(proposals_path)
            proposals_data = data.get('proposals', [])

            if DEBUG:
                logger.debug(f"Loaded {len(proposals_data)} proposals from file")

            proposals = []
            for i, p in enumerate(proposals_data):
                try:
                    # Sanitize proposal to fix common LLM output issues
                    sanitized = sanitize_proposal(p)
                    proposals.append(TaskProposal(**sanitized))
                    if DEBUG:
                        logger.debug(f"  ✅ Proposal {i+1}: {p.get('id', 'N/A')} - {p.get('title', 'N/A')}")
                except Exception as e:
                    logger.warning(f"Could not parse proposal {i+1} (id={p.get('id', 'N/A')}): {e}")

            return Stage2Output(
                proposals=proposals,
                exploration_notes=f"Generated proposals for: {user_query}"
            )

        logger.warning("❌ No proposals file found - agent did not call save_task_proposals!")
        return Stage2Output(proposals=[], exploration_notes="No proposals generated - save_task_proposals was not called")

    except Exception as e:
        import traceback
        logger.error(f"Stage 2 query failed: {e}")
        if DEBUG:
            logger.debug(f"Traceback: {traceback.format_exc()}")
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
