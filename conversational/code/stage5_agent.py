"""
Stage 5 Agent: Visualization

This agent creates visualizations and generates insights from the results.
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
    STAGE3_OUT_DIR, STAGE4_OUT_DIR, STAGE5_OUT_DIR, STAGE5_WORKSPACE,
    SECONDARY_LLM_CONFIG, STAGE_MAX_ROUNDS, DataPassingManager, logger
)
from code.models import VisualizationReport, PipelineState
from tools.stage5_tools import STAGE5_TOOLS


# ============================================================================
# STATE DEFINITION
# ============================================================================

class Stage5State(BaseModel):
    """State for Stage 5 agent."""
    messages: Annotated[list, add_messages] = []
    plan_id: str = ""
    data_loaded: bool = False
    plots_created: list = []
    insights_generated: list = []
    iteration: int = 0
    complete: bool = False


# ============================================================================
# SYSTEM PROMPT
# ============================================================================

STAGE5_SYSTEM_PROMPT = """You are a Visualization Agent responsible for creating informative visualizations.

## Your Role
1. Load execution results from Stage 4
2. Analyze the data structure
3. Create publication-quality visualizations
4. Generate insights from the analysis
5. Save a comprehensive visualization report

## Your Goals
- Create clear, informative visualizations
- Distinguish between ACTUAL (given) and PREDICTED (model output)
- Tell a coherent story about the analysis
- Generate actionable insights

## Available Tools
- load_execution_results: Load Stage 4 results
- analyze_data_columns: Understand what columns are available
- plan_visualization: Plan a plot before creating it
- create_plot: Create and save a custom plot
- create_standard_plots: Create a standard set of plots
- generate_insights: Extract insights from results
- save_visualization_report: Save final report

## Required Visualizations
Create at least these plots:
1. **Actual vs Predicted Scatter** - How well do predictions match?
2. **Time Series Comparison** - Actual and predicted over time
3. **Residual Distribution** - Are errors normally distributed?
4. **Residuals Over Time** - Any patterns in errors?

Optional (if relevant):
5. Feature importance or correlation
6. Predictions by category (if categorical variables)
7. Error analysis by time period

## Visualization Guidelines
- Use clear titles and labels
- Include legends
- Use distinct colors for actual vs predicted
- Add reference lines where helpful (e.g., perfect prediction line)
- Keep it professional and publication-quality

## Creating Custom Plots
Use create_plot with matplotlib code:
```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(10, 6))
# ... your plotting code ...
plt.title('Your Title')
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.tight_layout()
# Figure is automatically saved
```

## Insights Generation
After creating visualizations, generate insights about:
- Model performance (how good are the predictions?)
- Error patterns (any systematic biases?)
- Recommendations (what could be improved?)

## Workflow
1. Load execution results
2. Analyze data columns (understand what's available)
3. Create standard plots first
4. Generate insights
5. Create any additional custom plots if needed
6. Save visualization report

## Report Requirements
The report must include:
- plan_id
- visualizations: List of created plots
- insights: Key findings and observations
- summary: Overall assessment of results

IMPORTANT: Create informative visualizations that tell the story of this analysis.
"""


# ============================================================================
# AGENT LOGIC
# ============================================================================

def create_stage5_agent():
    """Create the Stage 5 agent graph."""

    llm = ChatOpenAI(**SECONDARY_LLM_CONFIG)
    llm_with_tools = llm.bind_tools(STAGE5_TOOLS, parallel_tool_calls=False)

    def agent_node(state: Stage5State) -> Dict[str, Any]:
        """Main agent reasoning node."""
        messages = state.messages

        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=STAGE5_SYSTEM_PROMPT)] + list(messages)

        if state.iteration >= STAGE_MAX_ROUNDS.get("stage5", 60):
            return {
                "messages": [AIMessage(content="Maximum iterations reached. Finalizing report.")],
                "complete": True
            }

        response = llm_with_tools.invoke(messages)

        return {
            "messages": [response],
            "iteration": state.iteration + 1
        }

    def should_continue(state: Stage5State) -> str:
        """Determine if we should continue or end."""
        if state.complete:
            return "end"

        if state.messages:
            last_message = state.messages[-1]
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "tools"

        return "end"

    builder = StateGraph(Stage5State)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(STAGE5_TOOLS))
    builder.set_entry_point("agent")
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END}
    )
    builder.add_edge("tools", "agent")

    return builder.compile(checkpointer=MemorySaver())


def run_stage5(plan_id: str, pipeline_state: PipelineState = None) -> VisualizationReport:
    """
    Run Stage 5: Visualization.

    Creates visualizations and generates insights.
    """
    logger.info(f"Starting Stage 5: Visualization for {plan_id}")

    graph = create_stage5_agent()

    initial_message = HumanMessage(content=f"""
Create visualizations for plan: {plan_id}

Steps:
1. Load execution results from Stage 4
2. Analyze data columns to understand what's available
3. Create standard plots (actual vs predicted, time series, residuals)
4. Generate insights from the results
5. Create any additional helpful visualizations
6. Save the visualization report

Required plots:
- Actual vs Predicted scatter plot
- Time series: actual and predicted over time
- Residual histogram
- Residuals over time

The results data is at: {STAGE4_OUT_DIR}/results_{plan_id}.parquet

Save plots to: {STAGE5_OUT_DIR}/
Save report as: visualization_report_{plan_id}.json
""")

    config = {"configurable": {"thread_id": f"stage5_{plan_id}"}}
    initial_state = Stage5State(messages=[initial_message], plan_id=plan_id)

    try:
        final_state = graph.invoke(initial_state, config)

        # Load report from disk
        report_path = STAGE5_OUT_DIR / f"visualization_report_{plan_id}.json"
        if report_path.exists():
            data = DataPassingManager.load_artifact(report_path)
            output = VisualizationReport(**data)
            logger.info(f"Stage 5 complete: {len(output.visualizations)} visualizations created")
            return output
        else:
            # Create minimal report
            plots = list(STAGE5_OUT_DIR.glob(f"{plan_id}_*.png"))
            output = VisualizationReport(
                plan_id=plan_id,
                visualizations=[],
                insights=[],
                summary="Visualization completed"
            )
            return output

    except Exception as e:
        logger.error(f"Stage 5 failed: {e}")
        return VisualizationReport(
            plan_id=plan_id,
            visualizations=[],
            summary=f"Visualization failed: {e}"
        )


# ============================================================================
# PIPELINE NODE FUNCTION
# ============================================================================

def stage5_node(state: PipelineState) -> PipelineState:
    """
    Stage 5 node for the master pipeline graph.
    """
    state.mark_stage_started("stage5")

    plan_id = f"PLAN-{state.selected_task_id}" if state.selected_task_id else None
    if not plan_id:
        state.mark_stage_failed("stage5", "No plan ID available")
        return state

    try:
        output = run_stage5(plan_id, state)
        state.stage5_output = output
        state.mark_stage_completed("stage5", output)
    except Exception as e:
        state.mark_stage_failed("stage5", str(e))

    return state


if __name__ == "__main__":
    import sys
    plan_id = sys.argv[1] if len(sys.argv) > 1 else "PLAN-TSK-001"
    output = run_stage5(plan_id)
    print(f"Created {len(output.visualizations)} visualizations")
    print(f"Summary: {output.summary}")
