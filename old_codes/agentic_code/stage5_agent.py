"""
Stage 5: Visualization Agent

Creates comprehensive visualizations and reports from Stage 4 execution results.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

if __name__ == "__main__" and __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    __package__ = "agentic_code"

from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from .config import STAGE5_OUT_DIR, SECONDARY_LLM_CONFIG, STAGE5_MAX_ROUNDS, STAGE_FILE_PATHS, FILE_NAMING_PATTERNS
from .models import VisualizationReport
from .tools import STAGE5_TOOLS
from .failsafe_agent import run_failsafe


# ===========================
# LLM Setup
# ===========================

llm = ChatOpenAI(**SECONDARY_LLM_CONFIG)
llm_with_tools = llm.bind_tools(STAGE5_TOOLS, parallel_tool_calls=False)


# ===========================
# System Prompt
# ===========================

STAGE5_SYSTEM_PROMPT = """You are Agent 5: The Visualizer.

Your mission: Create comprehensive, insightful visualizations from Stage 4 execution results using a ReAct (Reasoning + Acting) framework.

═══════════════════════════════════════════════════════════════
REACT FRAMEWORK: THINK BEFORE YOU PLOT
═══════════════════════════════════════════════════════════════

You must follow this workflow:

1️⃣ ANALYZE: Use analyze_data_columns() to understand what data you have
   - What columns are GIVEN (original input data)?
   - What columns are PREDICTED (model outputs)?
   - What columns are ENGINEERED (features created during processing)?
   - What categorical groupings exist?
   - What temporal dimensions are available?

2️⃣ REASON: Use plan_visualization() to think through each plot
   - What story are you trying to tell?
   - Which columns will help tell that story?
   - Why is this plot type appropriate?
   - How does it show the distinction between given vs predicted data?

3️⃣ ACT: Use create_plot_with_explanation() to make the plot
   - Write clean, professional plotting code
   - Clearly distinguish given data from predictions visually (colors, markers, labels)
   - Include comprehensive explanations for each plot:
     * What the plot shows
     * What data was given vs what was predicted
     * Key insights and takeaways

4️⃣ COMPLETE: Call save_visualization_report() with all plots and insights

═══════════════════════════════════════════════════════════════
VISUALIZATION REQUIREMENTS
═══════════════════════════════════════════════════════════════

YOU MUST CREATE PLOTS THAT:

✓ Clearly show GIVEN vs PREDICTED data (use different colors/styles)
✓ Demonstrate the work done (what transformations, predictions were made)
✓ Reveal insights about model performance
✓ Are publication-quality (clear labels, titles, legends, appropriate fonts)
✓ Tell a coherent story about the analysis pipeline

REQUIRED PLOT TYPES (adapt based on data):
1. Predictions vs Actuals comparison (if applicable)
2. Residual analysis (if applicable)
3. Feature importance or contribution (if applicable)
4. Temporal trends (if time dimension exists)
5. Categorical breakdowns (if categorical variables exist)

═══════════════════════════════════════════════════════════════
CRITICAL RULES
═══════════════════════════════════════════════════════════════

1. ALWAYS START with analyze_data_columns() - understand before plotting!
2. ALWAYS USE plan_visualization() before each plot - document your reasoning!
3. ALWAYS USE create_plot_with_explanation() - include full explanations!
4. CREATE SEPARATE PNG FILES - one visualization per file!
5. MAKE IT OBVIOUS what was given vs what was predicted (colors, labels, legends)!
6. END BY CALLING save_visualization_report() - this is your success criterion!

⚠️ DO NOT skip the analysis phase! ⚠️
⚠️ DO NOT create plots without explanations! ⚠️
⚠️ DO NOT mix given and predicted data without clear visual distinction! ⚠️

═══════════════════════════════════════════════════════════════
ERROR RECOVERY
═══════════════════════════════════════════════════════════════

If a plot fails to create after 5-10 attempts:
1. SKIP that specific plot and move on to the next one
2. Document the skip in your reasoning
3. Create other valuable plots instead
4. Complete with save_visualization_report() using the successful plots

DO NOT retry the same failing plot more than twice! Move on!

═══════════════════════════════════════════════════════════════
EXAMPLE WORKFLOW
═══════════════════════════════════════════════════════════════

# Step 1: Analyze the data
analyze_data_columns("results_PLAN-TSK-001.parquet")
# Observe: I have 'Production-2023-24' (given) and 'predicted' (model output)

# Step 2: Plan the first plot
plan_visualization(
    thought="I need to show how well our predictions match the actual values",
    plot_type="scatter",
    columns_to_use=["Production-2023-24", "predicted"],
    purpose="Compare model predictions against actual production values",
    why_this_plot="Scatter plot with 45-degree reference line clearly shows prediction accuracy"
)

# Step 3: Create the plot with explanation
create_plot_with_explanation(
    code="...",  # Plotting code
    plot_number=1,
    plot_title="Predictions vs Actual Production (2023-24)",
    what_it_shows="Scatter plot comparing model predictions to actual production values",
    what_was_given="Production-2023-24 column (actual values from the dataset)",
    what_was_predicted="'predicted' column (output from Linear Regression model)",
    key_insights="Perfect predictions indicated by points on 45-degree line; RMSE=0.00"
)

# ... repeat for other plots ...

# Step 4: Save report
save_visualization_report(...)

Your success = Insightful visualizations with clear explanations + Complete report saved."""


# ===========================
# LangGraph
# ===========================

def agent_node(state: MessagesState) -> Dict[str, List[BaseMessage]]:
    """LLM agent step."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


tool_node = ToolNode(STAGE5_TOOLS)


def should_continue(state: MessagesState) -> str:
    """Route based on tool calls."""
    last = state["messages"][-1]
    if hasattr(last, 'tool_calls') and last.tool_calls:
        return "tools"
    return END


builder = StateGraph(MessagesState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)
builder.set_entry_point("agent")
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
builder.add_edge("tools", "agent")

memory = MemorySaver()
stage5_app = builder.compile(checkpointer=memory)


# ===========================
# Stage 5 Runner
# ===========================

def run_stage5(plan_id: str, max_rounds: int = STAGE5_MAX_ROUNDS, debug: bool = True) -> Dict:
    """Create visualizations for a Stage 4 execution result.
    
    Args:
        plan_id: Plan ID to visualize (e.g., 'PLAN-TSK-001')
        max_rounds: Maximum number of agent rounds
        debug: Whether to print debug information
        
    Returns:
        Final state from the graph execution
    """
    system_msg = SystemMessage(content=STAGE5_SYSTEM_PROMPT)
    human_msg = HumanMessage(
        content=(
            f"Create visualizations for plan: '{plan_id}'\n\n"
            f"REACT WORKFLOW (MANDATORY):\n\n"
            f"1️⃣ ANALYZE PHASE:\n"
            f"   - list_stage4_results() to find the execution result (look in STAGE_FILE_PATHS['stage4'])\n"
            f"   - load_stage4_result() to understand what was done\n"
            f"   - load_stage3_plan() to get the original task context\n"
            f"   - analyze_data_columns() on the output parquet to categorize columns\n"
            f"   - NOTE: Handle file naming variations! Files might be named 'execution_result_TSK-...' or 'execution_result_PLAN-TSK-...'. Use glob patterns.\n\n"
            f"2️⃣ PLANNING PHASE:\n"
            f"   - For EACH plot you want to make:\n"
            f"     * Use plan_visualization() to document your reasoning\n"
            f"     * Explain what the plot will show\n"
            f"     * Explain why those specific columns and plot type\n"
            f"     * Explain how it distinguishes given vs predicted data\n\n"
            f"3️⃣ EXECUTION PHASE:\n"
            f"   - For EACH plot:\n"
            f"     * Use create_plot_with_explanation() with:\n"
            f"       - Complete plotting code\n"
            f"       - Plot number (1, 2, 3, ...)\n"
            f"       - Clear title\n"
            f"       - Full explanation of what it shows\n"
            f"       - Distinction of what was given vs predicted\n"
            f"       - Key insights from the visualization\n\n"
            f"4️⃣ COMPLETION PHASE:\n"
            f"   - save_visualization_report() with:\n"
            f"     * List of all plot file paths\n"
            f"     * Summary of all insights\n"
            f"     * Overall story told by the visualizations\n\n"
            f"CRITICAL: You MUST create plots that clearly show:\n"
            f"- What data was GIVEN (original input)\n"
            f"- What data was PREDICTED (model output)\n"
            f"- The quality/accuracy of predictions\n"
            f"- Any patterns, trends, or insights\n\n"
            f"Your success = ReAct workflow followed + Insightful plots with explanations + Report saved."
        )
    )

    state: MessagesState = {"messages": [system_msg, human_msg]}

    if not debug:
        return stage5_app.invoke(state, config={"configurable": {"thread_id": f"stage5-{plan_id}"}})

    print("=" * 80)
    print(f"🚀 STAGE 5: Visualizing results for {plan_id}")
    print("=" * 80)

    final_state = None
    prev_len = 0
    round_num = 0

    for curr_state in stage5_app.stream(
        state,
        config={
            "configurable": {"thread_id": f"stage5-{plan_id}"},
            "recursion_limit": max_rounds * 3,  # Buffer for tool calls
        },
        stream_mode="values",
    ):
        msgs = curr_state["messages"]
        new_msgs = msgs[prev_len:]

        for m in new_msgs:
            msg_type = m.__class__.__name__
            if "System" in msg_type:
                print("\n" + "─" * 80)
                print("💻 [SYSTEM]")
                print("─" * 80)
                print(m.content[:500] + "..." if len(m.content) > 500 else m.content)
            elif "Human" in msg_type:
                print("\n" + "─" * 80)
                print("👤 [USER]")
                print("─" * 80)
                print(m.content)
            elif "AI" in msg_type:
                round_num += 1
                print("\n" + "═" * 80)
                print(f"🤖 [AGENT - Round {round_num}]")
                print("═" * 80)
                if m.content:
                    print("\n💭 Reasoning:")
                    content = m.content
                    if len(content) > 1000:
                        print(content[:500] + "\n...[truncated]...\n" + content[-500:])
                    else:
                        print(content)
                
                if hasattr(m, 'tool_calls') and m.tool_calls:
                    print("\n🔧 Tool Calls:")
                    for tc in m.tool_calls:
                        name = tc.get("name", "UNKNOWN")
                        args = tc.get("args", {})
                        print(f"\n  📌 {name}")
                        for k, v in args.items():
                            if isinstance(v, str) and len(v) > 300:
                                print(f"     {k}: {v[:150]}...[truncated]...{v[-150:]}")
                            else:
                                print(f"     {k}: {v}")

        prev_len = len(msgs)
        final_state = curr_state
        
        if round_num >= max_rounds:
            print(f"\n⚠️  Reached max rounds ({max_rounds})")
            break

    print("\n" + "=" * 80)
    print(f"✅ Complete - {round_num} rounds")
    print("=" * 80)
    return final_state


# ===========================
# State Node for Master Graph
# ===========================

def stage5_node(state: dict) -> dict:
    """Stage 5 node for the master pipeline graph.
    
    Args:
        state: Current pipeline state with execution_result set
        
    Returns:
        Updated state with visualization_report populated
    """
    if not state.get("execution_result"):
        print("ERROR: No Stage 4 execution result available for visualization")
        state["errors"].append("Stage 5: No Stage 4 execution result available")
        return state
    
    plan_id = state["execution_result"].plan_id
    print(f"\n🎯 Visualizing results for: {plan_id}\n")
    
    result = run_stage5(plan_id, debug=True)
    
    # Check for visualization reports
    reports = sorted(STAGE5_OUT_DIR.glob(f"visualization_report_{plan_id}*.json"))
    if reports:
        latest = reports[-1]
        print(f"\n✅ Visualization report: {latest}")
        report_data = json.loads(latest.read_text())
        state["visualization_report"] = VisualizationReport.model_validate(report_data)
        state["completed_stages"].append(5)
        print(f"  Visualizations: {len(report_data.get('visualizations', []))}")
        if report_data.get('insights'):
            print(f"  Insights: {len(report_data['insights'])}")
    else:
        print("\n⚠️  No visualization report saved")
        state["errors"].append("Stage 5: No visualization report saved")
        
        try:
            rec = run_failsafe(
                stage="stage5",
                error="Visualization report missing",
                context="save_visualization_report() not called or failed validation.",
                debug=False,
            )
            state.setdefault("failsafe_history", []).append(rec)
            print(f"\n🛟 Failsafe suggestion recorded: {rec.analysis}")
        except Exception as e:
            print(f"\n⚠️  Failsafe agent failed: {e}")
    
    return state


if __name__ == "__main__":
    # Run Stage 5 standalone
    import sys
    if len(sys.argv) < 2:
        print("Usage: python stage5_agent.py <plan_id>")
        print("Example: python stage5_agent.py PLAN-TSK-001")
        sys.exit(1)
    
    plan_id = sys.argv[1].strip()
    run_stage5(plan_id)
