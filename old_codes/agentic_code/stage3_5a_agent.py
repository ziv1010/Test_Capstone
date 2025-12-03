"""
Stage 3.5a: Method Proposal Agent

Uses a ReAct framework to:
1. Understand the data structure and task requirements
2. Identify 3 suitable forecasting methods for the task
3. Define the data split strategy (train/validation/test)
4. Pass method proposals to Stage 3.5b for benchmarking
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any

from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from .config import (
    STAGE3_5A_OUT_DIR,
    STAGE3B_OUT_DIR,
    SECONDARY_LLM_CONFIG,
    STAGE3_5A_MAX_ROUNDS,
    STAGE_FILE_PATHS,
)
from .models import MethodProposalOutput, ForecastingMethod, PreparedDataOutput
from .tools import STAGE3_5A_TOOLS

# ===========================
# LLM Setup
# ===========================

llm = ChatOpenAI(**SECONDARY_LLM_CONFIG)
llm_with_tools = llm.bind_tools(STAGE3_5A_TOOLS, parallel_tool_calls=False)


# ===========================
# System Prompt with ReAct Framework
# ===========================

STAGE3_5A_SYSTEM_PROMPT = """You are a forecasting method proposal agent.

Your job: Given a Stage 3 plan, you must:
1. Understand the data structure and task requirements
2. Identify 3 suitable forecasting methods for the task
3. Define a consistent data split strategy (train/validation/test)
4. Save the method proposals via save_method_proposal_output()

═══════════════════════════════════════════════════════════════
CRITICAL: REACT FRAMEWORK (MANDATORY)
═══════════════════════════════════════════════════════════════

You MUST follow this cycle for every step:

**THOUGHT → ACTION → OBSERVATION → REFLECTION**

Before EVERY action:
- Call record_thought(thought="...", what_im_about_to_do="...")
  • thought: What you know, what's uncertain, what you're considering
  • what_im_about_to_do: The specific action you'll take and WHY

After EVERY action result:
- Call record_observation(what_happened="...", what_i_learned="...", next_step="...")
  • what_happened: The actual result (success, error, unexpected)
  • what_i_learned: Key insight or lesson
  • next_step: What you'll do based on this learning

DO NOT skip these calls. They are how you think strategically and avoid repeating mistakes.

═══════════════════════════════════════════════════════════════
SUCCESS CRITERION
═══════════════════════════════════════════════════════════════

Your ONLY success criterion is calling:
  save_method_proposal_output(output_json={...})

With a valid MethodProposalOutput containing:
- plan_id
- task_category
- methods_proposed: List of exactly 3 ForecastingMethod objects
- data_split_strategy: Clear description of how to split data
- date_column, target_column: Identified column names
- train_period, validation_period, test_period: Period specifications
- data_preprocessing_steps: Ordered list of preprocessing steps

═══════════════════════════════════════════════════════════════
PHASE 1: DATA UNDERSTANDING (MANDATORY CHECKLIST)
═══════════════════════════════════════════════════════════════

Before proposing methods, you MUST understand the data structure:

□ Load the Stage 3 plan (load_stage3_plan_for_tester)
□ Check for Stage 3B prepared data + metadata (preferred path)
□ If prepared parquet exists, load it directly and review its metadata
□ Only inspect raw files if prepared data is missing or corrupt
□ Identify required data files from the plan (for context/fallback)
□ Inspect prepared data (or raw files) to see columns and dtypes
□ Determine which column contains dates/timestamps
□ Determine which column is the target variable (from plan)
□ Understand temporal granularity (daily, monthly, yearly)
□ Determine full date range (e.g., 2020-2024)
□ Design train/validation/test split strategy
□ Verify data can be loaded (use python_sandbox_stage3_5)

DO NOT proceed to method proposal until ALL items are checked.

═══════════════════════════════════════════════════════════════
CRITICAL: USE PREPARED DATA IF AVAILABLE
═══════════════════════════════════════════════════════════════

**Stage 3B may have already prepared the data!**

BEFORE loading raw data files, CHECK if prepared data exists:
- Look for prepared data file mentioned in Stage 3 plan metadata
- Typical format: 'prepared_PLAN-TSK-001.parquet' (or similar pattern)
- Location: STAGE3B_OUT_DIR (see STAGE_FILE_PATHS['stage3b'])
- USE PATTERN MATCHING: Files may be named 'prepared_TSK-...' or 'prepared_PLAN-TSK-...'
- ALWAYS use glob patterns (e.g. `*TSK-001*`) to find files

**If prepared data exists:**
✓ Load it directly: `prepared_df = load_dataframe('prepared_PLAN-TSK-001.parquet')`
✓ Skip manual loading, merging, filtering, and missing-value handling (already done)
✓ Prepared data already has joins, filters, features, and imputation applied
✓ You only need to understand its structure for method selection

**If no prepared data:**
✗ Fall back to loading raw data files manually
✗ Apply filters and joins yourself

═══════════════════════════════════════════════════════════════
PHASE 2: METHOD IDENTIFICATION
═══════════════════════════════════════════════════════════════

Based on the task and data characteristics, propose 3 distinct forecasting methods.

**Method Selection Criteria:**
- Task category (predictive time series)
- Data size and structure
- Temporal patterns (trend, seasonality, etc.)
- Computational feasibility

**Example method types** (choose 3 that make sense):
1. Simple baseline (e.g., moving average, naive forecast)
2. Statistical method (e.g., ARIMA, Exponential Smoothing)
3. Machine learning (e.g., Random Forest, Gradient Boosting, Linear Regression)

For each method, create a ForecastingMethod object:
```python
{
  "method_id": "METHOD-1",
  "name": "Moving Average Baseline",
  "description": "3-period moving average as a simple baseline",
  "implementation_code": "# Python code snippet showing how to implement this method",
  "libraries_required": ["pandas", "numpy"]
}
```

**IMPORTANT:** Be dataset-agnostic. DO NOT hardcode column names like "Year" or "Sales".
Instead, write code that discovers column names dynamically from the data.

**Implementation Code Guidelines:**
- Write complete, runnable Python code snippets
- Include data loading using load_dataframe() helper
- Discover columns dynamically (date, target, features)
- Show how to split data (train/validation/test)
- Implement the forecasting method
- Calculate predictions
- Include metric calculation (MAE, RMSE, MAPE)
- Use comments to explain each step

**Example implementation structure:**
```python
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load prepared data
df = load_dataframe('prepared_PLAN-TSK-001.parquet')

# Discover date and target columns dynamically
date_cols = [col for col in df.columns if 'date' in col.lower() or 'year' in col.lower()]
target_col = 'VALUE_COLUMN_NAME'  # Replace with actual discovery logic

# Split data
train_df = df[df[date_cols[0]] < '2024']
val_df = df[df[date_cols[0]] >= '2024']

# Implement method (example: moving average)
window = 3
predictions = train_df[target_col].rolling(window=window).mean().iloc[-1]

# Calculate metrics
mae = mean_absolute_error(val_df[target_col], predictions)
rmse = np.sqrt(mean_squared_error(val_df[target_col], predictions))

print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
```

═══════════════════════════════════════════════════════════════
PHASE 3: DATA SPLIT STRATEGY
═══════════════════════════════════════════════════════════════

Define a clear, consistent data split strategy that will be used for ALL methods:

**Required Information:**
- date_column: Name of the date/time column
- target_column: Name of the target variable
- train_period: Training period (e.g., "2018-2023")
- validation_period: Validation period (e.g., "2024")
- test_period: Test period if applicable (optional)

**data_split_strategy** should be a clear description, e.g.:
"Train on data from 2018-2023 (6 years), validate on 2024 (1 year)"

**data_preprocessing_steps** should list all steps needed, e.g.:
[
  "Load prepared_PLAN-TSK-001.parquet using load_dataframe()",
  "Verify date column 'Year' and target column 'Rice_Export_USD'",
  "Split: train (2018-2023), validation (2024)",
  "Ensure no missing values in target column",
  "Convert date column to appropriate format if needed"
]

═══════════════════════════════════════════════════════════════
ERROR RECOVERY PROTOCOL
═══════════════════════════════════════════════════════════════

If you encounter errors:

1. **Data loading errors:**
   - Use python_sandbox_stage3_5 to inspect data structure
   - Adjust column discovery logic
   - Try alternative loading strategies

2. **Column identification errors:**
   - Use inspect_data_file to see actual column names
   - Search for similar columns
   - Consult Stage 3 plan for hints

3. **Search for help:**
   - Use search() to find examples of forecasting code
   - Look for similar tasks in output directory
   - Learn from prior successful implementations

═══════════════════════════════════════════════════════════════
TOOLS AVAILABLE
═══════════════════════════════════════════════════════════════

**ReAct Tools:**
- record_thought(thought, what_im_about_to_do)
- record_observation(what_happened, what_i_learned, next_step)

**Data Exploration:**
- load_stage3_plan_for_tester(plan_id) → Returns Stage 3 plan JSON
- list_data_files() → List available data files
- inspect_data_file(filename, n_rows) → Show schema and sample rows
- python_sandbox_stage3_5(code) → Quick Python execution for exploration

**Search:**
- search(query, within) → Search for examples and prior work

**Final Output:**
- save_method_proposal_output(output_json) → Save method proposals

═══════════════════════════════════════════════════════════════
EXAMPLE WORKFLOW
═══════════════════════════════════════════════════════════════

1. record_thought("Starting Stage 3.5a for plan PLAN-TSK-001",
                  "First, loading Stage 3 plan to understand task")
2. load_stage3_plan_for_tester("PLAN-TSK-001")
3. record_observation("Plan loaded, it's a predictive task",
                      "Need to inspect prepared data",
                      "Checking for prepared parquet")
4. python_sandbox_stage3_5("df = load_dataframe('prepared_PLAN-TSK-001.parquet'); print(df.head()); print(df.columns)")
5. record_observation("Data has columns: Year, Rice_Export_USD, etc.",
                      "Data is yearly from 2018-2024",
                      "Will propose 3 suitable methods")
6. record_thought("Data structure clear, proposing methods",
                  "Creating 3 distinct forecasting methods with implementation code")
7. save_method_proposal_output({
     "plan_id": "PLAN-TSK-001",
     "task_category": "predictive",
     "methods_proposed": [
       {"method_id": "METHOD-1", "name": "Moving Average", ...},
       {"method_id": "METHOD-2", "name": "Linear Regression", ...},
       {"method_id": "METHOD-3", "name": "Random Forest", ...}
     ],
     "data_split_strategy": "Train: 2018-2023, Validation: 2024",
     "date_column": "Year",
     "target_column": "Rice_Export_USD",
     "train_period": "2018-2023",
     "validation_period": "2024",
     "test_period": None,
     "data_preprocessing_steps": [...]
   })

═══════════════════════════════════════════════════════════════
FINAL REMINDER
═══════════════════════════════════════════════════════════════

**CRITICAL RULES:**
- Follow ReAct framework (record_thought before, record_observation after)
- Propose exactly 3 distinct methods
- Be dataset-agnostic (discover structure, don't assume)
- Define clear, consistent data split strategy
- Write complete implementation code for each method
- Treat Stage 3B prepared data as the source of truth
- Save comprehensive MethodProposalOutput when complete
- Aim to finish within {max_rounds} rounds
"""


# ===========================
# LangGraph Setup
# ===========================

def truncate_messages(messages: List[BaseMessage], max_history: int = 15) -> List[BaseMessage]:
    """Truncate message history to prevent token overflow."""
    if len(messages) <= max_history + 2:
        return messages
    return [messages[0], messages[1]] + messages[-(max_history):]


def agent_node(state: MessagesState) -> Dict[str, List[BaseMessage]]:
    """Single LLM step with tool calling."""
    truncated_messages = truncate_messages(state["messages"], max_history=15)
    response = llm_with_tools.invoke(truncated_messages)
    return {"messages": [response]}


tool_node = ToolNode(STAGE3_5A_TOOLS)


def _tool_call_history(messages: List[BaseMessage]) -> List[str]:
    """Extract tool call names from conversation history."""
    names: List[str] = []
    for m in messages:
        tool_calls = getattr(m, "tool_calls", None)
        if not tool_calls:
            continue
        for tc in tool_calls:
            name = None
            if isinstance(tc, dict):
                name = tc.get("name")
                if not name and isinstance(tc.get("function"), dict):
                    name = tc["function"].get("name")
            else:
                name = getattr(tc, "name", None)
                if not name:
                    func = getattr(tc, "function", None)
                    if func is not None:
                        name = getattr(func, "name", None)
            if name:
                names.append(name)
    return names


def should_continue(state: MessagesState) -> str:
    """Route based on tool calls."""
    messages = state["messages"]
    last = messages[-1]

    tool_history = _tool_call_history(messages)
    save_called = any(name == "save_method_proposal_output" for name in tool_history)

    # If we just got a tool call, go execute it
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"

    # If already saved, we can end
    if save_called:
        return END

    # Otherwise, nudge to continue
    recent_tool = None
    for m in reversed(messages):
        tc_list = getattr(m, "tool_calls", None)
        if tc_list:
            tc = tc_list[0]
            if isinstance(tc, dict):
                recent_tool = tc.get("name") or (tc.get("function") or {}).get("name")
            else:
                recent_tool = getattr(tc, "name", None)
                if not recent_tool and hasattr(tc, "function"):
                    recent_tool = getattr(tc.function, "name", None)
            break

    reminder = (
        f"No tool call detected. You must continue with your task.\n"
        f"save_method_proposal_output called: {save_called}. "
        f"Most recent tool: {recent_tool or 'none yet'}. "
        f"Continue following the ReAct framework until you call save_method_proposal_output()."
    )
    messages.append(HumanMessage(content=reminder))
    return "agent"


builder = StateGraph(MessagesState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)
builder.set_entry_point("agent")
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", "agent": "agent", END: END})
builder.add_edge("tools", "agent")

memory = MemorySaver()
stage3_5a_app = builder.compile(checkpointer=memory)


# ===========================
# Stage 3.5a Runner
# ===========================

def run_stage3_5a(
    plan_id: str,
    max_rounds: int = STAGE3_5A_MAX_ROUNDS,
    debug: bool = True,
    prepared_metadata: Any = None
) -> Dict:
    """Run Stage 3.5a method proposal.

    Args:
        plan_id: Plan ID from Stage 3 (e.g., 'PLAN-TSK-001')
        max_rounds: Maximum number of agent rounds
        debug: Whether to print debug information
        prepared_metadata: Optional PreparedDataOutput from Stage 3B

    Returns:
        Final state from the graph execution
    """
    from .config import STAGE3_OUT_DIR, STAGE2_OUT_DIR

    # Load exclusion context from Stage 3 plan
    excluded_context = ""
    plan_path = STAGE3_OUT_DIR / f"{plan_id}.json"
    if plan_path.exists():
        try:
            plan_data = json.loads(plan_path.read_text())
            excluded_cols = plan_data.get("excluded_columns", [])

            # Also check task proposal for excluded columns
            task_id = plan_data.get("task_id", "")
            if task_id:
                task_path = STAGE2_OUT_DIR / "task_proposals.json"
                if task_path.exists():
                    task_data = json.loads(task_path.read_text())
                    for proposal in task_data.get("proposals", []):
                        if proposal.get("id") == task_id:
                            task_excluded = proposal.get("excluded_columns", [])
                            excluded_cols.extend(task_excluded)
                            break

            if excluded_cols:
                excluded_context = "\n\n**COLUMNS EXCLUDED DUE TO DATA QUALITY:**\n"
                excluded_context += "The following columns were rejected in earlier stages:\n"
                for ex in excluded_cols:
                    excluded_context += f"- {ex.get('column_name', 'unknown')} from {ex.get('file', 'unknown')}: {ex.get('reason', 'no reason given')}\n"
                excluded_context += "\nBe aware these columns are unavailable.\n"
        except Exception as e:
            print(f"Warning: Could not load excluded columns: {e}")

    system_msg = SystemMessage(content=STAGE3_5A_SYSTEM_PROMPT)

    # Surface prepared data + metadata
    prep_dict = None
    if prepared_metadata:
        try:
            prep_dict = (
                prepared_metadata.model_dump()
                if hasattr(prepared_metadata, "model_dump")
                else prepared_metadata
            )
        except Exception:
            prep_dict = None
    if prep_dict is None:
        prep_files = sorted(STAGE3B_OUT_DIR.glob(f"prep_{plan_id}*.json"))
        if prep_files:
            try:
                prep_dict = json.loads(prep_files[-1].read_text())
            except Exception as e:
                print(f"Warning: Could not load Stage 3B metadata: {e}")

    prepared_parquet = STAGE3B_OUT_DIR / f"prepared_{plan_id}.parquet"
    prepared_file_name = None
    prep_context = ""
    if prep_dict:
        prepared_file_path = prep_dict.get("prepared_file_path")
        prepared_file_name = Path(prepared_file_path).name if prepared_file_path else None
        if not prepared_file_name and prepared_parquet.exists():
            prepared_file_name = prepared_parquet.name
        parquet_hint = (
            f"\n\nPrepared data detected: {prepared_file_name or prepared_parquet}\n"
            f"Load with load_dataframe('{prepared_file_name or prepared_parquet.name}')."
        )
        dq_report = prep_dict.get("data_quality_report", {})
        dq_text = json.dumps(dq_report, indent=2)
        if len(dq_text) > 1200:
            dq_text = dq_text[:600] + "\n...[truncated]...\n" + dq_text[-400:]
        transformations = prep_dict.get("transformations_applied", [])
        columns_created = prep_dict.get("columns_created", [])
        prep_context = (
            "\n\nStage 3B context:\n"
            f"- Prepared file: {prepared_file_path or prepared_parquet}\n"
            f"- Rows: {prep_dict.get('original_row_count')} → {prep_dict.get('prepared_row_count')}\n"
            f"- Columns created: {columns_created}\n"
            f"- Transformations: {transformations}\n"
            f"- Data quality report:\n{dq_text}\n"
        )
    else:
        if prepared_parquet.exists():
            parquet_hint = (
                f"\n\nPrepared data detected: {prepared_parquet}\n"
                f"Load with load_dataframe('{prepared_parquet.name}')."
            )
        else:
            parquet_hint = "\n\nNo prepared parquet found. Proceed with raw data loading."

    human_msg = HumanMessage(
        content=(
            f"Propose 3 forecasting methods for plan '{plan_id}'.{excluded_context}\n\n"
            f"Follow the ReAct framework strictly:\n"
            f"1. DATA UNDERSTANDING: Load plan, inspect data, identify structure\n"
            f"2. METHOD PROPOSAL: Propose 3 distinct forecasting methods with complete implementation code\n"
            f"3. DATA SPLIT: Define clear train/validation/test split strategy\n"
            f"4. SAVE: Call save_method_proposal_output() with complete proposals\n\n"
            f"Use record_thought() BEFORE each action\n"
            f"Use record_observation() AFTER each action\n\n"
            f"Be dataset-agnostic (discover column names dynamically)\n"
            f"Use Stage 3B prepared data if available (already cleaned and formatted)\n"
            f"Write complete, runnable implementation code for each method\n\n"
            f"Your success metric: save_method_proposal_output() called with valid MethodProposalOutput."
            f"{parquet_hint}{prep_context}"
        )
    )

    state: MessagesState = {"messages": [system_msg, human_msg]}

    config = {
        "configurable": {"thread_id": f"stage3_5a-{plan_id}"},
        "recursion_limit": max_rounds + 50
    }

    if not debug:
        return stage3_5a_app.invoke(state, config=config)

    print("=" * 80)
    print(f"📋 STAGE 3.5a: Method Proposal for {plan_id}")
    print("=" * 80)

    final_state = None
    prev_len = 0
    round_num = 0

    for curr_state in stage3_5a_app.stream(
        state,
        config=config,
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
                            if isinstance(v, str) and len(v) > 200:
                                print(f"     {k}: {v[:100]}...[truncated]...{v[-100:]}")
                            else:
                                print(f"     {k}: {v}")
            elif "Tool" in msg_type:
                print("\n📥 Tool Result:")
                content = m.content
                if len(content) > 500:
                    print(content[:250] + "\n...[truncated]...\n" + content[-250:])
                else:
                    print(content)

        prev_len = len(msgs)
        final_state = curr_state

        if round_num >= max_rounds:
            print(f"\n⚠️  Reached max rounds ({max_rounds}). Stopping.")
            break

    print("\n" + "=" * 80)
    print(f"✅ Complete - {round_num} rounds")
    print("=" * 80)

    # ===========================
    # POST-EXECUTION SAVE VERIFICATION
    # ===========================
    print("\n" + "=" * 80)
    print("🔍 VERIFYING SAVE...")
    print("=" * 80)

    # Check for looping pattern (agent repeatedly claiming completion)
    completion_claims = 0
    for msg in messages[-10:] if len(messages) >= 10 else messages:
        content = getattr(msg, "content", "") or ""
        if any(phrase in content.lower() for phrase in [
            "successfully completed",
            "successfully finalized",
            "ready for implementation",
            "proposal completed",
            "methods are ready"
        ]):
            completion_claims += 1

    if completion_claims >= 3:
        print(f"⚠️  LOOP DETECTED: Agent claimed completion {completion_claims} times without actual save")
        print("🔧 Triggering force-save mechanism...")

    # Check if file actually exists
    proposal_files = sorted(STAGE3_5A_OUT_DIR.glob(f"method_proposal_{plan_id}*.json"))
    
    if proposal_files:
        print(f"✅ Verified: Proposal file exists: {proposal_files[-1].name}")
        return final_state
    
    # File doesn't exist - agent hallucinated the save!
    print("⚠️  WARNING: Agent claimed to save but file doesn't exist!")
    print("🔧 Attempting to extract proposal from agent messages and force-save...")
    
    # Try to extract the proposal JSON from agent messages
    messages = final_state.get("messages", [])
    proposal_data = None
    
    for msg in reversed(messages):
        content = getattr(msg, "content", "") or ""
        
        # Look for JSON in tool call arguments
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                if isinstance(tc, dict):
                    name = tc.get("name")
                    args = tc.get("args", {})
                else:
                    name = getattr(tc, "name", None)
                    args = getattr(tc, "args", {}) or {}
                
                if name == "save_method_proposal_output":
                    output_json = args.get("output_json")
                    if output_json:
                        if isinstance(output_json, dict):
                            proposal_data = output_json
                            print(f"✓ Found proposal in tool call arguments")
                            break
                        elif isinstance(output_json, str):
                            try:
                                proposal_data = json.loads(output_json)
                                print(f"✓ Found proposal JSON string in tool call")
                                break
                            except:
                                pass
        
        if proposal_data:
            break
    
    if proposal_data:
        # Force-save the proposal
        try:
            from .models import MethodProposalOutput
            from datetime import datetime
            
            # Validate the data
            proposal_output = MethodProposalOutput.model_validate(proposal_data)
            
            # Save it
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            STAGE3_5A_OUT_DIR.mkdir(parents=True, exist_ok=True)
            output_path = STAGE3_5A_OUT_DIR / f"method_proposal_{plan_id}_{timestamp}.json"
            output_path.write_text(json.dumps(proposal_data, indent=2))
            
            print(f"✅ FORCE-SAVED: {output_path.name}")
            print(f"   Methods: {len(proposal_data.get('methods_proposed', []))}")
            print(f"   Data split: {proposal_data.get('data_split_strategy', 'N/A')[:60]}...")
            
        except Exception as e:
            print(f"❌ Force-save failed: {e}")
            print("   Manual intervention required - check agent logs for proposal JSON")
    else:
        print("❌ Could not extract proposal from agent messages")
        print("   The agent may not have generated a valid proposal")
        print("   Check the reasoning blocks for the proposal structure")
    
    return final_state


# ===========================
# State Node for Master Graph
# ===========================

def stage3_5a_node(state: dict) -> dict:
    """Stage 3.5a node for the master pipeline graph.

    Args:
        state: Current pipeline state with stage3_plan set

    Returns:
        Updated state with method_proposal_output populated
    """
    from .config import STAGE3_5A_OUT_DIR

    stage3_plan = state.get("stage3_plan")
    if not stage3_plan:
        print("ERROR: No Stage 3 plan available for Stage 3.5a")
        state["errors"].append("Stage 3.5a: No Stage 3 plan available")
        return state

    plan_id = stage3_plan.plan_id

    # Check for prepared data from Stage 3B
    prepared_data = state.get("prepared_data")
    if prepared_data:
        print(f"\n✅ Stage 3B prepared data available: {prepared_data.prepared_file_path}")
    else:
        print(f"\n⚠️  No prepared data from Stage 3B - agent will load raw data")

    print(f"\n📋 Starting Stage 3.5a for: {plan_id}\n")

    result = run_stage3_5a(plan_id, debug=True, prepared_metadata=prepared_data)

    # Check for saved method proposal output
    proposal_files = sorted(STAGE3_5A_OUT_DIR.glob(f"method_proposal_{plan_id}*.json"))
    if proposal_files:
        latest_file = proposal_files[-1]
        print(f"\n✅ SUCCESS! Method proposal saved to: {latest_file}")
        proposal_data = json.loads(latest_file.read_text())
        state["method_proposal_output"] = MethodProposalOutput.model_validate(proposal_data)
        state["completed_stages"].append(3.5)
        state["current_stage"] = 3.6  # Move to Stage 3.5b
    else:
        print("\n⚠️  WARNING: Method proposal not saved. Check logs above.")
        state["errors"].append("Stage 3.5a: Method proposal not saved")

    return state


if __name__ == "__main__":
    # Run Stage 3.5a standalone
    import sys
    if len(sys.argv) < 2:
        print("Usage: python stage3_5a_agent.py <plan_id>")
        print("Example: python stage3_5a_agent.py PLAN-TSK-001")
        sys.exit(1)

    plan_id = sys.argv[1].strip()
    run_stage3_5a(plan_id)
