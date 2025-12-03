"""
Stage 3B: Data Preparation Agent

Uses a ReAct framework to:
1. Read Stage 3 execution plan
2. Load and merge datasets per plan instructions
3. Apply filters, joins, and transformations
4. Create feature engineering columns
5. Handle missing values
6. Save clean, prepared data for downstream stages
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

from .config import STAGE3B_OUT_DIR, SECONDARY_LLM_CONFIG, STAGE3B_MAX_ROUNDS, STAGE_FILE_PATHS, FILE_NAMING_PATTERNS
from .models import PreparedDataOutput
from .tools import STAGE3B_TOOLS

# ===========================
# LLM Setup
# ===========================

llm = ChatOpenAI(**SECONDARY_LLM_CONFIG)
llm_with_tools = llm.bind_tools(STAGE3B_TOOLS, parallel_tool_calls=False)


# ===========================
# System Prompt with ReAct Framework
# ===========================

STAGE3B_SYSTEM_PROMPT = """You are a data preparation agent.

Your job: Given a Stage 3 execution plan, prepare clean, formatted data for downstream stages.

WORKFLOW:
1. Load Stage 3 execution plan
2. Load and merge data files per file_instructions
3. Apply filters per file_instructions
4. Perform joins per join_steps
5. Apply feature_engineering transformations
6. Handle missing values COMPLETELY (impute/drop with rationale; no NaN allowed in output)
7. Standardize dtypes so numeric/date columns are usable downstream
8. Validate expected columns exist
9. Save prepared, model-ready data to parquet
10. Call save_prepared_data() with detailed metadata

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
  save_prepared_data(
      plan_id=...,
      prepared_file_name=...,
      original_row_count=...,
      prepared_row_count=...,
      columns_created=...,
      transformations_applied=...,
      data_quality_report=...
  )

With valid metadata about the prepared data.

═══════════════════════════════════════════════════════════════
DATA PREPARATION CHECKLIST (MANDATORY)
═══════════════════════════════════════════════════════════════

Follow these steps systematically:

□ Load Stage 3 plan (load_stage3_plan_for_prep)
□ Understand task goal and data requirements
□ Inspect all required data files (inspect_data_file)
□ Load base dataset from file_instructions[0]
□ Apply filters to base dataset
□ Load additional datasets from file_instructions[1:]
□ Apply filters to each dataset
□ Perform joins per join_steps (use left_on/right_on)
□ Apply feature_engineering transformations
□ Handle missing values END-TO-END (impute or drop; ensure no NaN/inf remain)
□ Normalize dtypes for modeling (dates to datetime, numeric to floats/ints)
□ Validate all expected_columns exist
□ Calculate data quality metrics (nulls, duplicates, etc.)
□ VERIFY NO NANS: Run a specific check to ensure 0 nulls remain.
□ Save prepared DataFrame to parquet
□ Call save_prepared_data() with metadata

═══════════════════════════════════════════════════════════════
MODEL-READY GUARANTEE (NON-NEGOTIABLE)
═══════════════════════════════════════════════════════════════
- Stage 3.5 will directly consume your output. Deliver a fully formatted, imputed dataset.
- Do NOT leave NaN/inf/-inf in prepared_df. Impute or remove with justification.
- You MUST verify that `df.isnull().sum().sum() == 0` before saving.
- Document missing-value strategy and dtype fixes in transformations_applied and data_quality_report.

═══════════════════════════════════════════════════════════════
STAGE 3 PLAN STRUCTURE
═══════════════════════════════════════════════════════════════

The plan JSON has these key sections:

**file_instructions**: List of files to load
  - file_id: filename
  - alias: short name (e.g., 'export', 'production')
  - filters: list of pandas query strings (e.g., "Crop == 'Rice'")
  - keep_columns: columns to keep after loading
  - rename_columns: dict of old_name -> new_name

**join_steps**: How to merge datasets
  - step: order number
  - left_table: alias of left table
  - right_table: alias of right table (or null for base)
  - join_type: 'base', 'left', 'inner', 'outer'
  - left_on: list of left join keys
  - right_on: list of right join keys

**feature_engineering**: Features to create
  - feature_name: name of new column
  - description: what it represents
  - transform: type (lag, growth, etc.)
  - depends_on: source columns
  - implementation: pandas code snippet

**validation.time_split**: Train/test split info
  - method: how to split (year-based, etc.)
  - train_years: training period
  - test_years: testing period

═══════════════════════════════════════════════════════════════
HOW TO PREPARE DATA
═══════════════════════════════════════════════════════════════

Use run_data_prep_code(code="...", description="...") for main processing.

**Example workflow:**

```python
import pandas as pd
import numpy as np

# Load base dataset
base_df = load_dataframe('filename1.csv')

# Apply filters
base_df = base_df.query("filter expression")

# Keep specific columns
base_df = base_df[['col1', 'col2', 'col3']]

# Load second dataset
other_df = load_dataframe('filename2.csv')
other_df = other_df.query("another filter")

# Perform join
merged_df = base_df.merge(
    other_df,
    left_on=['key1'],
    right_on=['key2'],
    how='left'
)

# Feature engineering
merged_df['new_feature'] = merged_df['col1'].shift(1)
merged_df['growth_rate'] = (merged_df['col2'] - merged_df['col3']) / merged_df['col3']

# Handle missing values - CRITICAL STEP
# Check for NaNs
print("NaNs before fix:", merged_df.isnull().sum().sum())
# Impute or drop
merged_df.fillna(merged_df.mean(), inplace=True)
# Verify no NaNs remain
if merged_df.isnull().sum().sum() > 0:
    raise ValueError("NaNs still present after imputation!")

# Save prepared DataFrame
prepared_df = merged_df  # This is what run_data_prep_code looks for
prepared_df.to_parquet(STAGE3B_OUT_DIR / 'prepared_PLAN-TSK-001.parquet')
```

**Data Quality Report:**
After preparing data, calculate quality metrics:
- Total rows
- Null counts per column (MUST BE ZERO)
- Duplicate count
- Column dtypes
- Value ranges
- Missing-value strategy: which columns were imputed/dropped and how

═══════════════════════════════════════════════════════════════
ERROR RECOVERY PROTOCOL
═══════════════════════════════════════════════════════════════

If you encounter errors:

1. **First error:** Analyze what went wrong
   - Use record_observation to document the error
   - Try a different approach or fix the issue

2. **File not found:**
   - Use list_data_files() to see available files
   - Check spelling and path

3. **Join errors:**
   - Inspect both datasets before joining
   - Verify join keys exist in both datasets
   - Check for duplicate keys

4. **Missing column errors:**
   - Inspect DataFrame columns after each transformation
   - Verify expected_columns from plan

5. **Search for help:**
   - Use search() to find examples of data preparation
   - Look for similar tasks in output directory

═══════════════════════════════════════════════════════════════
STATE TRACKING (PREVENT REPETITION)
═══════════════════════════════════════════════════════════════

Keep mental track of:
- Which datasets have been loaded ✓
- Which filters have been applied ✓
- Which joins have been completed ✓
- Which features have been created ✓
- Current phase: LOAD → FILTER → JOIN → FEATURE → VALIDATE → SAVE

═══════════════════════════════════════════════════════════════
TOOLS AVAILABLE
═══════════════════════════════════════════════════════════════

**ReAct Tools:**
- record_thought(thought, what_im_about_to_do)
- record_observation(what_happened, what_i_learned, next_step)

**Data Exploration:**
- load_stage3_plan_for_prep(plan_id) → Returns Stage 3 plan JSON
- list_data_files() → List available data files
- inspect_data_file(filename, n_rows) → Show schema and sample rows
- python_sandbox_stage3b(code) → Quick Python execution for exploration

**Data Preparation:**
- run_data_prep_code(code, description) → Execute main data prep code
- search(query, within) → Search for examples and prior work

**Final Output:**
- save_prepared_data(plan_id, prepared_file_name, ...) → Save metadata

═══════════════════════════════════════════════════════════════
EXAMPLE WORKFLOW
═══════════════════════════════════════════════════════════════

1. record_thought("I need to load the plan and understand requirements", 
                  "Loading Stage 3 plan")
2. load_stage3_plan_for_prep("PLAN-TSK-001")
3. record_observation("Plan loaded, need 2 files: export and production", 
                      "Task is to predict rice exports", 
                      "Inspecting first data file")
4. inspect_data_file("export-file.csv")
5. record_observation("Export file has 8 rows, 23 columns",
                      "Data is in wide format (year columns)",
                      "Inspecting production file next")
6. inspect_data_file("production-file.csv")
7. record_thought("Both files inspected, ready to prepare data",
                  "Running main data preparation code")
8. run_data_prep_code(code="...", description="Load, merge, and transform data")
9. record_observation("Data prepared successfully, 48 rows, 15 columns",
                      "All transformations applied correctly",
                      "Saving prepared data metadata")
10. save_prepared_data(plan_id="PLAN-TSK-001", ...)

═══════════════════════════════════════════════════════════════
FINAL REMINDER
═══════════════════════════════════════════════════════════════

- Follow ReAct framework religiously (record_thought before, record_observation after)
- Execute transformations systematically per plan
- Validate data at each step
- **ENSURE NO NANS REMAIN** - Check `df.isnull().sum().sum() == 0`
- Save prepared data as parquet for fast loading
- Call save_prepared_data() when complete
- Aim to finish within reasonable rounds
"""


# ===========================
# LangGraph Setup
# ===========================

def agent_node(state: MessagesState) -> Dict[str, List[BaseMessage]]:
    """Single LLM step with tool calling."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


tool_node = ToolNode(STAGE3B_TOOLS)


def should_continue(state: MessagesState) -> str:
    """Route based on tool calls."""
    last = state["messages"][-1]
    
    # Check if save_prepared_data was successfully called in recent messages
    # If so, we're done and should terminate
    for msg in reversed(state["messages"][-5:]):  # Check last 5 messages
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tool_call in msg.tool_calls:
                if tool_call.get("name") == "save_prepared_data":
                    # Check if there's a successful response
                    msg_idx = state["messages"].index(msg)
                    if msg_idx + 1 < len(state["messages"]):
                        next_msg = state["messages"][msg_idx + 1]
                        if hasattr(next_msg, 'content') and 'saved::prep_' in next_msg.content:
                            # Success! Terminate
                            return END
    
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
stage3b_app = builder.compile(checkpointer=memory)


# ===========================
# Stage 3B Runner
# ===========================

def run_stage3b(plan_id: str, max_rounds: int = STAGE3B_MAX_ROUNDS, debug: bool = True) -> Dict:
    """Run Stage 3B data preparation.
    
    Args:
        plan_id: Plan ID from Stage 3 (e.g., 'PLAN-TSK-001')
        max_rounds: Maximum number of agent rounds
        debug: Whether to print debug information
        
    Returns:
        Final state from the graph execution
    """
    from .config import STAGE3_OUT_DIR
    import json
    
    # Load Stage 3 plan to check for excluded columns
    plan_path = STAGE3_OUT_DIR / f"{plan_id}.json"
    excluded_context = ""
    if plan_path.exists():
        try:
            plan_data = json.loads(plan_path.read_text())
            excluded_cols = plan_data.get("excluded_columns", [])
            if excluded_cols:
                excluded_context = "\n\n**EXCLUDED COLUMNS FROM STAGE 3:**\n"
                excluded_context += "The following columns were considered but excluded during planning:\n"
                for ex in excluded_cols:
                    excluded_context += f"- {ex.get('column_name', 'unknown')} from {ex.get('file', 'unknown')}: {ex.get('reason', 'no reason given')}\n"
                excluded_context += "\nDo NOT attempt to use these columns. They were excluded for data quality reasons.\n"
        except Exception as e:
            print(f"Warning: Could not load excluded columns: {e}")
    
    system_msg = SystemMessage(content=STAGE3B_SYSTEM_PROMPT)
    human_msg = HumanMessage(
        content=(
            f"Prepare data according to plan '{plan_id}'.{excluded_context}\n\n"
            f"Follow the ReAct framework strictly:\n"
            f"1. DATA LOADING: Load plan, inspect files, load datasets\n"
            f"2. FILTERING: Apply filters per file_instructions\n"
            f"3. JOINING: Merge datasets per join_steps\n"
            f"4. FEATURE ENGINEERING: Create new columns per plan\n"
            f"5. VALIDATION: Check expected columns exist and dtypes are consistent\n"
            f"6. MODEL-READY: Handle missing values fully (no NaN/inf left) and record the strategy\n"
            f"7. SAVE: Save prepared data and call save_prepared_data() with rich metadata\n\n"
            f"Remember:\n"
            f"- Use record_thought() BEFORE each action\n"
            f"- Use record_observation() AFTER each action\n"
            f"- **CRITICAL**: Ensure NO NaNs remain in the final dataset. Check `df.isnull().sum().sum() == 0`.\n"
            f"- **CRITICAL**: Ensure NO NaNs remain in the final dataset. Check `df.isnull().sum().sum() == 0`.\n"
            f"- Save as parquet: 'prepared_{plan_id}.parquet' (matches FILE_NAMING_PATTERNS['stage3b_data'])\n"
            f"- Calculate data quality report (include missing-value handling and dtype fixes)\n"
            f"- Do NOT leave unresolved missing values; Stage 3.5 will reuse this file directly\n"
            f"- Use search() if you need examples or guidance\n\n"
            f"Your success metric: save_prepared_data() called with valid metadata."
        )
    )

    state: MessagesState = {"messages": [system_msg, human_msg]}

    # Configure with higher recursion limit
    config = {
        "configurable": {"thread_id": f"stage3b-{plan_id}"},
        "recursion_limit": max_rounds + 5
    }

    if not debug:
        return stage3b_app.invoke(state, config=config)

    print("=" * 80)
    print(f"🔧 STAGE 3B: Data Preparation for {plan_id}")
    print("=" * 80)

    final_state = None
    prev_len = 0
    round_num = 0

    for curr_state in stage3b_app.stream(
        state,
        config=config,
        stream_mode="values",
    ):
        msgs = curr_state["messages"]
        new_msgs = msgs[prev_len:]

        for m in new_msgs:
            msg_type = m.__class__.__name__
            if "System" in msg_type:
                print("\\n" + "─" * 80)
                print("💻 [SYSTEM]")
                print("─" * 80)
                print(m.content[:500] + "..." if len(m.content) > 500 else m.content)
            elif "Human" in msg_type:
                print("\\n" + "─" * 80)
                print("👤 [USER]")
                print("─" * 80)
                print(m.content)
            elif "AI" in msg_type:
                round_num += 1
                print("\\n" + "═" * 80)
                print(f"🤖 [AGENT - Round {round_num}]")
                print("═" * 80)
                if m.content:
                    print("\\n💭 Reasoning:")
                    content = m.content
                    if len(content) > 1000:
                        print(content[:500] + "\\n...[truncated]...\\n" + content[-500:])
                    else:
                        print(content)
                
                if hasattr(m, 'tool_calls') and m.tool_calls:
                    print("\\n🔧 Tool Calls:")
                    for tc in m.tool_calls:
                        name = tc.get("name", "UNKNOWN")
                        args = tc.get("args", {})
                        print(f"\\n  📌 {name}")
                        for k, v in args.items():
                            if isinstance(v, str) and len(v) > 200:
                                print(f"     {k}: {v[:100]}...[truncated]...{v[-100:]}")
                            else:
                                print(f"     {k}: {v}")
            elif "Tool" in msg_type:
                print("\\n📥 Tool Result:")
                content = m.content
                if len(content) > 500:
                    print(content[:250] + "\\n...[truncated]...\\n" + content[-250:])
                else:
                    print(content)

        prev_len = len(msgs)
        final_state = curr_state
        
        if round_num >= max_rounds:
            print(f"\\n⚠️  Reached max rounds ({max_rounds}). Stopping.")
            break

    print("\\n" + "=" * 80)
    print(f"✅ Complete - {round_num} rounds")
    print("=" * 80)
    return final_state


# ===========================
# State Node for Master Graph
# ===========================

def stage3b_node(state: dict) -> dict:
    """Stage 3B node for the master pipeline graph.
    
    Args:
        state: Current pipeline state with stage3_plan set
        
    Returns:
        Updated state with prepared_data populated
    """
    stage3_plan = state.get("stage3_plan")
    if not stage3_plan:
        print("ERROR: No Stage 3 plan available for Stage 3B")
        state["errors"].append("Stage 3B: No Stage 3 plan available")
        return state
    
    plan_id = stage3_plan.plan_id
    print(f"\\n🔧 Starting Stage 3B for: {plan_id}\\n")
    
    result = run_stage3b(plan_id, debug=True)
    
    # Check for saved prep output
    prep_files = sorted(STAGE3B_OUT_DIR.glob(f"prep_{plan_id}*.json"))
    if prep_files:
        latest_file = prep_files[-1]
        print(f"\\n✅ SUCCESS! Prepared data metadata saved to: {latest_file}")
        prep_data = json.loads(latest_file.read_text())
        state["prepared_data"] = PreparedDataOutput.model_validate(prep_data)
        state["completed_stages"].append(3.2)  # Using 3.2 for ordering
        state["current_stage"] = 3.5
    else:
        print("\\n⚠️  WARNING: Prepared data metadata not saved. Check logs above.")
        state["errors"].append("Stage 3B: Prepared data metadata not saved")
    
    return state


if __name__ == "__main__":
    # Run Stage 3B standalone
    import sys
    if len(sys.argv) < 2:
        print("Usage: python stage3b_agent.py <plan_id>")
        print("Example: python stage3b_agent.py PLAN-TSK-001")
        sys.exit(1)
    
    plan_id = sys.argv[1].strip()
    run_stage3b(plan_id)
