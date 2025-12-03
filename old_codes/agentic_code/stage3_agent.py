"""
Stage 3: Execution Planning Agent

Creates detailed execution plans for selected analytical tasks using LangGraph.
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

from .config import STAGE3_OUT_DIR, SECONDARY_LLM_CONFIG, STAGE3_MAX_ROUNDS
from .models import Stage3Plan
from .tools import STAGE3_TOOLS
from .failsafe_agent import run_failsafe


# ===========================
# LLM Setup
# ===========================

llm = ChatOpenAI(**SECONDARY_LLM_CONFIG)
llm_with_tools = llm.bind_tools(STAGE3_TOOLS, parallel_tool_calls=False)


# ===========================
# System Prompt
# ===========================

STAGE3_SYSTEM_PROMPT = """You are a data pipeline planning agent.

Your job: given a selected analytical task and the available data files, you must produce a
**valid, executable Stage3Plan** and save it by calling the tool `save_stage3_plan(plan_json=...)`.

═══════════════════════════════════════════════════════════════
CRITICAL RULES
═══════════════════════════════════════════════════════════════

1. You MUST end by calling save_stage3_plan(plan_json=...)  
   • This is YOUR ONLY success criterion.

2. NEVER write JSON in your reasoning.  
   • Build the JSON **silently** and pass it ONLY as the tool argument.

3. Be dataset-agnostic.  
   • No domain-specific assumptions; infer everything from:
     - task proposal (load_task_proposal)
     - data file schemas (inspect_data_file)
     - search() results.

4. Set plan_id EXACTLY to: "PLAN-{task_id}"  
   • task_id is the selected task id (e.g. "TSK-001").

5. ADEQUACY CHECK  
   • If the task proposal + inspected data seem insufficient
     (e.g. unclear target, join keys, time dimension, or feature candidates),
     you MUST call search() and/or further inspect_data_file() **before**
     building the final plan.

6. **DATA VALIDATION (≥65% NON-NAN) - MANDATORY**
   • Before finalizing ANY column in file_instructions:
     - Verify (1 - null_fraction) >= 0.65 using inspect_data_file() or python_sandbox()
     - If a column has > 35% missing data, DO NOT use it
     - Find alternative columns or document in notes

7. **CURRENCY PREFERENCE (INR > USD)**
   • If both INR and USD columns exist, ALWAYS use INR
   • Exception: Only use USD if user explicitly requested it
   • Document choice in plan notes

8. **KEEP JSON CONCISE**
   • Feature engineering "implementation" fields MUST be brief (< 100 chars)
   • Use simple pandas syntax: "df['new_col'] = df['col1'] / df['col2']"
   • DO NOT include full column names repeatedly
   • Use abbreviations where possible without losing clarity

═══════════════════════════════════════════════════════════════
VALIDATION WORKFLOW (EXECUTE FIRST)
═══════════════════════════════════════════════════════════════

Before creating the execution plan:

STEP 1: Load task proposal and inspect all required_files

STEP 2: For EACH column mentioned (target, features, join keys):
   - Use python_sandbox() to check: `df['column'].notna().sum() / len(df)`
   - Require: completeness >= 0.65
   - If fails: Find alternative or document issue

STEP 3: If both INR and USD columns exist:
   - Select INR column
   - Update all references in plan

STEP 4: Document in plan notes:
   - "Data validation: All columns verified ≥65% complete"
   - "Currency: Using INR as per preference" (if applicable)

═══════════════════════════════════════════════════════════════
TOOLS YOU CAN USE
═══════════════════════════════════════════════════════════════

You have at least these tools (names may be more, but these are key):

- load_task_proposal(task_id)
    → returns the chosen task proposal with fields like:
      - id, category, title, problem_statement
      - required_files
      - join_plan (with hypothesized_keys)
      - target (name, granularity, horizon)
      - feature_plan (candidates, transform_ideas, handling_missingness)
      - validation_plan, quality_checks, expected_outputs

- list_data_files()
    → lists available raw data files and their identifiers.

- inspect_data_file(filename or file_id)
    → returns schema information (column names, dtypes, sample rows, etc.).

- search(query, within='project|data|output|code|all', ...)
    → workspace text search. Use it to:
      - see how a file was used previously,
      - find existing join logic,
      - find derived features or evaluation choices.

- save_stage3_plan(plan_json=...)
    → FINAL mandatory step with the complete Stage3Plan JSON (as a string).

═══════════════════════════════════════════════════════════════
REACT-STYLE LOOP (HOW YOU SHOULD THINK)
═══════════════════════════════════════════════════════════════

On each turn BEFORE the final save:

1. Briefly think about what you know and what is missing.  
   - Example: "I know the target and candidate features, but I still need the exact
     columns for the join between file A and file B."

2. Decide on the next ACTION:
   - load_task_proposal(...)
   - list_data_files()
   - inspect_data_file(...)
   - search(...)

3. Use the tool(s) and update your understanding based on the observations.

Repeat this loop (THOUGHT → ACTION → TOOL → NEW THOUGHT) a few times until you are
confident you can build a high-quality plan. Then, without showing the JSON, call:
   save_stage3_plan(plan_json="...")

Do NOT explain the plan to the user. The only observable effect must be the
save_stage3_plan tool call.

═══════════════════════════════════════════════════════════════
HOW TO USE THE TASK PROPOSAL
═══════════════════════════════════════════════════════════════

When you call load_task_proposal(task_id), carefully read and use:

1. category
   - "predictive" → task_category should be "predictive"
   - "descriptive" → "descriptive"
   - "unsupervised" → "unsupervised"

2. required_files
   - These are the primary files you should focus on in file_instructions and join_steps.

3. join_plan.hypothesized_keys
   - Treat these as **candidate join key sets**, not guaranteed truths.
   - For each candidate set:
     • Use inspect_data_file() to confirm which columns actually exist in EACH file.
     • Build a mapping: alias → list of its columns.
     • Only treat a column as a join key if:
         - It exists in BOTH tables you want to join, AND
         - Sample values suggest they refer to the same concept.
   - If the same column name appears in both tables and represents the same concept,
     prefer using "join_keys": ["col_name"].
   - If the column names differ but represent the same concept, use:
       "left_on": ["left_name"],
       "right_on": ["right_name"]
   - NEVER join on a column that does not appear in the inspected schema for that table.
   - If a hypothesized key appears only in one table:
       • You MAY keep it as a feature or grouping column in that table,
       • But you MUST NOT reference it as a join key for the other table
         unless you explicitly create a derived/normalized key in that table
         (and document this in key_normalization / rename_columns).
   - If you cannot find any safe common key for a join, mention this in "notes"
     and either:
       • Do NOT invent join columns.
       • Do NOT create a join step with empty join_keys and empty left_on/right_on.
       • Instead, keep only the base step in join_steps and explain in notes that
         the additional file is not joined due to missing compatible keys.
       • leave the tables unjoined, or
       • propose a cautious strategy in key_normalization (without fabricating
         non-existent columns).

4. target
   - Use target.name and target.granularity to:
     • Ensure keep_columns includes target column and grouping columns.
     • Inform expected_model_types:
       - If there's a clear time dimension / horizon → include "Time Series"
       - Otherwise for numeric target → include "Regression"
   - If no target (descriptive or unsupervised), set expected_model_types accordingly
     (e.g. "Aggregation", "Visualization", "Clustering").

5. feature_plan
   - Ensure file_instructions.keep_columns covers:
     • all candidate feature columns,
     • any lag/derived feature sources,
     • join keys and grouping columns.
   - In feature_engineering, create entries that:
     • describe the feature,
     • specify its transform (mean, sum, difference, growth rate, lag, etc.),
     • specify depends_on source columns,
     • give a short implementation hint (pseudo-code is fine).

6. validation_plan & quality_checks
   - Map to "validation.time_split" (if time-based validation is suggested),
     "coverage_checks", "cardinality_checks", and "additional_checks".
   - If there is no real time dimension or explicit validation plan, set
     time_split.method = "none" and note what you did.
   - If the proposal suggests time-series or temporal validation, choose one of:
       • method="year-based"   (splits by year-like columns)
       • method="date-based"   (splits by actual dates)
       • method="rolling-window" (rolling-window evaluation)
     and describe the ranges in train_years / test_years / validation_years as strings.

═══════════════════════════════════════════════════════════════
STRUCTURE OF Stage3Plan (WHAT YOU BUILD SILENTLY)
═══════════════════════════════════════════════════════════════

You must SILENTLY construct a JSON object like this (do not show it in reasoning):

{
  "plan_id": "PLAN-{task_id}",
  "selected_task_id": "{task_id}",
  "goal": "Short description of what this plan will achieve",
  "task_category": "descriptive|predictive|unsupervised",

  "artifacts": {
    "intermediate_table": "{task_id}_data.parquet",
    "intermediate_format": "parquet",
    "expected_columns": [ ... ],                 // at least join keys, target, group-by, features
    "expected_row_count_range": [min, max]       // optional but preferred, if you can infer a rough range
  },

  "key_normalization": [
    // If you need to harmonize keys (e.g. trimming spaces, standardizing codes),
    // describe the transformations here.
  ],

  "file_instructions": [
    {
      "file_id": "FILE-XXX",                     // whatever identifier your tools expose
      "original_name": "raw_filename.ext",
      "alias": "short_logical_name",
      "rename_columns": { "Raw Name": "clean_name" },
      "keep_columns": [ "clean_name1", "clean_name2", "..."],
      "filters": [ ... ],
      "join_keys": [ "key_col1", "key_col2" ],   // keys used by this file in joins (if any)
      "notes": null
    }
  ],

  "join_steps": [
    {
      "step": 1,
      "description": "Load base table",
      "left_table": "base_alias",
      "right_table": null,
      "join_type": "base",
      "join_keys": [],
      "left_on": [],
      "right_on": [],
      "expected_cardinality": "base",
      "validation": {
        "check_duplicates_on_keys": [],
        "expected_unique": false,
        "check_row_count_stable": false,
        "check_no_duplicates_introduced": false,
        "acceptable_coverage": null,
        "max_cardinality_ratio": null
      }
    }
    // Further join steps for other tables, with meaningful join_keys or left_on/right_on
  ],

  "feature_engineering": [
    {
      "feature_name": "new_feature",
      "description": "What it captures",
      "transform": "lag/growth/ratio/mean/etc",
      "depends_on": ["base_col1", "base_col2"],
      "implementation": "df['new_feature'] = df['col1'] / df['col2']"
    }
  ],

  "validation": {
    "time_split": {
      "method": "none|year-based|date-based|rolling-window",
      "train_years": null,
      "test_years": null,
      "validation_years": null,
      "leakage_check": "How you avoid temporal leakage if applicable"
    },
    "coverage_checks": [
      // Each coverage check must be an object with these fields:
      {
        "check": "Brief name of the check",
        "threshold": 0.65,  // numeric threshold (e.g., 0.65 for 65% coverage)
        "description": "Full description of what is being checked",
        "action_if_violation": "What to do if check fails (can be null)"
      }
    ],
    "cardinality_checks": [
      // Each cardinality check must be an object with these fields:
      {
        "check": "Brief name of the check",
        "expected": "Expected cardinality relationship (e.g., '1:1', '1:N', 'unique')",
        "action_if_violation": "What to do if check fails"
      }
    ],
    "additional_checks": ["Data loaded", "No duplicates on join keys"]
  },

  "expected_model_types": [
    "Aggregation",
    "Visualization",
    "Regression",
    "Classification",
    "Time Series",
    "Clustering"
    // pick only the ones that truly apply
  ],

  "evaluation_metrics": [
    "Summary Statistics",
    "MAE",
    "RMSE",
    "R2",
    "Accuracy",
    "Silhouette Score"
    // choose relevant metrics for this task
  ],

  "notes": [
    "Any important caveats, assumptions, or follow-up questions."
  ]
}

═══════════════════════════════════════════════════════════════
VALIDATION FIELD RULES (MUST FOLLOW SCHEMA)
═══════════════════════════════════════════════════════════════

- For validation.time_split.method you MUST use EXACTLY one of:
  • "none"
  • "year-based"
  • "date-based"
  • "rolling-window"
  Do NOT invent new values like "time_based" or "kfold".

- For validation.time_split.train_years, test_years, validation_years:
  • Each MUST be either null or a SINGLE STRING, not a list.
  • Examples of VALID values:
      "2018-19 to 2022-23"
      "all years before 2023-24"
      "rolling window over available years"
  • Examples of INVALID values you MUST NOT use:
      ["2020-21", "2021-22"]
      ["2023-24"]

If you are unsure about exact year ranges, keep these fields as null and explain
the intended strategy briefly in leakage_check or notes instead of guessing.

═══════════════════════════════════════════════════════════════
JOIN FIELD RULES (MUST MATCH REAL COLUMNS)
═══════════════════════════════════════════════════════════════

For EVERY entry in join_steps:

1. Alias sanity:
   - left_table and right_table:
     • MUST be aliases that appear in file_instructions.alias.
   - For each alias, you MUST base your join logic ONLY on the columns you actually saw
     in inspect_data_file() for that file.

2. Column existence:
   - You MUST build, in your head, a mapping:
       alias → set_of_columns_seen_in_inspect_data_file
   - For each join step:
       • Every column in join_keys MUST exist in BOTH alias column sets.
       • Every column in left_on MUST exist in the left_table’s column set.
       • Every column in right_on MUST exist in the right_table’s column set.
   - If a column does NOT appear in the inspected schema for that alias,
     you MUST NOT use it as a join column for that alias.

3. join_type rules:
   - For the base step (loading the first table):
       • join_type MUST be "base".
       • join_keys, left_on, right_on MUST all be empty lists.
   - For ALL OTHER steps (actual joins):
       • join_type MUST be one of: "inner", "left", "right", "outer".
       • You MUST choose EXACTLY ONE of these patterns:
           a) Non-empty join_keys and left_on = [] and right_on = []
           b) Empty join_keys and NON-empty left_on and NON-empty right_on
       • It is INVALID to have:
           - join_keys = [] AND left_on = [] AND right_on = []
           - or to mix join_keys with left_on/right_on at the same time.
       • If you do not have any valid columns to put in join_keys / left_on / right_on,
         you MUST NOT create that join step at all.

4. join_keys vs left_on/right_on:
   - Use join_keys ONLY when the SAME column name is used in both tables.
   - Use left_on/right_on ONLY when names differ but refer to the same concept.
   - NEVER put a column in left_on or right_on that is missing from that table’s schema.
   - NEVER claim "both files contain column X" unless inspect_data_file() showed X for BOTH.

5. Handling hypothesized keys (from join_plan.hypothesized_keys):
   - Treat them as HYPOTHESES, not facts.
   - After inspecting schemas:
       • Keep ONLY those keys that truly map to existing columns in BOTH tables
         (same name or clearly renamable).
       • If a hypothesized key appears only in ONE table:
- If NOT mentioned: Run your own validation (python_sandbox)

═══════════════════════════════════════════════════════════════
VALIDATION WORKFLOW (EXECUTE BEFORE CREATING PLAN)
═══════════════════════════════════════════════════════════════

STEP 1: Load and review task proposal
  - Understand target, features, required files

STEP 2: Validate data completeness
  - Use python_sandbox() to check null percentages
  - Verify ALL columns (target + features + keys) have ≥65% non-NaN
  - Document validation results

STEP 3: Check currency preference
  - If both INR and USD exist, select INR
  - Update column references accordingly

STEP 4: Document validation in plan
  - Add to notes: "Data validation: All columns verified ≥65% complete"
  - If using INR over USD: "Currency: Using INR as per preference"

STEP 5: Create file_instructions with validated columns only

═══════════════════════════════════════════════════════════════
YOUR CORE RESPONSIBILITIES
═══════════════════════════════════════════════════════════════

Once you are satisfied with your plan:

1. SILENTLY serialize the Stage3Plan JSON.
2. Call: save_stage3_plan(plan_json=<that JSON string>).
3. Do NOT print the JSON or explain it.

If you fail to call save_stage3_plan, your work is considered a failure.
"""

# ===========================
# LangGraph
# ===========================

def agent_node(state: MessagesState) -> Dict[str, List[BaseMessage]]:
    """Single LLM step."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


tool_node = ToolNode(STAGE3_TOOLS)


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
stage3_app = builder.compile(checkpointer=memory)


# ===========================
# Stage 3 Runner
# ===========================

def run_stage3(task_id: str, max_rounds: int = STAGE3_MAX_ROUNDS, debug: bool = True) -> Dict:
    """Run Stage 3 planning for a specific task.
    
    Args:
        task_id: Task ID from Stage  2 (e.g., 'TSK-001')
        max_rounds: Maximum number of agent rounds
        debug: Whether to print debug information
        
    Returns:
        Final state from the graph execution
    """
    system_msg = SystemMessage(content=STAGE3_SYSTEM_PROMPT)
    human_msg = HumanMessage(
        content=(
            f"Create a high-quality Stage 3 plan for task '{task_id}'.\n\n"
            f"Follow a ReAct-style loop:\n"
            f"- First call load_task_proposal('{task_id}') and carefully read category, "
            f"required_files, join_plan.hypothesized_keys, target, feature_plan, and validation_plan.\n"
            f"- Then use list_data_files() and inspect_data_file(...) on the required files to "
            f"confirm available columns and candidate join keys.\n"
            f"- If anything is unclear (join keys, target, time dimension, or feature candidates), "
            f"use search(...) to look for how these files or columns were used previously.\n"
            f"- After you fully understand the task and data, silently build the Stage3Plan JSON "
            f"and finish by calling save_stage3_plan(plan_json=...).\n\n"
            f"Rules:\n"
            f"- Set plan_id = 'PLAN-{task_id}'.\n"
            f"- Never print JSON in your reasoning.\n"
            f"- Your ONLY success criterion is to call save_stage3_plan(plan_json=...) with a valid plan.\n"
            f"- Aim to finish within {STAGE3_MAX_ROUNDS} rounds, using tools thoughtfully (not blindly)."
        )
    )


    state: MessagesState = {"messages": [system_msg, human_msg]}

    if not debug:
        return stage3_app.invoke(state, config={"configurable": {"thread_id": f"stage3-{task_id}"}})

    print("=" * 80)
    print(f"🚀 STAGE 3: Planning for {task_id}")
    print("=" * 80)

    final_state = None
    prev_len = 0
    round_num = 0

    for curr_state in stage3_app.stream(
        state,
        config={"configurable": {"thread_id": f"stage3-{task_id}"}},
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

        prev_len = len(msgs)
        final_state = curr_state
        
        if round_num >= max_rounds:
            print(f"\n⚠️  Reached max rounds ({max_rounds}). Stopping.")
            break

    print("\n" + "=" * 80)
    print(f"✅ Complete - {round_num} rounds")
    print("=" * 80)
    return final_state


# ===========================
# State Node for Master Graph
# ===========================

def stage3_node(state: dict) -> dict:
    """Stage 3 node for the master pipeline graph.
    
    Args:
        state: Current pipeline state with selected_task_id set
        
    Returns:
        Updated state with stage3_plan populated
    """
    task_id = state.get("selected_task_id")
    if not task_id:
        print("ERROR: No task_id selected for Stage 3")
        state["errors"].append("Stage 3: No task_id selected")
        return state
    
    print(f"\n🎯 Starting Stage 3 for: {task_id}\n")
    
    result = run_stage3(task_id, debug=True)
    
    # Check for saved plan
    plan_file = STAGE3_OUT_DIR / f"PLAN-{task_id}.json"
    if plan_file.exists():
        print(f"\n✅ SUCCESS! Plan saved to: {plan_file}")
        plan_data = json.loads(plan_file.read_text())
        state["stage3_plan"] = Stage3Plan.model_validate(plan_data)
        state["completed_stages"].append(3)
        state["current_stage"] = 4
    else:
        print("\n⚠️  WARNING: Plan not saved. Check logs above.")
        state["errors"].append("Stage 3: Plan not saved")
        
        try:
            rec = run_failsafe(
                stage="stage3",
                error=f"Plan not saved for task {task_id}",
                context="save_stage3_plan() was not called or failed validation.",
                debug=False,
            )
            state.setdefault("failsafe_history", []).append(rec)
            print(f"\n🛟 Failsafe suggestion recorded: {rec.analysis}")
        except Exception as e:
            print(f"\n⚠️  Failsafe agent failed: {e}")
    
    return state


if __name__ == "__main__":
    # Run Stage 3 standalone
    import sys
    if len(sys.argv) < 2:
        print("Usage: python stage3_agent.py <task_id>")
        print("Example: python stage3_agent.py TSK-001")
        sys.exit(1)
    
    task_id = sys.argv[1].strip()
    run_stage3(task_id)
