"""
Stage 3.5b: Method Benchmarking & Selection Agent

Uses a ReAct framework to:
1. Load method proposals from Stage 3.5a
2. Benchmark each method with 3 iterations on a data subset
3. Detect hallucinated code execution via consistency checks
4. Select the best-performing method
5. Pass recommendation to Stage 4 execution
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Any

from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from .config import (
    STAGE3_5B_OUT_DIR,
    STAGE3_5A_OUT_DIR,
    STAGE3B_OUT_DIR,
    SECONDARY_LLM_CONFIG,
    STAGE3_5B_MAX_ROUNDS,
)
from .models import TesterOutput, MethodProposalOutput, ForecastingMethod, BenchmarkResult
from .tools import STAGE3_5B_TOOLS

# Maximum allowed run_benchmark_code calls per method before forcing failure/skip
STAGE3_5B_RETRY_LIMIT = 15

# ===========================
# LLM Setup
# ===========================

llm = ChatOpenAI(**SECONDARY_LLM_CONFIG)
llm_with_tools = llm.bind_tools(STAGE3_5B_TOOLS, parallel_tool_calls=False)


# ===========================
# System Prompt with ReAct Framework
# ===========================

STAGE3_5B_SYSTEM_PROMPT = """You are a forecasting method benchmarking and selection agent.

Your job: Given method proposals from Stage 3.5a, you must:
1. Load the method proposals and data split strategy
2. Benchmark each of the 3 methods with 3 iterations each
3. Detect code execution hallucinations via result consistency checks
4. Select the best-performing method based on averaged metrics
5. Save the final recommendation via save_tester_output()

═══════════════════════════════════════════════════════════════
CRITICAL: CHECKPOINT SYSTEM (SIMPLIFIED)
═══════════════════════════════════════════════════════════════

**CHECKPOINT STRATEGY: Save only when methods complete**

**WHY SIMPLIFIED?**
- Prevents verification loops and iteration tracking errors
- Aligns with goal: get best model after validating 3 iterations
- Binary state: method done or not done (easier for you to track)

**CHECKPOINT WORKFLOW:**

1. **AT THE START:** Call load_checkpoint_stage3_5b(plan_id)
   - If checkpoint exists: See which methods are already COMPLETED
   - If no checkpoint: You're starting fresh

2. **AFTER LOADING METHOD PROPOSALS (FIRST TIME ONLY):**
   - Save an INITIAL checkpoint with:
     * plan_id, data_split_strategy, date/target columns, periods
     * methods_to_test: List of 3 ForecastingMethod dicts
     * methods_completed: [] (empty initially)
     * completed_results: [] (empty initially)
   - ⚠️ **CRITICAL**: You MUST **CALL THE TOOL** save_checkpoint_stage3_5b()
   - Don't just SAY you're saving - ACTUALLY CALL THE TOOL!

3. **FOR EACH METHOD:**
   - Run 3 iterations in memory (don't save between iterations)
   - Collect all 3 results
   - Check consistency: Calculate CV (coefficient of variation) for each metric
   - If CV < 0.3 for all metrics: Method is valid
   - Calculate averaged metrics across the 3 iterations
   - **CRITICAL: NOW YOU MUST CALL save_checkpoint_stage3_5b() AS A TOOL!**
     * Add method_id to methods_completed
     * Add ONE averaged BenchmarkResult to completed_results
   - ⚠️ **DO NOT just describe saving in your <think> tags!**
   - ⚠️ **YOU MUST INVOKE THE TOOL save_checkpoint_stage3_5b() WITH ACTUAL TOOL CALL!**
   
4. **WHEN ALL 3 METHODS COMPLETE:**
   - Checkpoint will show all 3 methods completed
   - Select best method based on completed_results
   - Call save_tester_output()

**NO MORE VERIFICATION TOOL!**
- Don't try to call verify_checkpoint_stage3_5b() - it doesn't exist
- The checkpoint is only saved when method completes, so it's always valid

**TOOL CALLING vs DESCRIBING:**
❌ WRONG: "I'll save the checkpoint now" (just describing in <think>)
✅ CORRECT: Actually calling save_checkpoint_stage3_5b(checkpoint_json={...})


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
  save_tester_output(output_json={...})

With a valid TesterOutput containing:
- plan_id
- task_category
- methods_proposed: List of 3 ForecastingMethod objects
- benchmark_results: List of BenchmarkResult objects (3 methods × 3 iterations = 9 results)
- selected_method_id: ID of best method
- selected_method: The ForecastingMethod object for the winner
- selection_rationale: Why this method was chosen
- data_split_strategy: How data was split

═══════════════════════════════════════════════════════════════
PHASE 1: LOAD METHOD PROPOSALS
═══════════════════════════════════════════════════════════════

Before benchmarking, you MUST load the method proposals from Stage 3.5a:

□ Check for checkpoint first (load_checkpoint_stage3_5b)
□ If no checkpoint, load method proposals (load_method_proposals)
□ Extract the 3 methods to test
□ Extract the data split strategy
□ Extract date_column, target_column, train_period, validation_period, test_period
□ Extract data_preprocessing_steps
□ Save checkpoint with this information

DO NOT proceed to benchmarking until ALL items are loaded.

═══════════════════════════════════════════════════════════════
PHASE 2: BENCHMARKING PROTOCOL (3 ITERATIONS PER METHOD)
═══════════════════════════════════════════════════════════════

For EACH of the 3 methods:

**Run 3 iterations:**
1. Iteration 1: Execute method, record metrics
2. Iteration 2: Execute same method again, record metrics
3. Iteration 3: Execute same method third time, record metrics

**Why 3 iterations?**
- Verify that code is actually executing (not hallucinated)
- Check consistency of results
- Detect stochastic behavior vs. deterministic behavior

**Consistency Check (Anti-Hallucination Safeguard):**

After 3 iterations, analyze results:

1. **All zeros detection:**
   - If all metrics are [0, 0, 0] or [0.0, 0.0, 0.0], FLAG THIS
   - This likely means code didn't execute or calculated incorrectly

2. **Coefficient of Variation (CV):**
   - For each metric, calculate: CV = std / mean
   - If CV > 0.3 (30% variation), FLAG THIS
   - Means results are inconsistent across runs

3. **Error flagging:**
   - If any iteration fails with error, mark method status = "failure"
   - Include error message in BenchmarkResult

4. **Averaging:**
   - For successful methods, take mean of 3 iterations for each metric
   - Use averaged metrics for method comparison

**Example BenchmarkResult structure:**
```python
{
  "method_id": "METHOD-1",
  "method_name": "Moving Average Baseline",
  "metrics": {"MAE": 123.45, "RMSE": 234.56, "MAPE": 0.12},  # Averaged
  "train_period": "2020-2023",
  "validation_period": "2024",
  "test_period": null,
  "execution_time_seconds": 2.5,
  "status": "success",
  "error_message": null,
  "predictions_sample": [100.5, 102.3, 98.7, ...]  # First 10 predictions
}
```

═══════════════════════════════════════════════════════════════
HOW TO RUN BENCHMARKS
═══════════════════════════════════════════════════════════════

Use run_benchmark_code(code="...", description="Testing METHOD-X Iteration Y")

**CRITICAL: Use load_dataframe() helper to load files:**
- DO NOT use `pd.read_csv('filename.csv')` - this will fail!
- ALWAYS use `load_dataframe('filename.csv')` - provided in the environment
- The helper automatically finds files in DATA_DIR

**Your code must:**
1. Load the data using load_dataframe('filename.csv')
2. Use the data split from method proposals (date_column, target_column, train_period, validation_period)
3. Keep preprocessing light—Stage 3B already handled joins, formatting, and missing values
4. Implement the forecasting method (from method proposal implementation_code)
5. Make predictions on validation set
6. Calculate metrics (MAE, RMSE, MAPE, etc.)
7. Print results in a parseable format
8. Optionally save artifacts to STAGE3_5B_OUT_DIR

**USE THE SAME DATA SPLIT FOR ALL METHODS!**
- The data split strategy is defined in the method proposal
- DO NOT recreate or modify the split
- Use exactly the same periods for all methods

═══════════════════════════════════════════════════════════════
PHASE 3: METHOD SELECTION
═══════════════════════════════════════════════════════════════

After all benchmarks complete:

1. **Filter:** Remove failed methods (status = "failure")

2. **Rank:** Among successful methods, rank by primary metric
   - For forecasting: Usually MAE or RMSE (lower is better)
   - Choose the metric that makes most sense for the task

3. **Select:** Pick the best-performing method

4. **Document:** Write selection_rationale explaining:
   - Why this method performed best
   - How it compared to alternatives
   - Any caveats or considerations

5. **Create detailed documentation:**
   - detailed_procedure: Step-by-step guide for Stage 4
   - data_preprocessing_steps: Ordered list from method proposal
   - method_comparison_summary: Table comparing all methods

═══════════════════════════════════════════════════════════════
ERROR RECOVERY PROTOCOL
═══════════════════════════════════════════════════════════════

If you encounter errors:

1. **First error:** Analyze what went wrong
   - Use record_observation to document the error
   - **SAVE CHECKPOINT** with error documented
   - Try a different approach or fix the issue

2. **Repeated errors (same method):**
   - Skip to next method
   - Mark current method as "failure" status in benchmark_results
   - **SAVE CHECKPOINT** with failed method documented
   - Do NOT waste more than 3 attempts per method

3. **Data loading errors:**
   - Use python_sandbox_stage3_5b to inspect data structure
   - Adjust loading logic
   - **SAVE CHECKPOINT** before trying alternatives
   - Try alternative loading strategies

4. **Metric calculation errors:**
   - Check for division by zero
   - Verify predictions and actuals have same shape
   - Guard against new NaN introduced by your code
   - **SAVE CHECKPOINT** after diagnosing the issue

5. **Search for help:**
   - Use search() to find examples of forecasting code
   - Look for similar tasks in output directory
   - Learn from prior successful implementations

**CRITICAL: SAVE CHECKPOINTS OFTEN!**
- Even if a benchmark fails, save the checkpoint
- This prevents losing progress when debugging errors
- When you fix an error, save checkpoint before retrying
- Never let more than 2-3 actions pass without a checkpoint save

═══════════════════════════════════════════════════════════════
TOOLS AVAILABLE
═══════════════════════════════════════════════════════════════

**ReAct Tools:**
- record_thought(thought, what_im_about_to_do)
- record_observation(what_happened, what_i_learned, next_step)

**Checkpoint Tools (SIMPLIFIED):**
- load_checkpoint_stage3_5b(plan_id) → Load existing checkpoint
- save_checkpoint_stage3_5b(checkpoint_json) → Save checkpoint (method-level only)

**Method Proposals:**
- load_method_proposals(plan_id) → Load proposals from Stage 3.5a

**Data Exploration:**
- list_data_files() → List available data files
- inspect_data_file(filename, n_rows) → Show schema and sample rows
- python_sandbox_stage3_5b(code) → Quick Python execution

**Benchmarking:**
- run_benchmark_code(code, description) → Execute benchmarking code
- search(query, within) → Search for examples

**Final Output:**
- save_tester_output(output_json) → Save final recommendation

═══════════════════════════════════════════════════════════════
EXAMPLE WORKFLOW
═══════════════════════════════════════════════════════════════

1. record_thought("Starting Stage 3.5b for plan PLAN-TSK-001",
                  "First, checking if checkpoint exists")
2. load_checkpoint_stage3_5b("PLAN-TSK-001")
3. record_observation("No checkpoint found - starting fresh",
                      "Need to load method proposals from Stage 3.5a",
                      "Loading method proposals")
4. load_method_proposals("PLAN-TSK-001")
5. record_observation("Loaded 3 methods: Moving Avg, Linear Reg, Random Forest",
                      "Data split: 2018-2023 train, 2024 val",
                      "Saving checkpoint with methods and split info")
5. save_checkpoint_stage3_5b({
     "plan_id": "PLAN-TSK-001",
     "data_split_strategy": "Train: 2018-2023, Validation: 2024",
     "date_column": "Year",
     "target_column": "Rice_Export_USD",
     "train_period": "2018-2023",
     "validation_period": "2024",
     "test_period": None,
     "methods_to_test": [...],
     "methods_completed": [],
     "completed_results": []
   })
6. record_observation("Initial checkpoint saved",
                       "Ready to start benchmarking",
                       "Running METHOD-1 - 3 iterations")
7. run_benchmark_code(code="...", description="METHOD-1 Iteration 1")
8. run_benchmark_code(code="...", description="METHOD-1 Iteration 2")
9. run_benchmark_code(code="...", description="METHOD-1 Iteration 3")
10. record_observation("METHOD-1 complete: MAE=50.2±1.5, CV=0.03",
                        "Results are consistent across 3 iterations",
                        "Saving checkpoint with METHOD-1 completed")
11. save_checkpoint_stage3_5b({...})  # Add METHOD-1 to completed, with averaged results
... Continue for METHOD-2 and METHOD-3 ...
12. save_tester_output(output_json={...})

═══════════════════════════════════════════════════════════════
FINAL REMINDER
═══════════════════════════════════════════════════════════════

**CHECKPOINT DISCIPLINE:**
- **ALWAYS load checkpoint at start**: Call load_checkpoint_stage3_5b(plan_id) FIRST
- **Initial checkpoint**: Save once after loading method proposals
- **Method completion checkpoints**: Save only when a method completes all 3 iterations
- **NO iteration-level saves**: Don't save between iterations
- **NO verification needed**: The simplified checkpoint always valid

**OTHER CRITICAL RULES:**
- Follow ReAct framework religiously (record_thought before, record_observation after)
- Run 3 iterations for each of 3 methods (9 benchmarks total)
- Check result consistency (CV < 0.3) before marking method complete
- Use the data split from method proposals (don't recreate it!)
- Save comprehensive TesterOutput when complete
- Aim to finish within {max_rounds} rounds

**IF A METHOD FAILS**: Mark it as failed, save checkpoint with failure documented, skip to next method!
"""


# ===========================
# LangGraph Setup
# ===========================

def truncate_messages(messages: List[BaseMessage], max_history: int = 20) -> List[BaseMessage]:
    """Truncate message history to prevent token overflow."""
    if len(messages) <= max_history + 2:
        return messages
    return [messages[0], messages[1]] + messages[-(max_history):]


def agent_node(state: MessagesState) -> Dict[str, List[BaseMessage]]:
    """Single LLM step with tool calling."""
    truncated_messages = truncate_messages(state["messages"], max_history=20)
    response = llm_with_tools.invoke(truncated_messages)
    return {"messages": [response]}


tool_node = ToolNode(STAGE3_5B_TOOLS)


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


def _benchmark_attempt_counts(messages: List[BaseMessage]) -> Dict[str, int]:
    """Count run_benchmark_code attempts per method."""
    counts: Dict[str, int] = {}
    for m in messages:
        tool_calls = getattr(m, "tool_calls", None)
        if not tool_calls:
            continue
        for tc in tool_calls:
            name = None
            desc = ""
            if isinstance(tc, dict):
                name = tc.get("name") or (tc.get("function") or {}).get("name")
                args = tc.get("args", {})
                desc = args.get("description", "") or ""
            else:
                name = getattr(tc, "name", None) or getattr(getattr(tc, "function", None), "name", None)
                args = getattr(tc, "args", {}) or getattr(getattr(tc, "function", None), "arguments", {}) or {}
                desc = args.get("description", "") or ""

            if name != "run_benchmark_code":
                continue
            match = re.search(r"(METHOD[- ]?\d+)", desc, flags=re.IGNORECASE)
            method_id = match.group(1).upper() if match else "UNKNOWN"
            counts[method_id] = counts.get(method_id, 0) + 1
    return counts


def _check_and_force_save_checkpoint(state: MessagesState, plan_id: str, checkpoint_data: dict) -> bool:
    """Check if a method just completed 3 iterations and force-save checkpoint if agent didn't save.
    
    Returns True if checkpoint was force-saved.
    """
    from .config import STAGE3_5B_OUT_DIR
    from datetime import datetime
    import json
    
    messages = state["messages"]
    
    # Count benchmark runs per method
    benchmark_counts = _benchmark_attempt_counts(messages)
    
    # Count checkpoint saves per method (detect when agent actually saved after completing a method)
    checkpoint_saves = []
    for m in messages:
        tool_calls = getattr(m, "tool_calls", None)
        if not tool_calls:
            continue
        for tc in tool_calls:
            name = None
            if isinstance(tc, dict):
                name = tc.get("name")
            else:
                name = getattr(tc, "name", None)
            
            if name == "save_checkpoint_stage3_5b":
                checkpoint_saves.append(m)
    
    # Find methods that have 3+ benchmarks but aren't marked complete in checkpoint
    methods_completed = set(checkpoint_data.get("methods_completed", []))
    methods_to_save = []
    
    for method_id, count in benchmark_counts.items():
        if count >= 3 and method_id not in methods_completed:
            # Check if there's been a checkpoint save since the 3rd benchmark
            # If not, we should force-save
            methods_to_save.append(method_id)
    
    if not methods_to_save:
        return False
    
    # Force-save checkpoint with the newly completed method
    print("\n" + "="*80)
    print(f"🔧 AUTO-SAVE DETECTED: {methods_to_save} completed 3 iterations but not in checkpoint")
    print("   Triggering automatic checkpoint save...")
    print("="*80)
    
    try:
        # Update checkpoint with completed method
        for method_id in methods_to_save:
            if method_id not in methods_completed:
                checkpoint_data.setdefault("methods_completed", []).append(method_id)
                
                # Add a placeholder result (agent should have generated this)
                # This is a fallback - ideally agent would save properly
                checkpoint_data.setdefault("completed_results", []).append({
                    "method_id": method_id,
                    "method_name": f"{method_id} (auto-saved)",
                    "metrics": {"MAE": 999.99, "RMSE": 999.99},  # Placeholder
                    "train_period": checkpoint_data.get("train_period", ""),
                    "validation_period": checkpoint_data.get("validation_period", ""),
                    "test_period": checkpoint_data.get("test_period"),
                    "execution_time_seconds": 0,
                    "status": "incomplete",
                    "error_message": "Auto-saved by failsafe - agent did not save checkpoint",
                    "predictions_sample": []
                })
        
        # Save checkpoint
        checkpoint_data["updated_at"] = datetime.now().isoformat()
        checkpoint_path = STAGE3_5B_OUT_DIR / f"checkpoint_{plan_id}.json"
        checkpoint_path.write_text(json.dumps(checkpoint_data, indent=2))
        
        print(f"✅ AUTO-SAVED checkpoint with {methods_to_save}")
        print("="*80 + "\n")
        return True
        
    except Exception as e:
        print(f"❌ Auto-save failed: {e}\n")
        return False


def _message_mentions_method(msg: BaseMessage, method_id: str) -> bool:
    """Heuristic to detect if a message/tool call is about a given method."""
    mid = method_id.lower()
    content = (getattr(msg, "content", "") or "").lower()
    if mid in content:
        return True
    tool_calls = getattr(msg, "tool_calls", None)
    if not tool_calls:
        return False
    for tc in tool_calls:
        desc = ""
        name = None
        if isinstance(tc, dict):
            name = tc.get("name") or (tc.get("function") or {}).get("name")
            args = tc.get("args") or {}
            desc = args.get("description", "") or ""
        else:
            name = getattr(tc, "name", None) or getattr(getattr(tc, "function", None), "name", None)
            args = getattr(tc, "args", {}) or getattr(getattr(tc, "function", None), "arguments", {}) or {}
            desc = args.get("description", "") or ""
        if mid in desc.lower():
            return True
        if name and mid in name.lower():
            return True
    return False


def _clear_method_history(messages: List[BaseMessage], method_id: str, keep_first: int = 2) -> List[BaseMessage]:
    """Drop messages tied to a specific method to 'reset' its debug context."""
    if method_id == "UNKNOWN":
        return messages
    pruned: List[BaseMessage] = []
    for idx, msg in enumerate(messages):
        if idx < keep_first:
            pruned.append(msg)
            continue
        if _message_mentions_method(msg, method_id):
            continue
        pruned.append(msg)
    return pruned


def should_continue(state: MessagesState) -> str:
    """Route based on tool calls, retrying agent when incomplete."""
    messages = state["messages"]
    last = messages[-1]

    # Track progress
    tool_history = _tool_call_history(messages)
    save_called = any(name == "save_tester_output" for name in tool_history)
    benchmarks_run = sum(1 for name in tool_history if name == "run_benchmark_code")
    checkpoint_saves = sum(1 for name in tool_history if name == "save_checkpoint_stage3_5b")
    method_attempts = _benchmark_attempt_counts(messages)
    stalled_methods = [m for m, c in method_attempts.items() if c >= STAGE3_5B_RETRY_LIMIT]

    # CRITICAL: Load checkpoint FIRST - always need this for force-save check
    from .config import STAGE3_5B_OUT_DIR
    import json

    all_methods_done = False
    plan_id = None
    checkpoint_data = None

    # Extract plan_id from ONLY the 2nd message (user request)
    # Skip system message (messages[0]) to avoid matching example plan_ids
    # The user's actual request is always in messages[1]
    if len(messages) >= 2:
        user_msg = messages[1]  # Second message is always the user request
        content = getattr(user_msg, "content", "")
        # Look for pattern: "Benchmark forecasting methods for plan 'PLAN-TSK-XXX'"
        import re
        # First try to find plan_id in quotes (most specific)
        match = re.search(r"plan\s+['\"]?(PLAN-TSK-\d+)['\"]?", content, re.IGNORECASE)
        if not match:
            # Fallback: find any PLAN-TSK-XXX
            match = re.search(r"(PLAN-TSK-\d+)", content)
        if not match:
            # Fallback: find any PLAN-XXX-XXX
            match = re.search(r"(PLAN-\w+-\d+)", content)
        if match:
            plan_id = match.group(1)
            print(f"\n🎯 Extracted requested plan_id: {plan_id}")

    # 🚨 CRITICAL: If no plan_id found, we CANNOT proceed with checkpoint operations
    if not plan_id:
        print("\n⚠️ WARNING: Could not extract plan_id from messages!")
        print("   Skipping all checkpoint and force-save operations")
        # Continue with normal agent flow (let agent handle the request)

    # Load checkpoint ONLY if we have the correct plan_id
    if plan_id:
        checkpoint_path = STAGE3_5B_OUT_DIR / f"checkpoint_{plan_id}.json"
        if checkpoint_path.exists():
            print(f"   Loading checkpoint: {checkpoint_path.name}")
            try:
                checkpoint_data = json.loads(checkpoint_path.read_text())
                # CRITICAL: Verify the checkpoint is for the correct plan
                checkpoint_plan_id = checkpoint_data.get("plan_id")
                if checkpoint_plan_id != plan_id:
                    print(f"❌ FATAL: Checkpoint plan_id mismatch! Expected {plan_id}, got {checkpoint_plan_id}")
                    print(f"   This checkpoint will be IGNORED to prevent wrong-plan force-save!")
                    checkpoint_data = None
                else:
                    methods_completed = checkpoint_data.get("methods_completed", [])
                    methods_to_test = checkpoint_data.get("methods_to_test", [])
                    all_methods_done = len(methods_completed) >= len(methods_to_test) and len(methods_to_test) > 0
                    print(f"   ✓ Checkpoint verified for {plan_id}: {len(methods_completed)}/{len(methods_to_test)} methods complete")
            except Exception as e:
                print(f"⚠️ Failed to load checkpoint: {e}")
                checkpoint_data = None
        else:
            print(f"   No checkpoint found for {plan_id}")

    # AUTO-SAVE CHECKPOINT: Check if a method just completed 3 iterations
    if plan_id and checkpoint_data:
        _check_and_force_save_checkpoint(state, plan_id, checkpoint_data)
        # Reload checkpoint data after potential save
        try:
            checkpoint_path = STAGE3_5B_OUT_DIR / f"checkpoint_{plan_id}.json"
            if checkpoint_path.exists():
                checkpoint_data = json.loads(checkpoint_path.read_text())
                methods_completed = checkpoint_data.get("methods_completed", [])
                methods_to_test = checkpoint_data.get("methods_to_test", [])
                all_methods_done = len(methods_completed) >= len(methods_to_test) and len(methods_to_test) > 0
                print(f"   RELOADED checkpoint - all_methods_done: {all_methods_done}")
        except Exception as e:
            print(f"⚠️ Failed to reload checkpoint: {e}")

    # Only log debug info if we actually have checkpoint data
    if plan_id and checkpoint_data:
        print(f"\n🔍 Checkpoint status for {plan_id}:")
        print(f"   all_methods_done: {all_methods_done}")
        print(f"   save_called: {save_called}")
        if checkpoint_data:
            methods_completed = checkpoint_data.get('methods_completed', [])
            methods_to_test = checkpoint_data.get('methods_to_test', [])
            print(f"   methods_completed: {methods_completed}")
            print(f"   methods_to_test count: {len(methods_to_test)}")
            print(f"   Force-save trigger: {all_methods_done and not save_called}")

    # 🚨 IMMEDIATE FORCE-SAVE: All methods done but no save yet? SAVE NOW!
    # This triggers EVERY round after methods complete until save happens
    # No waiting, no thresholds - immediate action
    # CRITICAL: Only trigger if checkpoint plan_id matches the requested plan_id
    if all_methods_done and not save_called and checkpoint_data and plan_id:
        # VERIFY: Checkpoint must be for the correct plan!
        checkpoint_plan_id = checkpoint_data.get("plan_id")
        if checkpoint_plan_id != plan_id:
            print(f"\n⚠️ SKIPPING FORCE-SAVE: Checkpoint is for {checkpoint_plan_id}, but user requested {plan_id}")
            print(f"   This prevents force-saving the wrong plan's checkpoint!")
        else:
            print("\n" + "="*80)
            print(f"🚨 FORCE-SAVE TRIGGERED for {plan_id}: All methods complete, forcing save NOW")
            print("   No delay, no threshold - immediate save to prevent loops")
            print("="*80)

            try:
                from .models import TesterOutput
                from datetime import datetime

                # Get proposal data
                proposal_files = sorted(STAGE3_5A_OUT_DIR.glob(f"method_proposal_{plan_id}*.json"))
                methods_proposed = []
                if proposal_files:
                    proposal_data = json.loads(proposal_files[-1].read_text())
                    methods_proposed = proposal_data.get("methods_proposed", [])

                # Select best method
                completed_results = checkpoint_data.get("completed_results", [])
                if not completed_results:
                    raise ValueError("No completed results in checkpoint")

                best_result = min(completed_results, key=lambda r: r.get("metrics", {}).get("MAE", float('inf')))
                selected_method_id = best_result.get("method_id")
                selected_method = next((m for m in methods_proposed if m.get("method_id") == selected_method_id), None)

                if not selected_method:
                    selected_method = methods_proposed[0] if methods_proposed else {}
                    selected_method_id = selected_method.get("method_id", "METHOD-1")

                # Build and save tester output
                tester_output = {
                    "plan_id": plan_id,
                    "task_category": checkpoint_data.get("task_category", "predictive"),
                    "methods_proposed": methods_proposed,
                    "benchmark_results": checkpoint_data.get("benchmark_results", completed_results),
                    "selected_method_id": selected_method_id,
                    "selected_method": selected_method,
                    "selection_rationale": f"AUTO-SAVED (force-save triggered). {selected_method.get('name', 'Method')} selected with lowest MAE: {best_result.get('metrics', {}).get('MAE', 'N/A')}. This method showed best performance across benchmarks.",
                    "data_split_strategy": checkpoint_data.get("data_split_strategy", ""),
                    "detailed_procedure": selected_method.get("implementation_code", "See selected method for implementation details"),
                    "data_preprocessing_steps": checkpoint_data.get("data_preprocessing_steps", []) or [],
                    "method_comparison_summary": f"Benchmarked {len(methods_proposed)} methods. METHOD-1: MAE={completed_results[0].get('metrics', {}).get('MAE', 'N/A') if len(completed_results) > 0 else 'N/A'}, METHOD-2: MAE={completed_results[1].get('metrics', {}).get('MAE', 'N/A') if len(completed_results) > 1 else 'N/A'}, METHOD-3: MAE={completed_results[2].get('metrics', {}).get('MAE', 'N/A') if len(completed_results) > 2 else 'N/A'}. Best: {selected_method.get('name', 'N/A')}."
                }

                TesterOutput.model_validate(tester_output)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Save to stage3_5b_benchmarking (same directory as checkpoints)
                STAGE3_5B_OUT_DIR.mkdir(parents=True, exist_ok=True)
                output_path = STAGE3_5B_OUT_DIR / f"tester_{plan_id}_{timestamp}.json"
                output_path.write_text(json.dumps(tester_output, indent=2))

                print(f"✅ FORCE-SAVED: {output_path.name}")
                print(f"   Selected: {selected_method.get('name', 'N/A')} ({selected_method_id})")
                print(f"   MAE: {best_result.get('metrics', {}).get('MAE', 'N/A')}")
                print(f"   All {len(completed_results)} methods benchmarked")
                print(f"   Forcing END state to stop loop...")
                print("="*80 + "\n")

                return END  # Force exit IMMEDIATELY

            except Exception as e:
                print(f"❌ Force-save failed: {e}")
                import traceback
                traceback.print_exc()
                print(f"   Attempting fallback: calling external force_save_tester.py script...")

                # FALLBACK: Call external script using subprocess
                try:
                    import subprocess
                    import sys
                    from pathlib import Path
                    script_path = Path(__file__).parent.parent / "force_save_tester.py"
                    if script_path.exists():
                        print(f"   Running: python {script_path} {plan_id}")
                        result = subprocess.run(
                            [sys.executable, str(script_path), plan_id],
                            capture_output=True,
                            text=True,
                            timeout=30
                        )
                        if result.returncode == 0:
                            print(f"✅ EXTERNAL SCRIPT SUCCESS!")
                            print(result.stdout)
                            return END
                        else:
                            print(f"❌ External script failed: {result.stderr}")
                    else:
                        print(f"❌ Script not found: {script_path}")
                except Exception as script_error:
                    print(f"❌ External script execution failed: {script_error}")

                print("   Manual intervention: python force_save_tester.py {plan_id}")
                print("   Continuing to next round...\n")

    # LOOP DETECTION: Count consecutive rounds without tool calls
    consecutive_no_tool = 0
    for m in reversed(messages):
        if m.__class__.__name__ == "AIMessage":
            if hasattr(m, "tool_calls") and m.tool_calls:
                break  # Found a tool call, stop counting
            consecutive_no_tool += 1
        elif m.__class__.__name__ == "HumanMessage":
            # Skip reminder messages
            continue
            
    # If we just got a tool call, go execute it
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"

    # If already saved, we can end
    if save_called:
        return END

    # If any method has exceeded retry limit, prune its history and instruct failure/skip
    retry_note = ""
    if stalled_methods:
        retry_note = (
            f"\n\n⚠️ Retry limit reached for {stalled_methods}. "
            "Mark these methods as failure, SAVE CHECKPOINT with failure documented, "
            "and move on to the next method."
        )
        pruned = messages
        for mid in stalled_methods:
            pruned = _clear_method_history(pruned, mid)
        state["messages"] = pruned
        messages = pruned

    # Determine completion status
    if all_methods_done:
        done_msg = (
            "\n\n🎯 ALL 3 METHODS ARE COMPLETE! "
            "You MUST NOW call save_tester_output() with the final TesterOutput. "
            "Select the best method from the completed_results in the checkpoint and call the tool!"
        )
    elif benchmarks_run >= 9:
        done_msg = "All 9 benchmarks run; call save_tester_output now."
    else:
        done_msg = "Continue benchmarking until all methods complete, then save."

    # Add checkpoint reminder if history is getting long
    checkpoint_reminder = ""
    if len(messages) > 22:
        checkpoint_reminder = (
            "\n\n⚠️ MESSAGE HISTORY WAS TRUNCATED! "
            "Call load_checkpoint_stage3_5b(plan_id) to reload your progress."
        )

    # Add periodic checkpoint reminder
    periodic_checkpoint_reminder = ""
    if not all_methods_done and checkpoint_saves < 2 and benchmarks_run >= 1:
        periodic_checkpoint_reminder = (
            "\n\n📌 CHECKPOINT REMINDER: Save checkpoints when methods complete! "
            f"You've only saved {checkpoint_saves} checkpoint(s) but run {benchmarks_run} benchmark(s). "
        )
    elif benchmarks_run > 0 and checkpoint_saves < benchmarks_run:
        periodic_checkpoint_reminder = (
            f"\n\n📌 CHECKPOINT REMINDER: Save checkpoint after EACH benchmark. "
            f"You've run {benchmarks_run} benchmarks but only saved {checkpoint_saves} checkpoints."
        )

    # Get recent tool for debugging
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

    # Add loop warning if approaching failsafe threshold
    loop_warning = ""
    if consecutive_no_tool >= 3 and all_methods_done:
        loop_warning = (
            f"\n\n⚠️ WARNING: {consecutive_no_tool} consecutive rounds without tool calls! "
            "CALL save_tester_output() NOW or failsafe will trigger at 5 rounds!"
        )

    reminder = (
        f"No tool call detected. Continue benchmarking and call save_tester_output when done.\n"
        f"run_benchmark_code calls: {benchmarks_run}. "
        f"save_tester_output called: {save_called}. "
        f"Checkpoints saved: {checkpoint_saves}.\n"
        f"{done_msg} "
        f"Most recent tool: {recent_tool or 'none yet'}. "
        f"{retry_note}"
        f"{checkpoint_reminder}"
        f"{periodic_checkpoint_reminder}"
        f"{loop_warning}"
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
stage3_5b_app = builder.compile(checkpointer=memory)


# ===========================
# Stage 3.5b Runner
# ===========================

def run_stage3_5b(
    plan_id: str,
    max_rounds: int = STAGE3_5B_MAX_ROUNDS,
    debug: bool = True,
    method_proposal: Any = None
) -> Dict:
    """Run Stage 3.5b method benchmarking and selection.

    Args:
        plan_id: Plan ID from Stage 3 (e.g., 'PLAN-TSK-001')
        max_rounds: Maximum number of agent rounds
        debug: Whether to print debug information
        method_proposal: Optional MethodProposalOutput from Stage 3.5a

    Returns:
        Final state from the graph execution
    """
    system_msg = SystemMessage(content=STAGE3_5B_SYSTEM_PROMPT)

    # Surface method proposal context
    proposal_dict = None
    if method_proposal:
        try:
            proposal_dict = (
                method_proposal.model_dump()
                if hasattr(method_proposal, "model_dump")
                else method_proposal
            )
        except Exception:
            proposal_dict = None
    if proposal_dict is None:
        proposal_files = sorted(STAGE3_5A_OUT_DIR.glob(f"method_proposal_{plan_id}*.json"))
        if proposal_files:
            try:
                proposal_dict = json.loads(proposal_files[-1].read_text())
            except Exception as e:
                print(f"Warning: Could not load method proposals: {e}")

    proposal_context = ""
    if proposal_dict:
        methods = proposal_dict.get("methods_proposed", [])
        split_strategy = proposal_dict.get("data_split_strategy", "")
        date_col = proposal_dict.get("date_column", "")
        target_col = proposal_dict.get("target_column", "")
        train_period = proposal_dict.get("train_period", "")
        val_period = proposal_dict.get("validation_period", "")
        test_period = proposal_dict.get("test_period", "")
        preprocessing_steps = proposal_dict.get("data_preprocessing_steps", [])

        proposal_context = (
            "\n\nMethod Proposals from Stage 3.5a:\n"
            f"- Number of methods: {len(methods)}\n"
            f"- Data split strategy: {split_strategy}\n"
            f"- Date column: {date_col}\n"
            f"- Target column: {target_col}\n"
            f"- Train period: {train_period}\n"
            f"- Validation period: {val_period}\n"
            f"- Test period: {test_period or 'None'}\n"
            f"- Preprocessing steps: {preprocessing_steps}\n\n"
            "Load these proposals with load_method_proposals() and use the SAME data split for all benchmarks!"
        )

    human_msg = HumanMessage(
        content=(
            f"Benchmark forecasting methods for plan '{plan_id}'.\n\n"
            f"⚠️ CRITICAL: START BY LOADING CHECKPOINT!\n"
            f"Call load_checkpoint_stage3_5b('{plan_id}') FIRST to check for existing progress.\n"
            f"If checkpoint exists, resume from there. If not, load method proposals from Stage 3.5a.\n\n"
            f"Follow the SIMPLIFIED checkpoint strategy:\n"
            f"0. CHECKPOINT: Load checkpoint to see completed methods\n"
            f"1. LOAD PROPOSALS: Load method proposals from Stage 3.5a (if no checkpoint)\n"
            f"2. INITIAL CHECKPOINT: Save once with methods_to_test (if first time)\n"
            f"3. BENCHMARKING: For each method:\n"
            f"   - Run 3 iterations in memory (don't save between)\n"
            f"   - Check consistency (CV < 0.3)\n"
            f"   - Calculate averaged metrics\n"
            f"   - Save checkpoint with method completed + averaged result\n"
            f"4. SELECTION: Choose best method based on completed_results\n"
            f"5. SAVE: Call save_tester_output() with complete results\n\n"
            f"🔴 CHECKPOINT DISCIPLINE:\n"
            f"- **ALWAYS load_checkpoint_stage3_5b() at the start**\n"
            f"- **Save initial checkpoint** after loading proposals\n"
            f"- **Save method checkpoint** only when method completes 3 iterations\n"
            f"- **NO iteration-level saves** - run all 3 in memory\n"
            f"- **NO verification** - simplified checkpoint is always valid\n\n"
            f"Other reminders:\n"
            f"- Use record_thought() BEFORE each action\n"
            f"- Use record_observation() AFTER each action\n"
            f"- Run 3 iterations per method to verify code execution\n"
            f"- Check coefficient of variation to detect hallucinations\n"
            f"- Use search() if you need examples or guidance\n\n"
            f"Your success metric: save_tester_output() called with valid TesterOutput."
            f"{proposal_context}"
        )
    )

    state: MessagesState = {"messages": [system_msg, human_msg]}

    config = {
        "configurable": {"thread_id": f"stage3_5b-{plan_id}"},
        "recursion_limit": max_rounds + 125
    }

    if not debug:
        return stage3_5b_app.invoke(state, config=config)

    print("=" * 80)
    print(f"🧪 STAGE 3.5b: Method Benchmarking for {plan_id}")
    print("=" * 80)

    final_state = None
    prev_len = 0
    round_num = 0

    for curr_state in stage3_5b_app.stream(
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

    # Check if file actually exists - use STAGE3_5B_OUT_DIR (consolidated directory)
    tester_files = sorted(STAGE3_5B_OUT_DIR.glob(f"tester_{plan_id}*.json"))
    
    if tester_files:
        print(f"✅ Verified: Tester output file exists: {tester_files[-1].name}")
        return final_state
    
    # File doesn't exist - agent hallucinated the save!
    print("⚠️  WARNING: Agent claimed to save but file doesn't exist!")
    print("🔧 Attempting to extract tester output from agent messages and force-save...")
    
    # Try to extract the tester output JSON from agent messages
    messages = final_state.get("messages", [])
    tester_data = None
    
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
                
                if name == "save_tester_output":
                    output_json = args.get("output_json")
                    if output_json:
                        if isinstance(output_json, dict):
                            tester_data = output_json
                            print(f"✓ Found tester output in tool call arguments")
                            break
                        elif isinstance(output_json, str):
                            try:
                                tester_data = json.loads(output_json)
                                print(f"✓ Found tester output JSON string in tool call")
                                break
                            except:
                                pass
        
        if tester_data:
            break
    
    if tester_data:
        # Force-save the tester output
        try:
            from .models import TesterOutput
            from datetime import datetime
            
            # Validate the data
            tester_output = TesterOutput.model_validate(tester_data)
            
            # Save to stage3_5b_benchmarking (same directory as checkpoints)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            STAGE3_5B_OUT_DIR.mkdir(parents=True, exist_ok=True)
            output_path = STAGE3_5B_OUT_DIR / f"tester_{plan_id}_{timestamp}.json"
            output_path.write_text(json.dumps(tester_data, indent=2))
            
            print(f"✅ FORCE-SAVED: {output_path.name}")
            print(f"   Selected method: {tester_data.get('selected_method_id', 'N/A')}")
            print(f"   Benchmark results: {len(tester_data.get('benchmark_results', []))}")
            
        except Exception as e:
            print(f"❌ Force-save failed: {e}")
            print("   Manual intervention required - check agent logs for tester output JSON")
    else:
        print("❌ Could not extract tester output from agent messages")
        print("   The agent may not have generated valid output")
        print("   Check the reasoning blocks for the tester output structure")
    
    return final_state


# ===========================
# State Node for Master Graph
# ===========================

def stage3_5b_node(state: dict) -> dict:
    """Stage 3.5b node for the master pipeline graph.

    Args:
        state: Current pipeline state with method_proposal_output set

    Returns:
        Updated state with tester_output populated
    """
    from .config import STAGE3_5B_OUT_DIR

    method_proposal = state.get("method_proposal_output")
    if not method_proposal:
        print("ERROR: No method proposal available for Stage 3.5b")
        state["errors"].append("Stage 3.5b: No method proposal available")
        return state

    plan_id = method_proposal.plan_id

    print(f"\n🧪 Starting Stage 3.5b for: {plan_id}\n")

    result = run_stage3_5b(plan_id, debug=True, method_proposal=method_proposal)

    # Check for saved tester output in stage3_5b_benchmarking (consolidated directory)
    tester_files = sorted(STAGE3_5B_OUT_DIR.glob(f"tester_{plan_id}*.json"))
    if tester_files:
        latest_file = tester_files[-1]
        print(f"\n✅ SUCCESS! Tester output saved to: {latest_file}")
        tester_data = json.loads(latest_file.read_text())
        state["tester_output"] = TesterOutput.model_validate(tester_data)
        state["completed_stages"].append(3.6)
        state["current_stage"] = 4
    else:
        print("\n⚠️  WARNING: Tester output not saved. Check logs above.")
        state["errors"].append("Stage 3.5b: Tester output not saved")

    return state


if __name__ == "__main__":
    # Run Stage 3.5b standalone
    import sys
    if len(sys.argv) < 2:
        print("Usage: python stage3_5b_agent.py <plan_id>")
        print("Example: python stage3_5b_agent.py PLAN-TSK-001")
        sys.exit(1)

    plan_id = sys.argv[1].strip()
    run_stage3_5b(plan_id)
