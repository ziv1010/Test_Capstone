"""
Stage 3.5: Method Testing & Benchmarking Agent (Tester)

Uses a ReAct framework to:
1. Identify 3 suitable forecasting methods for the task
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
    STAGE3_5_OUT_DIR,
    STAGE3B_OUT_DIR,
    SECONDARY_LLM_CONFIG,
    STAGE3_5_MAX_ROUNDS,
    STAGE_FILE_PATHS,
    FILE_NAMING_PATTERNS,
)
from .models import TesterOutput, ForecastingMethod, BenchmarkResult, PreparedDataOutput
from .tools import STAGE3_5_TOOLS

# Maximum allowed run_benchmark_code calls per method before forcing failure/skip
STAGE3_5_RETRY_LIMIT = 15

# ===========================
# LLM Setup
# ===========================

llm = ChatOpenAI(**SECONDARY_LLM_CONFIG)
llm_with_tools = llm.bind_tools(STAGE3_5_TOOLS, parallel_tool_calls=False)


# ===========================
# System Prompt with ReAct Framework
# ===========================

STAGE3_5_SYSTEM_PROMPT = """You are a forecasting method testing and benchmarking agent.

Your job: Given a Stage 3 plan, you must:
1. Identify 3 suitable forecasting methods for the task
2. Benchmark each method with 3 iterations on a data subset
3. Detect code execution hallucinations via result consistency checks
4. Select the best-performing method based on averaged metrics
5. Save the recommendation via save_tester_output()

═══════════════════════════════════════════════════════════════
CRITICAL: CHECKPOINT SYSTEM (MEMORY MANAGEMENT)
═══════════════════════════════════════════════════════════════

**YOU MUST USE CHECKPOINTS TO MAINTAIN MEMORY!**

Due to conversation history truncation, you WILL lose memory of:
- What methods you planned to test
- What data split you decided on
- Which methods are already completed
- What results you've collected

**CHECKPOINT WORKFLOW (MANDATORY):**

1. **AT THE START:** Call load_checkpoint_stage3_5(plan_id)
   - If checkpoint exists: Resume from where you left off
   - If no checkpoint: You're starting fresh

2. **AFTER DATA UNDERSTANDING:** Once you've identified the data split, save a checkpoint:
   - Call save_checkpoint_stage3_5() with:
     * plan_id
     * data_split_strategy (description of your split)
     * date_column, target_column (if identified)
     * train_period, validation_period, test_period
     * methods_to_test (list of 3 ForecastingMethod dicts)
     * methods_completed: [] (empty at start)
     * benchmark_results: [] (empty at start)
     * iteration_counts: {} (empty at start)

3. **BEFORE AND AFTER EACH BENCHMARK ATTEMPT:** Update the checkpoint:

   **BEFORE attempting a benchmark iteration:**
   - Save checkpoint showing you're ABOUT to run this iteration
   - This ensures progress is preserved even if the benchmark fails

   **AFTER each benchmark attempt (success OR failure):**
   - If successful: Increment iteration_counts, append BenchmarkResult
   - If failed: Document the error in your observations, still increment attempt count
   - If a method reaches 3 iterations, add method_id to methods_completed
   - Save updated checkpoint
   - **CRITICAL**: IMMEDIATELY verify the checkpoint was saved correctly!
     Call verify_checkpoint_stage3_5(plan_id, expected_iterations)
   - If verification fails, re-save with corrected data before continuing
   - Never proceed to next iteration/method without successful verification

   **KEY INSIGHT**: Save checkpoints FREQUENTLY, not just on success. This prevents losing progress when errors occur.

4. **WHEN YOU RESUME:** Check the checkpoint to know:
   - What methods still need testing
   - What data split to use (MUST be consistent!)
   - What results you already have

**Example checkpoint after proposing methods:**
```python
save_checkpoint_stage3_5({
    "plan_id": "PLAN-TSK-001",
    "data_split_strategy": "Train: 2018-2023, Validation: 2024",
    "date_column": "Year",
    "target_column": "Rice_Export_USD",
    "train_period": "2018-2023",
    "validation_period": "2024",
    "test_period": None,
    "methods_to_test": [
        {"method_id": "METHOD-1", "name": "Moving Average", ...},
        {"method_id": "METHOD-2", "name": "Linear Regression", ...},
        {"method_id": "METHOD-3", "name": "Random Forest", ...}
    ],
    "methods_completed": [],
    "benchmark_results": [],
    "iteration_counts": {}
})
```

**Example checkpoint after METHOD-1 iteration 1:**
```python
save_checkpoint_stage3_5({
    # ... same as above ...
    "methods_completed": [],  # METHOD-1 not done yet (needs 3 iterations)
    "benchmark_results": [
        {"method_id": "METHOD-1", "method_name": "Moving Average", "metrics": {...}, ...}
    ],
    "iteration_counts": {"METHOD-1": 1}
})
```

**Example checkpoint after METHOD-1 complete (3 iterations):**
```python
save_checkpoint_stage3_5({
    # ... same as above ...
    "methods_completed": ["METHOD-1"],  # ✓ METHOD-1 is done!
    "benchmark_results": [
        # 3 results for METHOD-1
    ],
    "iteration_counts": {"METHOD-1": 3}
})
```

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

**NEW - DETAILED DOCUMENTATION FOR STAGE 4:**
- detailed_procedure: Step-by-step guide on how to replicate this benchmarking process
  and apply the selected method. Include:
  * How data files were loaded
  * How columns were identified (date, target, features)
  * How data was split (train/validation/test periods)
  * Specific metrics used and why
  * How to interpret the selected method's implementation code
  
- data_preprocessing_steps: Ordered list of preprocessing steps, e.g.:
  * ["Loaded prepared_PLAN-TSK-001.parquet", "Identified date column: Year", 
     "Identified target: Rice_Export_USD", "Split: 2018-2023 train, 2024 val"]
  
- method_comparison_summary: Table or formatted text comparing all methods, e.g.:
  * "METHOD-1 (Moving Avg): MAE=150.2, RMSE=200.5 - Simple baseline
     METHOD-2 (Linear Reg): MAE=120.3, RMSE=165.8 - Best performer ✓
     METHOD-3 (Random Forest): FAILED - overfitting on small dataset"

═══════════════════════════════════════════════════════════════
PHASE 1: DATA UNDERSTANDING (MANDATORY CHECKLIST)
═══════════════════════════════════════════════════════════════

Before benchmarking, you MUST understand the data structure:

□ First, check for Stage 3B prepared data + metadata (preferred path)
□ If prepared parquet exists, load it directly and review its metadata (missing-value handling, columns created)
□ Only inspect raw files if prepared data is missing or corrupt
□ Load the Stage 3 plan (load_stage3_plan_for_tester)
□ Identify required data files from the plan (for context/fallback)
□ Inspect prepared data (or raw files if no prepared data) to see columns and dtypes
□ Determine which column contains dates/timestamps
□ Determine which column is the target variable (from plan)
□ Understand temporal granularity (daily, monthly, yearly)
□ Determine full date range (e.g., 2020-2024)
□ Design train/validation/test split strategy
□ Verify data can be loaded and split (use python_sandbox_stage3_5) WITHOUT redoing cleaning/imputation

DO NOT proceed to benchmarking until ALL items are checked.

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
  "implementation_code": "# Python code snippet",
  "libraries_required": ["pandas", "numpy"]
}
```

**IMPORTANT:** Be dataset-agnostic. DO NOT hardcode column names like "Year" or "Sales".
Instead, write code that discovers column names dynamically from the data.

═══════════════════════════════════════════════════════════════
PHASE 3: BENCHMARKING PROTOCOL (3 ITERATIONS PER METHOD)
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
CRITICAL: USE PREPARED DATA IF AVAILABLE
═══════════════════════════════════════════════════════════════

**Stage 3B may have already prepared the data!**

BEFORE loading raw data files, CHECK if prepared data exists:
- Look for prepared data file mentioned in Stage 3 plan metadata
- Typical format: 'prepared_PLAN-TSK-001.parquet' (or similar pattern)
- Location: STAGE3B_OUT_DIR (see STAGE_FILE_PATHS['stage3b'])
- USE PATTERN MATCHING: Files may be named 'prepared_TSK-...' or 'prepared_PLAN-TSK-...'
- ALWAYS use glob patterns (e.g. `*TSK-001*`) to find files to avoid "not found" errors.

**If prepared data exists:**
✓ Load it directly: `prepared_df = load_dataframe('prepared_PLAN-TSK-001.parquet')`
✓ Skip manual loading, merging, filtering, and missing-value handling (already done)
✓ Prepared data already has joins, filters, features, and imputation applied
✓ You only need to split it for benchmarking; keep preprocessing minimal (e.g., ensure datetime parsing)

**If no prepared data:**
✗ Fall back to loading raw data files manually
✗ Apply filters and joins yourself

═══════════════════════════════════════════════════════════════
HOW TO RUN BENCHMARKS
═══════════════════════════════════════════════════════════════

Use run_benchmark_code(code="...", description="Testing METHOD-X Iteration Y")

**CRITICAL: Use load_dataframe() helper to load files:**
- DO NOT use `pd.read_csv('filename.csv')` - this will fail!
- ALWAYS use `load_dataframe('filename.csv')` - this is provided in the environment
- The helper automatically finds files in DATA_DIR

**Your code must:**
1. Load the data using load_dataframe('filename.csv')
2. Identify date column and target column (DO NOT HARDCODE)
3. Split data:
   - Training: Earlier period (e.g., 2020-2023)
   - Validation: Later period (e.g., 2024)
   - Test: Optional future period (e.g., 2025 if available)
4. Keep preprocessing light—Stage 3B already handled joins, formatting, and missing values. Only enforce type conversions needed for modeling (e.g., to_datetime).
5. Implement the forecasting method
6. Make predictions on validation set
7. Calculate metrics (MAE, RMSE, MAPE, etc.)
8. Print results in a parseable format
9. Optionally save artifacts to STAGE3_5_OUT_DIR

**Metric Calculation:**
- MAE = Mean Absolute Error
- RMSE = Root Mean Squared Error  
- MAPE = Mean Absolute Percentage Error
- R² = Coefficient of determination
- Choose metrics appropriate for forecasting tasks

**Example code structure (dataset-agnostic):**
```python
import pandas as pd
import numpy as np
from pathlib import Path

# CORRECT: Use load_dataframe() helper
df = load_dataframe('Export-of-Rice-Varieties-to-Bangladesh,-2018-19-to-2024-25.csv')

# WRONG: DO NOT use pd.read_csv() directly
# df = pd.read_csv('filename.csv')  # This will fail!

# Discover date columns dynamically
date_cols = [col for col in df.columns if 'date' in col.lower() or 'year' in col.lower() or 'time' in col.lower()]
if not date_cols:
    # Try finding numeric column that looks like years
    date_cols = [col for col in df.columns if df[col].dtype in ['int64', 'int32'] and df[col].min() > 1900]

# For wide-format data (columns are years), reshape to long format
# Example: '2020 - 21-Value (USD)', '2021 - 22-Value (USD)', etc.
value_cols = [col for col in df.columns if 'Value (USD)' in col or 'value' in col.lower()]
if value_cols:
    # Extract years from column names
    years = []
    for col in value_cols:
        # Extract year like "2020 - 21" from "2020 - 21-Value (USD)"
        year_match = col.split('-')[0].strip()
        years.append((year_match, col))
    
    # Use the year columns for forecasting
    # Train on older years, validate on newest year
    
# Calculate metrics
mae = np.mean(np.abs(predictions - actuals))
rmse = np.sqrt(np.mean((predictions - actuals)**2))
mape = np.mean(np.abs((predictions - actuals) / actuals)) * 100

print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"MAPE: {mape}")
```

═══════════════════════════════════════════════════════════════
PHASE 4: METHOD SELECTION
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

═══════════════════════════════════════════════════════════════
ERROR RECOVERY PROTOCOL
═══════════════════════════════════════════════════════════════

If you encounter errors:

1. **First error:** Analyze what went wrong
   - Use record_observation to document the error
   - **SAVE CHECKPOINT** with error documented in observations
   - Try a different approach or fix the issue

2. **Repeated errors (same method):**
   - Skip to next method
   - Mark current method as "failure" status in benchmark_results
   - **SAVE CHECKPOINT** with failed method documented
   - Do NOT waste more than 3 attempts per method

3. **Data loading errors:**
   - Use python_sandbox_stage3_5 to inspect data structure
   - Adjust column discovery logic
   - **SAVE CHECKPOINT** before trying alternative loading strategies
   - Try alternative loading strategies

4. **Metric calculation errors:**
   - Check for division by zero
   - Verify predictions and actuals have same shape
   - Missing values should already be handled by Stage 3B; only guard against new NaN introduced by your split/code
   - **SAVE CHECKPOINT** after diagnosing the issue

5. **Search for help:**
   - Use search() to find examples of forecasting code
   - Look for similar tasks in output directory
   - Learn from prior successful implementations

**CRITICAL FOR ERROR RECOVERY: SAVE CHECKPOINTS OFTEN!**
- Even if a benchmark fails, save the checkpoint documenting the attempt
- This prevents losing progress when debugging errors
- When you fix an error, save checkpoint before retrying
- Never let more than 2-3 actions pass without a checkpoint save

═══════════════════════════════════════════════════════════════
STATE TRACKING (PREVENT REPETITION)
═══════════════════════════════════════════════════════════════

Keep mental track of:
- Which methods have been proposed ✓
- Which methods have been benchmarked ✓
- How many iterations completed per method
- Which approaches have failed (don't retry the same failure)
- Current phase: DATA_UNDERSTANDING → METHOD_PROPOSAL → BENCHMARKING → SELECTION

═══════════════════════════════════════════════════════════════
TOOLS AVAILABLE
═══════════════════════════════════════════════════════════════

**ReAct Tools:**
- record_thought(thought, what_im_about_to_do)
- record_observation(what_happened, what_i_learned, next_step)

**Checkpoint Tools (CRITICAL - USE THESE!):**
- load_checkpoint_stage3_5(plan_id) → Load existing checkpoint to resume progress
- save_checkpoint_stage3_5(checkpoint_json) → Save checkpoint to maintain memory
- verify_checkpoint_stage3_5(plan_id, expected_iterations) → Verify checkpoint was saved correctly

**Data Exploration:**
- load_stage3_plan_for_tester(plan_id) → Returns Stage 3 plan JSON
- list_data_files() → List available data files
- inspect_data_file(filename, n_rows) → Show schema and sample rows
- python_sandbox_stage3_5(code) → Quick Python execution for exploration

**Benchmarking:**
- run_benchmark_code(code, description) → Execute benchmarking code
- search(query, within) → Search for examples and prior work

**Final Output:**
- save_tester_output(output_json) → Save final recommendation

═══════════════════════════════════════════════════════════════
EXAMPLE WORKFLOW
═══════════════════════════════════════════════════════════════

1. record_thought("Starting Stage 3.5 for plan PLAN-TSK-001",
                  "First, checking if checkpoint exists to resume progress")
2. load_checkpoint_stage3_5("PLAN-TSK-001")
3. record_observation("No checkpoint found - starting fresh",
                      "Need to load plan and understand data",
                      "Loading Stage 3 plan")
4. load_stage3_plan_for_tester("PLAN-TSK-001")
5. record_observation("Plan loaded, it's a predictive task with files X, Y",
                      "Need to inspect prepared data",
                      "Checking for prepared parquet")
6. python_sandbox_stage3_5("df = load_dataframe('prepared_PLAN-TSK-001.parquet'); print(df.head()); print(df.columns)")
7. record_observation("Data has columns: Year, Rice_Export_USD, etc.",
                      "Data is yearly from 2018-2024",
                      "Will split: train 2018-2023, validation 2024")
8. record_thought("Data structure clear, now proposing 3 methods and saving checkpoint",
                  "Creating checkpoint with methods and split info to preserve memory")
9. save_checkpoint_stage3_5({
     "plan_id": "PLAN-TSK-001",
     "data_split_strategy": "Train: 2018-2023, Validation: 2024",
     "date_column": "Year",
     "target_column": "Rice_Export_USD",
     "train_period": "2018-2023",
     "validation_period": "2024",
     "test_period": None,
     "methods_to_test": [
       {"method_id": "METHOD-1", "name": "Moving Average", "description": "...", "implementation_code": "...", "libraries_required": [...]},
       {"method_id": "METHOD-2", "name": "Linear Regression", "description": "...", "implementation_code": "...", "libraries_required": [...]},
       {"method_id": "METHOD-3", "name": "Random Forest", "description": "...", "implementation_code": "...", "libraries_required": [...]}
     ],
     "methods_completed": [],
     "benchmark_results": [],
     "iteration_counts": {}
   })
10. record_observation("Checkpoint saved with 3 methods and split strategy",
                       "Memory preserved - can resume from here if needed",
                       "Running METHOD-1 iteration 1")
11. run_benchmark_code(code="...", description="METHOD-1 Iteration 1")
12. record_observation("METHOD-1 Iter 1: MAE=50.2, RMSE=75.3",
                       "Code executed successfully, updating checkpoint",
                       "Saving progress before continuing")
13. save_checkpoint_stage3_5({
      "plan_id": "PLAN-TSK-001",
      "data_split_strategy": "Train: 2018-2023, Validation: 2024",
      "date_column": "Year",
      "target_column": "Rice_Export_USD",
      "train_period": "2018-2023",
      "validation_period": "2024",
      "test_period": None,
      "methods_to_test": [...],  # same 3 methods
      "methods_completed": [],  # METHOD-1 not done yet
      "benchmark_results": [
        {"method_id": "METHOD-1", "method_name": "Moving Average", "metrics": {"MAE": 50.2, "RMSE": 75.3}, "status": "success", ...}
      ],
      "iteration_counts": {"METHOD-1": 1}
    })
14. verify_checkpoint_stage3_5("PLAN-TSK-001", {"METHOD-1": 1})  # VERIFY IT WAS SAVED!
15. record_observation("Checkpoint verified - 1 iteration saved correctly",
                      "Safe to proceed",
                      "Running METHOD-1 iteration 2")
... Continue: run_benchmark_code for METHOD-1 iter 2, save checkpoint, VERIFY, iter 3, save checkpoint, VERIFY with METHOD-1 in methods_completed ...
... Then METHOD-2 and METHOD-3, always saving AND VERIFYING after each iteration ...
16. save_tester_output(output_json={...})

═══════════════════════════════════════════════════════════════
FINAL REMINDER
═══════════════════════════════════════════════════════════════

**CHECKPOINT DISCIPLINE (MOST IMPORTANT!):**
- **ALWAYS load checkpoint at start**: Call load_checkpoint_stage3_5(plan_id) FIRST
- **SAVE CHECKPOINTS FREQUENTLY**:
  * After proposing methods and identifying data split
  * BEFORE each benchmark attempt (to preserve intent)
  * AFTER each benchmark attempt (success OR failure)
  * When fixing errors or trying alternative approaches
  * When a method completes all 3 iterations
  * **RULE OF THUMB**: Save checkpoint every 2-3 significant actions
- **ALWAYS verify checkpoint after saving**: Call verify_checkpoint_stage3_5() immediately after every save
- **If verification fails**: Re-save with corrected data before continuing
- **USE THE SAME SPLIT FOR ALL METHODS**: Get it from checkpoint, don't recreate it

**OTHER CRITICAL RULES:**
- Follow ReAct framework religiously (record_thought before, record_observation after)
- Run 3 iterations for each of 3 methods (9 benchmarks total)
- Check result consistency to detect hallucinations
- Be dataset-agnostic (discover structure, don't assume)
- Treat Stage 3B prepared data as the source of truth; avoid re-cleaning or re-imputing
- Save comprehensive TesterOutput when complete
- Aim to finish within {max_rounds} rounds

**IF YOU'RE STUCK IN ERRORS**: Save checkpoint documenting the issue, then try a different approach or skip to next method. Don't waste time on endless debugging without saving progress!
"""


# ===========================
# LangGraph Setup
# ===========================

def truncate_messages(messages: List[BaseMessage], max_history: int = 20) -> List[BaseMessage]:
    """Truncate message history to prevent token overflow.
    
    Keeps:
    - System message (first message)
    - Initial user message (second message)  
    - Last max_history messages (recent conversation)
    
    Args:
        messages: Full message list
        max_history: Number of recent messages to keep
        
    Returns:
        Truncated message list
    """
    if len(messages) <= max_history + 2:
        return messages
    
    # Keep system message, user message, and last N messages
    return [messages[0], messages[1]] + messages[-(max_history):]


def agent_node(state: MessagesState) -> Dict[str, List[BaseMessage]]:
    """Single LLM step with tool calling."""
    # Truncate message history to prevent token overflow
    truncated_messages = truncate_messages(state["messages"], max_history=20)
    response = llm_with_tools.invoke(truncated_messages)
    return {"messages": [response]}


tool_node = ToolNode(STAGE3_5_TOOLS)


def _tool_call_history(messages: List[BaseMessage]) -> List[str]:
    """Extract tool call names from conversation history."""
    names: List[str] = []
    for m in messages:
        tool_calls = getattr(m, "tool_calls", None)
        if not tool_calls:
            continue
        for tc in tool_calls:
            # LangChain/OpenAI tool_calls can be dict-like or objects with nested function.name
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
    """Count run_benchmark_code attempts per method (parsed from description)."""
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
                args = tc.get("args") or {}
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
    checkpoint_saves = sum(1 for name in tool_history if name == "save_checkpoint_stage3_5")
    method_attempts = _benchmark_attempt_counts(messages)
    stalled_methods = [m for m, c in method_attempts.items() if c >= STAGE3_5_RETRY_LIMIT]

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
            "record the error, clear debug context, and move on to the next method."
        )
        pruned = messages
        for mid in stalled_methods:
            pruned = _clear_method_history(pruned, mid)
        state["messages"] = pruned
        messages = pruned

    # If benchmarks are incomplete or save not called, nudge and continue agent
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

    done_msg = (
        "All 3 methods × 3 iterations are complete; call save_tester_output now."
        if benchmarks_run >= 9
        else "Continue benchmarking until 9 run_benchmark_code calls, then save."
    )

    # Add checkpoint reminder if history is getting long (near truncation threshold)
    checkpoint_reminder = ""
    if len(messages) > 22:  # Just after truncation threshold (20 + system + user)
        checkpoint_reminder = (
            "\n\n⚠️ MESSAGE HISTORY WAS TRUNCATED! "
            "Call load_checkpoint_stage3_5(plan_id) to reload your progress. "
            "The checkpoint contains all methods, data split, and results collected so far."
        )

    # Add periodic checkpoint reminder if not saving frequently enough
    periodic_checkpoint_reminder = ""
    if checkpoint_saves < 2 and benchmarks_run >= 1:
        periodic_checkpoint_reminder = (
            "\n\n📌 CHECKPOINT REMINDER: You should be saving checkpoints frequently! "
            f"You've only saved {checkpoint_saves} checkpoint(s) but run {benchmarks_run} benchmark(s). "
            "Save checkpoint BEFORE and AFTER each benchmark attempt (even if it fails). "
            "This prevents losing progress when errors occur."
        )
    elif benchmarks_run > 0 and checkpoint_saves < benchmarks_run:
        periodic_checkpoint_reminder = (
            f"\n\n📌 CHECKPOINT REMINDER: Save checkpoint after EACH benchmark attempt. "
            f"You've run {benchmarks_run} benchmarks but only saved {checkpoint_saves} checkpoints. "
            "Save more frequently to prevent progress loss!"
        )

    reminder = (
        f"No tool call detected. You must continue benchmarking and call save_tester_output when done.\n"
        f"run_benchmark_code calls so far: {benchmarks_run}/9. "
        f"save_tester_output called: {save_called}. "
        f"Checkpoints saved: {checkpoint_saves}.\n"
        f"{done_msg} "
        f"Most recent tool: {recent_tool or 'none yet'}. "
        f"Use the SAME train/val/test split for every method. {retry_note}"
        f"{checkpoint_reminder}"
        f"{periodic_checkpoint_reminder}"
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
stage3_5_app = builder.compile(checkpointer=memory)


# ===========================
# Stage 3.5 Runner
# ===========================

def run_stage3_5(
    plan_id: str,
    max_rounds: int = STAGE3_5_MAX_ROUNDS,
    debug: bool = True,
    prepared_metadata: Any = None
) -> Dict:
    """Run Stage 3.5 method testing and benchmarking.
    
    Args:
        plan_id: Plan ID from Stage 3 (e.g., 'PLAN-TSK-001')
        max_rounds: Maximum number of agent rounds
        debug: Whether to print debug information
        prepared_metadata: Optional PreparedDataOutput (or dict) from Stage 3B to pass richer context
        
    Returns:
        Final state from the graph execution
    """
    from .config import STAGE3_OUT_DIR, STAGE2_OUT_DIR
    import json
    
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
                excluded_context += "\nBe aware these columns are unavailable. Use alternatives if needed.\n"
        except Exception as e:
            print(f"Warning: Could not load excluded columns: {e}")
    
    system_msg = SystemMessage(content=STAGE3_5_SYSTEM_PROMPT)
    # Surface prepared data + metadata so benchmarking reuses the modeling-ready output.
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
            f"Always load with load_dataframe('{prepared_file_name or prepared_parquet.name}') "
            "and reuse it instead of raw CSVs. Stage 3B already handled formatting and missing values."
        )
        dq_report = prep_dict.get("data_quality_report", {})
        dq_text = json.dumps(dq_report, indent=2)
        if len(dq_text) > 1200:
            dq_text = dq_text[:600] + "\n...[truncated]...\n" + dq_text[-400:]
        transformations = prep_dict.get("transformations_applied", [])
        columns_created = prep_dict.get("columns_created", [])
        prep_context = (
            "\n\nStage 3B context (model-ready):\n"
            f"- Prepared file path: {prepared_file_path or prepared_parquet}\n"
            f"- Original rows → prepared rows: {prep_dict.get('original_row_count')} → {prep_dict.get('prepared_row_count')}\n"
            f"- Columns created: {columns_created}\n"
            f"- Transformations applied: {transformations}\n"
            f"- Data quality & missing-value report:\n{dq_text}\n"
        )
    else:
        if prepared_parquet.exists():
            parquet_hint = (
                f"\n\nPrepared data detected at: {prepared_parquet}\n"
                f"Load with load_dataframe('{prepared_parquet.name}') and reuse it instead of raw CSVs."
            )
        else:
            parquet_hint = (
                "\n\nNo prepared parquet found in Stage 3B output directory. "
                "Proceed with raw data loading."
            )
    human_msg = HumanMessage(
        content=(
            f"Test and benchmark forecasting methods for plan '{plan_id}'.{excluded_context}\n\n"
            f"⚠️ CRITICAL: START BY LOADING CHECKPOINT!\n"
            f"Call load_checkpoint_stage3_5('{plan_id}') FIRST to check for existing progress.\n"
            f"If checkpoint exists, resume from there. If not, start fresh.\n\n"
            f"Follow the ReAct framework strictly:\n"
            f"0. CHECKPOINT: Load checkpoint to resume or start fresh\n"
            f"1. DATA UNDERSTANDING: Load plan, inspect data, identify structure\n"
            f"2. METHOD PROPOSAL: Identify 3 suitable forecasting methods + SAVE CHECKPOINT + VERIFY\n"
            f"3. BENCHMARKING: For each benchmark iteration:\n"
            f"   - SAVE CHECKPOINT (before running benchmark)\n"
            f"   - Run benchmark code\n"
            f"   - SAVE CHECKPOINT (after running benchmark, even if failed)\n"
            f"   - VERIFY checkpoint was saved correctly\n"
            f"4. SELECTION: Choose best method based on averaged metrics\n"
            f"5. SAVE: Call save_tester_output() with complete results\n\n"
            f"🔴 CHECKPOINT DISCIPLINE (MOST CRITICAL!):\n"
            f"- **ALWAYS load_checkpoint_stage3_5() at the start**\n"
            f"- **SAVE CHECKPOINTS FREQUENTLY** - every 2-3 significant actions:\n"
            f"  * After proposing methods and identifying data split\n"
            f"  * BEFORE each benchmark attempt\n"
            f"  * AFTER each benchmark attempt (SUCCESS OR FAILURE)\n"
            f"  * When encountering errors or trying alternative approaches\n"
            f"  * When a method completes all iterations\n"
            f"- **VERIFY every checkpoint save** with verify_checkpoint_stage3_5()\n"
            f"- **If you encounter errors, SAVE CHECKPOINT before trying fixes**\n"
            f"- **USE THE SAME DATA SPLIT for all methods** (from checkpoint)\n\n"
            f"Other reminders:\n"
            f"- Use record_thought() BEFORE each action\n"
            f"- Use record_observation() AFTER each action\n"
            f"- Run 3 iterations per method to verify code execution\n"
            f"- Check coefficient of variation to detect hallucinations\n"
            f"- Be dataset-agnostic (discover column names)\n"
            f"- Assume Stage 3B already handled formatting + missing values; avoid re-cleaning unless corruption is detected\n"
            f"- Use search() if you need examples or guidance\n\n"
            f"Your success metric: save_tester_output() called with valid TesterOutput."
            f"{parquet_hint}{prep_context}"
        )
    )

    state: MessagesState = {"messages": [system_msg, human_msg]}

    # Configure with higher recursion limit for benchmarking tasks  
    config = {
        "configurable": {"thread_id": f"stage3_5-{plan_id}"},
        "recursion_limit": max_rounds + 125  # Increased buffer for 9 benchmarks + selection
    }

    if not debug:
        return stage3_5_app.invoke(state, config=config)

    print("=" * 80)
    print(f"🧪 STAGE 3.5: Method Testing & Benchmarking for {plan_id}")
    print("=" * 80)

    final_state = None
    prev_len = 0
    round_num = 0

    for curr_state in stage3_5_app.stream(
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
    return final_state


# ===========================
# State Node for Master Graph
# ===========================

def stage3_5_node(state: dict) -> dict:
    """Stage 3.5 node for the master pipeline graph.
    
    Args:
        state: Current pipeline state with stage3_plan set
        
    Returns:
        Updated state with tester_output populated
    """
    from .config import STAGE3_5_OUT_DIR
    
    stage3_plan = state.get("stage3_plan")
    if not stage3_plan:
        print("ERROR: No Stage 3 plan available for Stage 3.5")
        state["errors"].append("Stage 3.5: No Stage 3 plan available")
        return state
    
    plan_id = stage3_plan.plan_id
    
    # Check for prepared data from Stage 3B
    prepared_data = state.get("prepared_data")
    if prepared_data:
        print(f"\n✅ Stage 3B prepared data available: {prepared_data.prepared_file_path}")
        print(f"   Rows: {prepared_data.prepared_row_count}, Features: {len(prepared_data.columns_created)}")
    else:
        print(f"\n⚠️  No prepared data from Stage 3B - agent will load raw data")
    
    print(f"\n🧪 Starting Stage 3.5 for: {plan_id}\n")
    
    result = run_stage3_5(plan_id, debug=True, prepared_metadata=prepared_data)
    
    # Check for saved tester output
    tester_files = sorted(STAGE3_5_OUT_DIR.glob(f"tester_{plan_id}*.json"))
    if tester_files:
        latest_file = tester_files[-1]
        print(f"\n✅ SUCCESS! Tester output saved to: {latest_file}")
        tester_data = json.loads(latest_file.read_text())
        state["tester_output"] = TesterOutput.model_validate(tester_data)
        state["completed_stages"].append(3.5)
        state["current_stage"] = 4
    else:
        print("\n⚠️  WARNING: Tester output not saved. Check logs above.")
        state["errors"].append("Stage 3.5: Tester output not saved")
    
    return state


if __name__ == "__main__":
    # Run Stage 3.5 standalone
    import sys
    if len(sys.argv) < 2:
        print("Usage: python stage3_5_agent.py <plan_id>")
        print("Example: python stage3_5_agent.py PLAN-TSK-001")
        sys.exit(1)
    
    plan_id = sys.argv[1].strip()
    run_stage3_5(plan_id)
