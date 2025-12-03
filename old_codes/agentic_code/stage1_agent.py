"""
Stage 1: Dataset Summarization Agent

Enhanced with:
- Auto tool calling using LangGraph
- Checkpoint management for summaries
- Rolling history management with checkpoint loading
- Semi-deterministic calculation verification
- Uses SECONDARY_LLM_CONFIG (Qwen/Qwen3-32B)
- Ensures all summaries are generated and output

Profiles CSV files and generates structured summaries using an LLM with agentic tool calling.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool

from .config import DATA_DIR, SUMMARIES_DIR, SECONDARY_LLM_CONFIG, STAGE1_SAMPLE_ROWS
from .models import DatasetSummary
from .utils import profile_csv


# ===========================
# Checkpoint Management
# ===========================

def save_summary_checkpoint(summary: DatasetSummary, dataset_name: str) -> Path:
    """Save checkpoint for individual dataset summary.

    Args:
        summary: Generated DatasetSummary object
        dataset_name: Name of the dataset

    Returns:
        Path to saved checkpoint
    """
    checkpoint_data = {
        "dataset_name": dataset_name,
        "timestamp": datetime.now().isoformat(),
        "summary": summary.model_dump()
    }

    checkpoint_path = SUMMARIES_DIR / f"checkpoint_{dataset_name}.json"
    checkpoint_path.write_text(json.dumps(checkpoint_data, indent=2))
    return checkpoint_path


def save_thinking_checkpoint(thinking_content: str, dataset_name: str, phase: str) -> Path:
    """Save thinking/reasoning checkpoint.

    Args:
        thinking_content: The reasoning/thinking content
        dataset_name: Name of the dataset being processed
        phase: Phase identifier (e.g., 'analysis', 'validation')

    Returns:
        Path to saved checkpoint
    """
    checkpoint_data = {
        "dataset_name": dataset_name,
        "phase": phase,
        "timestamp": datetime.now().isoformat(),
        "thinking": thinking_content
    }

    checkpoint_path = SUMMARIES_DIR / f"checkpoint_thinking_{dataset_name}_{phase}.json"
    checkpoint_path.write_text(json.dumps(checkpoint_data, indent=2))
    return checkpoint_path


def load_summary_checkpoint(dataset_name: str) -> Optional[Dict]:
    """Load existing summary checkpoint if exists.

    Args:
        dataset_name: Name of the dataset

    Returns:
        Checkpoint data or None if no checkpoint exists
    """
    checkpoint_path = SUMMARIES_DIR / f"checkpoint_{dataset_name}.json"
    if checkpoint_path.exists():
        print(f"📂 Loading checkpoint: {checkpoint_path.name}")
        return json.loads(checkpoint_path.read_text())
    return None


def load_all_summary_checkpoints() -> Dict[str, DatasetSummary]:
    """Load all existing summary checkpoints.

    Returns:
        Dictionary mapping dataset names to DatasetSummary objects
    """
    summaries = {}
    for checkpoint_file in SUMMARIES_DIR.glob("checkpoint_*.json"):
        if "thinking" in checkpoint_file.name:
            continue
        try:
            checkpoint_data = json.loads(checkpoint_file.read_text())
            dataset_name = checkpoint_data["dataset_name"]
            summary = DatasetSummary.model_validate(checkpoint_data["summary"])
            summaries[dataset_name] = summary
            print(f"📂 Loaded checkpoint: {checkpoint_file.name}")
        except Exception as e:
            print(f"⚠️  Failed to load {checkpoint_file.name}: {e}")
    return summaries


def clear_stage1_checkpoints():
    """Clear all Stage 1 checkpoints to start fresh."""
    for checkpoint_file in SUMMARIES_DIR.glob("checkpoint_*.json"):
        checkpoint_file.unlink()
        print(f"🗑️  Cleared: {checkpoint_file.name}")


# ===========================
# Stage 1 Specific Tools
# ===========================

@tool
def analyze_profile() -> str:
    """Analyze the current dataset profile and provide insights.

    This tool helps understand the dataset structure, identify key columns,
    detect data quality issues, and suggest primary keys.

    No arguments needed - uses the current profile being processed.

    Returns:
        Analysis insights as text
    """
    try:
        profile = _CURRENT_PROFILE

        insights = []
        insights.append("=== PROFILE ANALYSIS ===\n")

        # Basic info
        insights.append(f"Dataset: {profile.get('filename', 'Unknown')}")
        insights.append(f"Rows: {profile.get('num_rows', 0)}")
        insights.append(f"Columns: {len(profile.get('columns', []))}\n")

        # Column analysis
        columns = profile.get('columns', [])
        numeric_cols = []
        categorical_cols = []
        high_null_cols = []
        potential_keys = []

        for col in columns:
            col_name = col['name']
            null_frac = col.get('null_fraction', 0)
            unique_frac = col.get('unique_fraction', 0)
            dtype = col.get('physical_dtype', '')

            # Categorize columns
            if 'int' in dtype.lower() or 'float' in dtype.lower():
                numeric_cols.append(col_name)
            else:
                categorical_cols.append(col_name)

            # Check for high nulls
            if null_frac > 0.5:
                high_null_cols.append(f"{col_name} ({null_frac*100:.1f}% null)")

            # Check for potential keys
            if unique_frac > 0.95 and null_frac < 0.01:
                potential_keys.append(f"{col_name} ({unique_frac*100:.1f}% unique)")

        insights.append(f"Numeric columns: {len(numeric_cols)}")
        insights.append(f"Categorical columns: {len(categorical_cols)}\n")

        if high_null_cols:
            insights.append(f"⚠️  High null columns:")
            for col in high_null_cols[:5]:
                insights.append(f"  - {col}")
            insights.append("")

        if potential_keys:
            insights.append(f"🔑 Potential key columns:")
            for col in potential_keys:
                insights.append(f"  - {col}")
            insights.append("")
        else:
            insights.append("⚠️  No obvious single-column keys found\n")

        insights.append("✅ Analysis complete")

        return "\n".join(insights)

    except Exception as e:
        return f"[ERROR] Failed to analyze profile: {e}"


@tool
def verify_calculations() -> str:
    """Verify that calculated statistics in the current profile are accurate.

    This ensures semi-deterministic behavior by validating:
    - Null fractions are correctly calculated
    - Unique fractions are correctly calculated
    - Data types are properly identified

    No arguments needed - uses the current profile being processed.

    Returns:
        Verification results
    """
    try:
        profile = _CURRENT_PROFILE

        results = []
        results.append("=== CALCULATION VERIFICATION ===\n")

        columns = profile.get('columns', [])
        num_rows = profile.get('num_rows', 0)

        all_valid = True

        for col in columns:
            col_name = col['name']
            null_frac = col.get('null_fraction', 0)
            unique_frac = col.get('unique_fraction', 0)

            # Verify null_fraction is in valid range
            if not (0 <= null_frac <= 1):
                results.append(f"❌ {col_name}: Invalid null_fraction {null_frac}")
                all_valid = False

            # Verify unique_fraction is in valid range
            if not (0 <= unique_frac <= 1):
                results.append(f"❌ {col_name}: Invalid unique_fraction {unique_frac}")
                all_valid = False

            # Verify consistency
            if unique_frac > 0 and null_frac >= 1:
                results.append(f"⚠️  {col_name}: Has unique values but all nulls?")
                all_valid = False

        if all_valid:
            results.append(f"✅ All calculations verified for {len(columns)} columns")
            results.append(f"✅ Data quality metrics are accurate")
        else:
            results.append(f"\n⚠️  Some calculations need review")

        return "\n".join(results)

    except Exception as e:
        return f"[ERROR] Failed to verify calculations: {e}"


@tool
def create_dataset_summary(
    dataset_name: str,
    path: str,
    candidate_primary_keys: List[List[str]],
    notes: str
) -> str:
    """Create and validate a dataset summary.

    This tool automatically builds the columns list from the loaded profile data.
    You only need to provide the dataset_name, path, candidate_primary_keys, and notes.

    The tool will:
    - Extract all columns from the current profile
    - Enhance each column with proper logical_type based on physical_dtype
    - Set nullable based on null_fraction
    - Set is_potential_key based on uniqueness criteria

    Args:
        dataset_name: Name of the dataset
        path: Path to the dataset file
        candidate_primary_keys: List of potential primary key combinations (e.g., [["HS Code"], ["HS Code", "Description"]])
        notes: Comprehensive notes about data quality, observations, and suggestions

    Returns:
        Confirmation message with validation status
    """
    try:
        # Get profile data
        profile = _CURRENT_PROFILE
        approx_n_rows = profile.get('num_rows', 0)

        # Build enhanced columns list from profile
        columns = []
        for col in profile.get('columns', []):
            # Determine logical type based on physical_dtype
            physical_dtype = col.get('physical_dtype', 'unknown')
            if physical_dtype == 'object':
                # For object types, check if it's categorical or text
                unique_frac = col.get('unique_fraction', 0)
                logical_type = 'categorical' if unique_frac < 0.5 else 'text'
            elif 'int' in physical_dtype.lower():
                logical_type = 'integer'
            elif 'float' in physical_dtype.lower():
                logical_type = 'float'
            elif 'datetime' in physical_dtype.lower():
                logical_type = 'datetime'
            elif 'bool' in physical_dtype.lower():
                logical_type = 'boolean'
            else:
                logical_type = 'numeric'

            # Generate description based on column name
            col_name = col.get('name', '')
            description = f"{col_name.replace('-', ' ').replace('_', ' ')}"

            # Determine if nullable
            null_fraction = col.get('null_fraction', 0)
            nullable = null_fraction > 0

            # Determine if potential key
            unique_fraction = col.get('unique_fraction', 0)
            is_potential_key = unique_fraction > 0.9 and null_fraction < 0.05

            # Build column dict
            column_dict = {
                "name": col_name,
                "physical_dtype": physical_dtype,
                "logical_type": logical_type,
                "description": description,
                "nullable": nullable,
                "null_fraction": null_fraction,
                "unique_fraction": unique_fraction,
                "examples": col.get('examples', []),
                "is_potential_key": is_potential_key
            }
            columns.append(column_dict)

        # Build summary dict
        summary_dict = {
            "dataset_name": dataset_name,
            "path": path,
            "approx_n_rows": approx_n_rows,
            "columns": columns,
            "candidate_primary_keys": candidate_primary_keys,
            "notes": notes
        }

        # Validate with Pydantic
        summary = DatasetSummary.model_validate(summary_dict)

        # Save checkpoint
        save_summary_checkpoint(summary, dataset_name)

        # Save final summary file
        out_path = SUMMARIES_DIR / f"{dataset_name}.summary.json"
        summary_json = summary.model_dump_json(indent=2)
        out_path.write_text(summary_json)

        # Print the complete JSON summary
        print("\n" + "=" * 80)
        print(f"📄 COMPLETED SUMMARY: {dataset_name}")
        print("=" * 80)
        print(summary_json)
        print("=" * 80 + "\n")

        return f"✅ Summary created and saved: {out_path.name}\n\nColumns: {len(columns)}\nPrimary keys: {len(candidate_primary_keys)}\n\nFull JSON summary has been printed above."

    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        return f"❌ [ERROR] Failed to create summary: {e}\n\nDetails:\n{error_detail}"


# Global storage for current profile being processed
_CURRENT_PROFILE: Dict[str, Any] = {}

# Stage 1 tool list
STAGE1_TOOLS = [analyze_profile, verify_calculations, create_dataset_summary]


# ===========================
# LLM Setup with SECONDARY_LLM_CONFIG
# ===========================

llm = ChatOpenAI(**SECONDARY_LLM_CONFIG)
llm_with_tools = llm.bind_tools(STAGE1_TOOLS, parallel_tool_calls=False)


# ===========================
# System Prompt
# ===========================

system_prompt = """
You are Agent 1 in a multi-stage, agentic data analytics pipeline.

Your job is to analyze CSV dataset profiles and generate structured summaries with rich metadata.

You have access to THREE TOOLS:

1. analyze_profile()
   - No arguments needed - automatically uses the current loaded profile
   - Analyzes the dataset profile to understand structure and identify patterns
   - Returns insights about columns, data quality, and potential keys

2. verify_calculations()
   - No arguments needed - automatically uses the current loaded profile
   - Verifies that statistical calculations are accurate
   - Ensures semi-deterministic behavior and data quality

3. create_dataset_summary(dataset_name, path, candidate_primary_keys, notes)
   - Automatically builds enhanced columns from the loaded profile
   - Creates the final DatasetSummary object
   - Validates and saves the summary
   - Args needed: dataset_name, path, candidate_primary_keys (list of lists), notes

═══════════════════════════════════════════════════════════════
YOUR TASK - MANDATORY WORKFLOW
═══════════════════════════════════════════════════════════════

You MUST complete ALL THREE steps for each dataset. Do NOT stop until all steps are done.

STEP 1: ANALYZE
- Call analyze_profile() to understand the dataset structure
- Wait for the analysis results

STEP 2: VERIFY
- Call verify_calculations() to ensure accuracy
- Wait for the verification results

STEP 3: CREATE SUMMARY
- Based on the analysis and verification, determine:

  A) CANDIDATE PRIMARY KEYS:
     * Look for single columns with unique_fraction > 0.95 and null_fraction < 0.01
     * Consider composite keys (combinations like [["Crop", "Season"]] or [["HS Code"]])
     * If no good keys found, use empty list []
     * Format as list of lists: [["column1"], ["column1", "column2"]]

  B) COMPREHENSIVE NOTES:
     * Overall data quality assessment
     * Key observations or quirks found
     * Any warnings about missing data (especially columns with >50% nulls)
     * Suggestions for Stage 2 analysis (trends, predictions, correlations)
     * Mention any potential key columns

- Call create_dataset_summary() with ONLY 4 arguments:
  * dataset_name: The dataset name (e.g., "Export-of-Rice-Varieties-to-Bangladesh,-2018-19-to-2024-25")
  * path: The file path
  * candidate_primary_keys: List of key combinations (e.g., [["HS Code"], ["Description"]])
  * notes: Your comprehensive notes as a string

  NOTE: The tool will automatically extract and enhance ALL columns from the loaded profile.
        You do NOT need to manually build the columns list!

CRITICAL: You must call all three tools. After create_dataset_summary() returns success, your job is complete.

═══════════════════════════════════════════════════════════════
COLUMN METADATA REQUIREMENTS
═══════════════════════════════════════════════════════════════

For each column, you must provide (in this exact order):
- name: Original column name (from profile)
- physical_dtype: Original data type (from profile)
- logical_type: Refined logical type - choose from: categorical, numeric, integer, float, text, datetime, boolean, unknown
- description: Clear, concise description of what this column represents
- nullable: true if null_fraction > 0, false otherwise
- null_fraction: Copy from original profile
- unique_fraction: Copy from original profile
- examples: Copy from original profile (as list of strings)
- is_potential_key: true if this could be part of a primary key (unique_fraction > 0.9 and null_fraction < 0.05)

EXAMPLE COLUMN FORMAT:
{
  "name": "Crop",
  "physical_dtype": "object",
  "logical_type": "categorical",
  "description": "Type of crop being measured.",
  "nullable": false,
  "null_fraction": 0.0,
  "unique_fraction": 0.31666666666666665,
  "examples": ["Rice", "Wheat", "Maize", "Barley", "Jowar"],
  "is_potential_key": true
}

═══════════════════════════════════════════════════════════════
LOGICAL TYPE GUIDELINES
═══════════════════════════════════════════════════════════════

Choose logical_type carefully:
- numeric: General numeric data (could be int or float)
- integer: Specifically integer values (counts, IDs, years)
- float: Specifically floating-point values (measurements, ratios)
- categorical: Limited set of discrete values (categories, labels, codes)
- text: Free-form text data (descriptions, names)
- datetime: Dates, times, or timestamps
- boolean: True/False or Yes/No values
- unknown: When type cannot be determined

═══════════════════════════════════════════════════════════════
NOTES FIELD
═══════════════════════════════════════════════════════════════

Always include a short 'notes' field summarizing:
- Overall data quality
- Key observations or quirks
- Potential issues or warnings
- Suggestions for Stage 2 (task proposal)

═══════════════════════════════════════════════════════════════
IMPORTANT REMINDERS
═══════════════════════════════════════════════════════════════

- ALWAYS call analyze_profile() first to understand the data
- ALWAYS call verify_calculations() to ensure accuracy
- ALWAYS call create_dataset_summary() to save the final result
- Ensure descriptions are clear and based on actual column content
- Be thorough in identifying potential primary keys
- Include meaningful notes to help downstream stages
"""


# ===========================
# Rolling History Management
# ===========================

HISTORY_WINDOW = 20  # Keep last 20 messages in active history


def manage_rolling_history(messages: List[BaseMessage], dataset_name: str) -> List[BaseMessage]:
    """Manage rolling history window to prevent context overflow.

    When history exceeds the window, save checkpoint and trim to recent messages.

    Args:
        messages: Current message history
        dataset_name: Current dataset being processed

    Returns:
        Trimmed message list
    """
    if len(messages) > HISTORY_WINDOW:
        # Save thinking checkpoint before trimming
        last_ai_msg = None
        for m in reversed(messages):
            if isinstance(m, AIMessage) and m.content:
                last_ai_msg = m.content
                break

        if last_ai_msg:
            save_thinking_checkpoint(last_ai_msg, dataset_name, "rolling_history")
            print(f"💾 Rolling history checkpoint saved for {dataset_name}")

        # Keep system message + recent messages
        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        recent_msgs = messages[-HISTORY_WINDOW:]

        # Ensure we don't duplicate system messages
        if recent_msgs and isinstance(recent_msgs[0], SystemMessage):
            return recent_msgs
        else:
            return system_msgs + recent_msgs

    return messages


# ===========================
# LangGraph State + Nodes
# ===========================

def agent_node(state: MessagesState) -> Dict[str, List[BaseMessage]]:
    """Single LLM step with auto tool calling."""
    response = llm_with_tools.invoke(state["messages"])

    # Save thinking checkpoint if response has substantial reasoning
    if hasattr(response, 'content') and response.content and len(response.content) > 50:
        # Extract dataset name from recent messages
        dataset_name = "unknown"
        for m in reversed(state["messages"]):
            if isinstance(m, HumanMessage) and "Dataset:" in m.content:
                lines = m.content.split("\n")
                for line in lines:
                    if line.startswith("Dataset:"):
                        dataset_name = line.replace("Dataset:", "").strip()
                        break
                break

        save_thinking_checkpoint(response.content, dataset_name, "agent_reasoning")

    return {"messages": [response]}


tool_node = ToolNode(STAGE1_TOOLS)


def should_continue(state: MessagesState) -> str:
    """Route based on tool calls and completion status."""
    last = state["messages"][-1]

    # Check if we just got a tool response indicating summary creation success
    if hasattr(last, 'content') and isinstance(last.content, str):
        if "✅ Summary created and saved:" in last.content:
            print("🎉 Summary creation detected - workflow complete!")
            return END

    # Continue if there are more tool calls to make
    if hasattr(last, 'tool_calls') and last.tool_calls:
        return "tools"

    return END


# ===========================
# Build LangGraph
# ===========================

builder = StateGraph(MessagesState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)
builder.set_entry_point("agent")
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
builder.add_edge("tools", "agent")

# Add memory checkpointing
memory = MemorySaver()
stage1_app = builder.compile(checkpointer=memory)


# ===========================
# Core Processing Functions
# ===========================

def process_single_dataset(
    csv_path: Path,
    sample_rows: int = STAGE1_SAMPLE_ROWS,
    resume: bool = True,
    debug: bool = True
) -> DatasetSummary:
    """Process a single dataset with LangGraph agent.

    Args:
        csv_path: Path to CSV file
        sample_rows: Number of rows to sample
        resume: Whether to resume from checkpoint
        debug: Whether to print debug information

    Returns:
        DatasetSummary object
    """
    dataset_name = csv_path.stem

    print("\n" + "=" * 80)
    print(f"📊 Processing Dataset: {dataset_name}")
    print("=" * 80)

    # Check for existing checkpoint
    if resume:
        checkpoint = load_summary_checkpoint(dataset_name)
        if checkpoint:
            print(f"✅ Found checkpoint for {dataset_name}, skipping...")
            return DatasetSummary.model_validate(checkpoint["summary"])

    # Generate profile
    print(f"🔍 Profiling dataset ({sample_rows} rows)...")
    profile = profile_csv(csv_path, sample_rows=sample_rows)

    # Store profile globally for tools to access
    global _CURRENT_PROFILE
    _CURRENT_PROFILE = profile

    # Create readable summary for display
    num_cols = len(profile.get('columns', []))
    num_rows = profile.get('num_rows', 0)

    # Get column names and first few examples
    col_names = [col['name'] for col in profile.get('columns', [])]
    col_summary = []
    for col in profile.get('columns', [])[:15]:  # Show first 15 columns
        dtype = col.get('physical_dtype', 'unknown')
        null_pct = col.get('null_fraction', 0) * 100
        col_summary.append(f"  - {col['name']} ({dtype}, {null_pct:.1f}% null)")

    profile_summary = f"""Dataset: {dataset_name}
Path: {csv_path}
Rows: {num_rows}
Columns: {num_cols}

Column Details:
{chr(10).join(col_summary)}
{"..." if num_cols > 15 else ""}

The full profile data has been loaded and is available to your tools.
"""

    # Build initial messages with simpler prompt
    messages: List[BaseMessage] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"""{profile_summary}

Please process this dataset by:
1. Calling analyze_profile() to understand the structure
2. Calling verify_calculations() to ensure data quality
3. Calling create_dataset_summary() with complete enhanced metadata

All tools will automatically access the loaded profile data.""")
    ]

    initial_state: MessagesState = {"messages": messages}

    # Run agent
    if not debug:
        final_state = stage1_app.invoke(
            initial_state,
            config={"configurable": {"thread_id": f"stage1_{dataset_name}"}}
        )
    else:
        print("\n" + "=" * 80)
        print(f"🤖 Agent Processing: {dataset_name}")
        print("=" * 80)

        final_state = None
        prev_len = 0
        round_num = 0

        for curr_state in stage1_app.stream(
            initial_state,
            stream_mode="values",
            config={"recursion_limit": 50, "configurable": {"thread_id": f"stage1_{dataset_name}"}},
        ):
            msgs = curr_state["messages"]
            new_msgs = msgs[prev_len:]

            for m in new_msgs:
                msg_type = m.__class__.__name__

                if "System" in msg_type:
                    continue  # Skip system message in output

                elif "Human" in msg_type:
                    print("\n" + "─" * 80)
                    print("👤 [USER]")
                    print("─" * 80)
                    # Truncate long profile data
                    content = m.content
                    if len(content) > 500:
                        print(content[:250] + "\n...[truncated]...\n" + content[-250:])
                    else:
                        print(content)

                elif "AI" in msg_type:
                    round_num += 1
                    print("\n" + "═" * 80)
                    print(f"🤖 [AGENT - Round {round_num}]")
                    print("═" * 80)
                    if m.content:
                        print("\n💭 Reasoning:")
                        print(m.content)

                    if hasattr(m, 'tool_calls') and m.tool_calls:
                        print("\n🔧 Tool Calls:")
                        for tc in m.tool_calls:
                            name = tc.get("name", "UNKNOWN")
                            args = tc.get("args", {})
                            print(f"\n  📌 {name}")
                            for k, v in args.items():
                                if isinstance(v, str) and len(v) > 200:
                                    print(f"     {k}: [truncated, length={len(v)}]")
                                elif isinstance(v, list) and len(v) > 5:
                                    print(f"     {k}: [list with {len(v)} items]")
                                else:
                                    print(f"     {k}: {v}")

                elif "Tool" in msg_type:
                    print("\n🔍 Tool Result:")
                    result = m.content
                    if len(result) > 300:
                        print(result[:150] + "\n...[truncated]...\n" + result[-150:])
                    else:
                        print(result)

            # Manage rolling history
            if len(msgs) > HISTORY_WINDOW:
                msgs = manage_rolling_history(msgs, dataset_name)
                curr_state["messages"] = msgs
                print(f"\n♻️  History trimmed to {len(msgs)} messages")

            prev_len = len(msgs)
            final_state = curr_state

        print("\n" + "=" * 80)
        print(f"✅ Agent Complete - {round_num} rounds")
        print("=" * 80)

    # Load the saved summary - try final file first, then checkpoint
    final_path = SUMMARIES_DIR / f"{dataset_name}.summary.json"
    if final_path.exists():
        print(f"✅ Loading completed summary from: {final_path.name}")
        summary_data = json.loads(final_path.read_text())
        return DatasetSummary.model_validate(summary_data)

    # Fallback to checkpoint
    checkpoint = load_summary_checkpoint(dataset_name)
    if checkpoint:
        print(f"✅ Loading summary from checkpoint")
        return DatasetSummary.model_validate(checkpoint["summary"])

    # If neither exists, something went wrong
    raise ValueError(f"No summary file or checkpoint found for {dataset_name} after processing")


# ===========================
# Main Stage 1 Runner
# ===========================

def run_stage1(
    data_dir: Path = DATA_DIR,
    out_dir: Path = SUMMARIES_DIR,
    pattern: str = "*.csv",
    sample_rows: int = STAGE1_SAMPLE_ROWS,
    resume: bool = True,
    debug: bool = True,
) -> List[DatasetSummary]:
    """Run Stage 1: Dataset summarization with LangGraph agent.

    Enhanced with:
    - Auto tool calling via LangGraph
    - Checkpoint management for resuming
    - Rolling history to prevent context overflow
    - Semi-deterministic calculation verification
    - Uses SECONDARY_LLM_CONFIG

    Args:
        data_dir: Directory containing CSV files
        out_dir: Directory to save summaries
        pattern: Glob pattern for files to process
        sample_rows: Number of rows to sample for profiling
        resume: Whether to resume from checkpoints
        debug: Whether to print debug information

    Returns:
        List of DatasetSummary objects
    """
    print("\n" + "=" * 80)
    print("🚀 STAGE 1: Dataset Summarization (Enhanced)")
    print("=" * 80)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {out_dir}")
    print(f"Resume mode: {resume}")
    print(f"Debug mode: {debug}")
    print(f"LLM: {SECONDARY_LLM_CONFIG['model']}")
    print("=" * 80)

    # Ensure output directory exists
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find CSV files
    paths = sorted(data_dir.glob(pattern))
    print(f"\n📁 Found {len(paths)} CSV files in {data_dir}")

    if not paths:
        print("⚠️  No CSV files found!")
        return []

    # Load existing checkpoints if resuming
    existing_summaries = {}
    if resume:
        existing_summaries = load_all_summary_checkpoints()
        if existing_summaries:
            print(f"📂 Loaded {len(existing_summaries)} existing checkpoints")

    summaries = []

    for idx, path in enumerate(paths, 1):
        print(f"\n{'=' * 80}")
        print(f"Dataset {idx}/{len(paths)}: {path.name}")
        print(f"{'=' * 80}")

        try:
            summary = process_single_dataset(
                path,
                sample_rows=sample_rows,
                resume=resume,
                debug=debug
            )
            summaries.append(summary)
            print(f"✅ Summary generated: {path.stem}.summary.json")

        except Exception as e:
            print(f"❌ Failed to process {path.name}: {e}")
            import traceback
            traceback.print_exc()

    # Final summary
    print("\n" + "=" * 80)
    print("📊 STAGE 1 COMPLETE")
    print("=" * 80)
    print(f"Total datasets processed: {len(summaries)}/{len(paths)}")
    print(f"Summary files saved to: {out_dir}")

    # List all generated summaries
    print("\nGenerated summaries:")
    for summary in summaries:
        print(f"  ✓ {summary.dataset_name}: {len(summary.columns)} columns")

    print("=" * 80)

    return summaries


# ===========================
# State Node for Master Graph
# ===========================

def stage1_node(state: dict) -> dict:
    """Stage 1 node for the master pipeline graph.

    Args:
        state: Current pipeline state

    Returns:
        Updated state with dataset_summaries populated
    """
    print("\n" + "=" * 80)
    print("STAGE 1: Dataset Summarization")
    print("=" * 80)

    summaries = run_stage1()

    state["dataset_summaries"] = summaries
    state["completed_stages"].append(1)
    state["current_stage"] = 2

    print(f"\n✅ Stage 1 complete: Generated {len(summaries)} summaries")

    return state


if __name__ == "__main__":
    # Run Stage 1 standalone
    run_stage1()