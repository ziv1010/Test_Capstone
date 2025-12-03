"""
Failsafe / Thinking Agent

Provides a ReAct-style helper that any stage can call when stuck.
Uses a lightweight loop with the shared search/inspection tools, then returns
concise next steps and saves them for traceability.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Dict, List, Optional

from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from .config import SECONDARY_LLM_CONFIG, FAILSAFE_OUT_DIR
from .models import FailsafeRecommendation
from .tools import FAILSAFE_TOOLS


# ===========================
# LLM Setup
# ===========================

llm = ChatOpenAI(**SECONDARY_LLM_CONFIG)
llm_with_tools = llm.bind_tools(FAILSAFE_TOOLS, parallel_tool_calls=False)


# ===========================
# System Prompt
# ===========================

FAILSAFE_SYSTEM_PROMPT = """You are the Failsafe/Thinking Agent.

Your job: unblock another stage by diagnosing the error and proposing concrete recovery steps.

Tools available (choose based on need; nothing is hardcoded):
- failsafe_python(code): run any Python for ad-hoc checks (can read data/output/code).
- search(query, within): search files; set within to project/output/code/data/all.
- inspect_data_file / list_data_files: quick schema checks.

Workflow (compact ReAct):
- Think briefly.
- Call tools if they add evidence (<=2 calls preferred).
- Choose the search scope yourself; default is project-wide.
- End with short next steps.

Output format (STRICT JSON, no prose):
{
  "stage": "<stage name>",
  "error": "<error message>",
  "analysis": "<root cause hypothesis>",
  "next_steps": ["step 1", "step 2"],
  "tool_evidence": "key findings from tools or 'none'"
}
"""


# ===========================
# LangGraph
# ===========================

def agent_node(state: MessagesState) -> Dict[str, List[BaseMessage]]:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


tool_node = ToolNode(FAILSAFE_TOOLS)


def should_continue(state: MessagesState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


builder = StateGraph(MessagesState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)
builder.set_entry_point("agent")
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
builder.add_edge("tools", "agent")

memory = MemorySaver()
failsafe_app = builder.compile(checkpointer=memory)


# ===========================
# Runner
# ===========================

def run_failsafe(
    stage: str,
    error: str,
    context: Optional[str] = None,
    max_rounds: int = 4,
    debug: bool = False,
) -> FailsafeRecommendation:
    """Invoke the failsafe agent to diagnose and suggest recovery steps.
    
    Args:
        stage: Name of the stage requesting help
        error: Error message or symptom
        context: Optional recent log snippet or notes
        max_rounds: Maximum LLM turns
        debug: Whether to print reasoning and tool calls
        
    Returns:
        FailsafeRecommendation
    """
    system_msg = SystemMessage(content=FAILSAFE_SYSTEM_PROMPT)
    human_msg = HumanMessage(
        content=(
            f"Stage: {stage}\n"
            f"Error: {error}\n"
            f"Context: {context or 'n/a'}\n"
            "Return STRICT JSON as specified."
        )
    )

    state: MessagesState = {"messages": [system_msg, human_msg]}

    final_state = None
    prev_len = 0
    round_num = 0

    for curr_state in failsafe_app.stream(
        state,
        config={
            "configurable": {"thread_id": f"failsafe-{stage}"},
            "recursion_limit": max_rounds * 2,
        },
        stream_mode="values",
    ):
        msgs = curr_state["messages"]
        new_msgs = msgs[prev_len:]

        for m in new_msgs:
            msg_type = m.__class__.__name__
            if debug and "AI" in msg_type and m.content:
                round_num += 1
                print(f"\n[FAILSAFE AI round {round_num}]")
                print(m.content[:800] + ("..." if len(m.content) > 800 else ""))
            if debug and hasattr(m, "tool_calls") and m.tool_calls:
                print("\n[FAILSAFE tool calls]")
                for tc in m.tool_calls:
                    print(f"- {tc.get('name')} args={tc.get('args')}")

        prev_len = len(msgs)
        final_state = curr_state
        if round_num >= max_rounds:
            break

    if not final_state or not final_state["messages"]:
        raise RuntimeError("Failsafe agent did not produce a response.")

    last_ai = [m for m in final_state["messages"] if m.__class__.__name__.startswith("AI")]
    if not last_ai:
        raise RuntimeError("Failsafe agent finished without AI content.")
    raw = last_ai[-1].content
    raw_text = raw if isinstance(raw, str) else str(raw)

    # Extract JSON (grab from first '{' to last '}')
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Failsafe agent did not return JSON: {raw_text}")

    obj = json.loads(raw_text[start : end + 1])
    rec = FailsafeRecommendation.model_validate(obj)

    # Persist for traceability
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = FAILSAFE_OUT_DIR / f"failsafe_{stage}_{timestamp}.json"
    out_path.write_text(rec.model_dump_json(indent=2))

    if debug:
        print(f"\n[FAILSAFE saved] {out_path}")

    return rec


__all__ = ["run_failsafe"]
