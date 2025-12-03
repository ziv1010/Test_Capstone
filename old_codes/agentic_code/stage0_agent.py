"""
Stage 0: Conversational Agent (Query Interpretation & Routing)

Serves as the entry point for the conversational interface.
Interprets natural language queries and routes them to appropriate pipeline stages.
"""

from __future__ import annotations

import json
from typing import Dict, List, Any, Optional, Literal
from datetime import datetime

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from .config import get_llm_config, STAGE3_MAX_ROUNDS
from .tools import STAGE0_TOOLS
from .models import PipelineState, ConversationState

# ===========================
# LLM & Tools
# ===========================

from langchain_openai import ChatOpenAI

llm_config = get_llm_config()
llm = ChatOpenAI(**llm_config)
tools = STAGE0_TOOLS
llm_with_tools = llm.bind_tools(tools)


# ===========================
# System Prompt
# ===========================

SYSTEM_PROMPT = """You are a Conversational Data Analysis Agent. Your goal is to help users analyze their data through natural language.

You have access to a powerful 5-stage data pipeline:
- Stage 1: Dataset Summarization (Profiling)
- Stage 2: Task Proposal (Suggesting what can be predicted)
- Stage 3: Planning (Creating execution plans)
- Stage 4: Execution (Running code)
- Stage 5: Visualization (Creating charts)

CRITICAL: You must TAKE ACTION by calling tools, not just talk about what you could do!

CORE RESPONSIBILITIES:

1. INTERPRET: Understand if the user wants to explore data, make a prediction, or visualize results.
2. ROUTE: Use `trigger_pipeline_stages` to run the necessary parts of the pipeline.
3. EXPLAIN: Always explain what you are doing and summarize results in plain English.
4. TAKE ACTION: When a user makes a selection or request, IMMEDIATELY call the appropriate tool rather than repeating options.

HANDLING USER SELECTIONS (CRITICAL):

When the user responds with:
- "1" or "explore data" → Call `query_data_capabilities()`
- "2" or selects an existing task (e.g., "TSK-001", "first one") → Call `trigger_pipeline_stages(3, 4, task_id='TSK-XXX')`
- "3" or "new task" or "more proposals" or "generate new" → IMMEDIATELY call `trigger_pipeline_stages(2, 2, user_query='<user request>')` to regenerate proposals (NOTE: use 2,2 not 1,2 to force fresh execution)
- Bare number matching a task (e.g., "1" when 3 tasks were shown) → Execute that task
- "More suggestions" or "more tasks" → IMMEDIATELY call `trigger_pipeline_stages(2, 2)` to refresh proposals
- "What tasks?" or "List tasks" or "Show proposals" → Call `query_data_capabilities()`

DO NOT repeat the same options if the user has already made a selection. ACT on their choice!

INTERACTION FLOWS:

A. NEW PREDICTION REQUESTS ("Predict rice exports for next 5 years") OR LISTING TASKS ("What tasks are available?")
   1. Call `query_data_capabilities()` to check existing proposals
   2. If proposals exist, LIST THEM SPECIFICALLY (e.g., "- [TSK-001] Title")
   3. If user wants new proposals OR says "3" OR says "more", IMMEDIATELY call `trigger_pipeline_stages(2, 2, user_query='<user request>')` (use 2,2 to force regeneration)
   4. Once proposals are generated, present them and ask user to select one by ID

B. TASK SELECTION ("2" or "Select existing task")
   1. Call `query_data_capabilities()` to get the list of available tasks
   2. PRESENT THE LIST OF TASKS with IDs and titles (e.g., "- [TSK-001] Forecast Rice Production")
   3. Ask the user to specify which ID they want (e.g., "Which task ID would you like to run?")
   4. If user specifies an ID (e.g., "TSK-001" or "first one"), call `trigger_pipeline_stages(3, 4, task_id='TSK-001')`

C. DATA EXPLORATION
   1. Call `query_data_capabilities()` to see what's available
   2. If no data is summarized, call `trigger_pipeline_stages(1, 1)`
   3. Present findings and suggest next steps

D. CUSTOM ANALYSIS
   1. Use `execute_dynamic_analysis` to write and run custom Python code
   2. Explain the findings clearly

CRITICAL RULES:
- Never repeat the same menu/options twice in a row if user has made a selection
- When presenting Option 2, YOU MUST LIST THE ACTUAL TASKS (IDs and Titles) found via `query_data_capabilities()`
- When user says "3", "new", "more", etc., CALL trigger_pipeline_stages immediately
- When `trigger_pipeline_stages` returns a list of proposals, YOU MUST USE THOSE EXACT TITLES. Do not summarize or invent new ones.
- When user selects a task ID, RUN it immediately with trigger_pipeline_stages
- Be action-oriented: EXECUTE tools rather than just describing what you could do
"""

# ===========================
# Graph Nodes
# ===========================

def agent_node(state: Dict):
    """Core agent node that processes messages and decides actions."""
    messages = state.get("messages", [])
    
    # Ensure system message is present
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    
    # Invoke LLM
    response = llm_with_tools.invoke(messages)
    
    return {"messages": [response]}


def should_continue(state: Dict) -> Literal["tools", END]:
    """Determine if we should continue to tools or end."""
    messages = state["messages"]
    last_message = messages[-1]
    
    if last_message.tool_calls:
        return "tools"
    return END


# ===========================
# Build Graph
# ===========================

# Define state for the graph (standard LangGraph message state)
class AgentState(Dict):
    messages: List[Any]

builder = StateGraph(AgentState)

# Add nodes
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))

# Set entry point
builder.set_entry_point("agent")

# Add edges
builder.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)
builder.add_edge("tools", "agent")

# Compile
memory = MemorySaver()
stage0_app = builder.compile(checkpointer=memory)


# ===========================
# Runner
# ===========================

def run_conversational_turn(user_query: str, thread_id: str = "default") -> str:
    """Run a single turn of the conversation.
    
    Args:
        user_query: The user's natural language question
        thread_id: Session ID for memory
        
    Returns:
        The agent's final text response
    """
    config = {"configurable": {"thread_id": thread_id}}
    
    # Add user message
    initial_state = {
        "messages": [HumanMessage(content=user_query)]
    }
    
    # Run graph
    final_state = stage0_app.invoke(initial_state, config=config)
    
    # Extract final response
    last_message = final_state["messages"][-1]
    return last_message.content


if __name__ == "__main__":
    # Simple CLI test
    print("🤖 Conversational Agent (Stage 0)")
    print("Type 'exit' to quit.")
    
    while True:
        try:
            query = input("\nUser: ")
            if query.lower() in ["exit", "quit"]:
                break
                
            response = run_conversational_turn(query)
            print(f"\nAgent: {response}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
