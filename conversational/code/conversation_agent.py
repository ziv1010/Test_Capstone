"""
Conversation Agent: User Interaction Handler

This agent handles user queries, determines intent, and orchestrates
the appropriate pipeline stages to fulfill requests.
"""

import json
from typing import Dict, Any, Optional, Annotated, List
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
    CONVERSATION_LLM_CONFIG, CONVERSATION_STATE_DIR,
    DataPassingManager, logger
)
from code.models import ConversationContext, ConversationMessage, PipelineState
from tools.conversation_tools import CONVERSATION_TOOLS


# ============================================================================
# STATE DEFINITION
# ============================================================================

class ConversationState(BaseModel):
    """State for the conversation agent."""
    messages: Annotated[list, add_messages] = []
    session_id: str = ""
    user_intent: str = ""
    current_task_id: Optional[str] = None
    pipeline_action: Optional[str] = None  # run_stage, get_info, custom_query
    pipeline_stages: List[str] = []
    response_ready: bool = False
    iteration: int = 0


# ============================================================================
# SYSTEM PROMPT
# ============================================================================

CONVERSATION_SYSTEM_PROMPT = """You are a helpful AI assistant for a data analysis and forecasting pipeline.

## Your Role
You help users:
1. Understand their available data
2. Get summaries of analyzed datasets
3. See proposed analytical tasks
4. Create custom forecasting queries
5. Run the analysis pipeline
6. View results and visualizations

## Available Actions
Based on user requests, you can:
- **Show data**: Get summaries of available datasets
- **Show tasks**: Get proposed analytical tasks
- **Check status**: See what pipeline stages have been completed
- **Evaluate query**: Assess if a user's forecasting idea is feasible
- **Create task**: Create a custom task from user's query
- **Run pipeline**: Execute stages for a specific task

## Available Tools
- get_available_data: List datasets in the data directory
- get_summaries: Get Stage 1 dataset summaries
- get_task_proposals: Get Stage 2 task proposals
- check_pipeline_status: See which stages are complete
- evaluate_user_query: Check if a forecasting query is feasible
- create_custom_task_from_query: Create a task from user's query
- get_execution_results: Get results from completed executions
- get_visualizations: Get visualization reports

## Conversation Guidelines
1. Be helpful and conversational
2. Explain what's available and possible
3. Guide users toward successful analyses
4. When unclear, ask clarifying questions
5. Summarize technical details in plain language

## Response Format
- Keep responses concise but informative
- Use bullet points for lists
- Highlight key findings
- Suggest next steps when appropriate

## Intent Recognition
Recognize these user intents:
- "show me the data" / "what data is available" → get_available_data
- "show summaries" / "what's in the data" → get_summaries
- "what tasks can I do" / "show tasks" → get_task_proposals
- "what's the status" / "what's been done" → check_pipeline_status
- "can I predict X" / "is forecasting possible" → evaluate_user_query
- "run task X" / "execute task X" → trigger pipeline execution
- "show results" / "what were the results" → get_execution_results
- "show visualizations" / "show plots" → get_visualizations

## Pipeline Orchestration
When a user wants to run a task:
1. Confirm which task they want to run
2. Explain what stages will be executed (3 → 3B → 3.5A → 3.5B → 4 → 5)
3. Set pipeline_action = "run_pipeline" and include the task_id
4. The master orchestrator will handle actual execution

IMPORTANT: You don't execute the pipeline directly - you set the intent and let the orchestrator handle it.
"""


# ============================================================================
# AGENT LOGIC
# ============================================================================

def create_conversation_agent():
    """Create the conversation agent graph."""

    llm = ChatOpenAI(**CONVERSATION_LLM_CONFIG)
    llm_with_tools = llm.bind_tools(CONVERSATION_TOOLS, parallel_tool_calls=False)

    def agent_node(state: ConversationState) -> Dict[str, Any]:
        """Main conversation agent node."""
        messages = state.messages

        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=CONVERSATION_SYSTEM_PROMPT)] + list(messages)

        if state.iteration >= 20:  # Conversation limit
            return {
                "messages": [AIMessage(content="I've reached my conversation limit. Please start a new session.")],
                "response_ready": True
            }

        response = llm_with_tools.invoke(messages)

        return {
            "messages": [response],
            "iteration": state.iteration + 1
        }

    def should_continue(state: ConversationState) -> str:
        """Determine if we should continue or end."""
        if state.response_ready:
            return "end"

        if state.messages:
            last_message = state.messages[-1]
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "tools"

        return "end"

    builder = StateGraph(ConversationState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(CONVERSATION_TOOLS))
    builder.set_entry_point("agent")
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END}
    )
    builder.add_edge("tools", "agent")

    return builder.compile(checkpointer=MemorySaver())


# ============================================================================
# CONVERSATION HANDLER
# ============================================================================

class ConversationHandler:
    """
    Handles user conversations and manages session state.
    """

    def __init__(self, session_id: str = None):
        self.session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.graph = create_conversation_agent()
        self.context = ConversationContext(session_id=self.session_id)
        self.state = None

    def process_message(self, user_message: str) -> Dict[str, Any]:
        """
        Process a user message and return response with any pipeline actions.

        Returns:
            Dict with:
            - response: The assistant's response text
            - action: Pipeline action to take (if any)
            - task_id: Task ID for pipeline action (if any)
            - stages: Stages to run (if any)
        """
        # Add user message to context
        self.context.add_message("user", user_message)

        # Create initial state
        initial_state = ConversationState(
            messages=[HumanMessage(content=user_message)],
            session_id=self.session_id
        )

        # Run the conversation agent
        config = {"configurable": {"thread_id": self.session_id}}

        try:
            final_state = self.graph.invoke(initial_state, config)

            # Extract response
            response = ""
            for msg in reversed(final_state.get("messages", [])):
                if isinstance(msg, AIMessage):
                    response = msg.content
                    break

            # Add assistant response to context
            self.context.add_message("assistant", response)

            # Check for pipeline action
            action = self._detect_pipeline_action(user_message, response)

            return {
                "response": response,
                "action": action.get("action"),
                "task_id": action.get("task_id"),
                "stages": action.get("stages", [])
            }

        except Exception as e:
            logger.error(f"Conversation error: {e}")
            return {
                "response": f"I encountered an error: {e}. Please try again.",
                "action": None,
                "task_id": None,
                "stages": []
            }

    def _detect_pipeline_action(self, user_message: str, response: str) -> Dict[str, Any]:
        """
        Detect if the conversation indicates a pipeline action.
        """
        user_lower = user_message.lower()
        response_lower = response.lower()

        # Check for run/execute intent
        if any(word in user_lower for word in ["run", "execute", "start", "do"]):
            # Try to extract task ID with flexible parsing
            import re
            
            # Try to match TSK-digits pattern
            task_match = re.search(r'tsk[- ]?(\d+)', user_lower)
            if task_match:
                task_num = task_match.group(1)
                # Normalize to TSK-XXX format (zero-padded to 3 digits if <=999)
                if len(task_num) <= 3:
                    task_id = f"TSK-{int(task_num):03d}"
                else:
                    task_id = f"TSK-{task_num}"
                
                # Validate task exists by checking task_proposals.json
                task_id = self._validate_task_id(task_id, task_num)
                
                if task_id:
                    return {
                        "action": "run_pipeline",
                        "task_id": task_id,
                        "stages": ["stage3", "stage3b", "stage3_5a", "stage3_5b", "stage4", "stage5"]
                    }

        # Check for analyze/summarize intent (stages 1-2)
        if any(word in user_lower for word in ["analyze", "summarize", "profile"]):
            if "data" in user_lower or "dataset" in user_lower:
                return {
                    "action": "run_stages",
                    "task_id": None,
                    "stages": ["stage1", "stage2"]
                }

        return {"action": None, "task_id": None, "stages": []}
    
    def _validate_task_id(self, task_id: str, task_num: str) -> Optional[str]:
        """
        Validate that a task ID exists in task proposals.
        
        Args:
            task_id: Normalized task ID (e.g., "TSK-001")
            task_num: Raw number from user input (e.g., "1", "001", "9586")
            
        Returns:
            Valid task ID or None if not found
        """
        from code.config import STAGE2_OUT_DIR
        import json
        
        proposals_path = STAGE2_OUT_DIR / "task_proposals.json"
        if not proposals_path.exists():
            logger.warning("Task proposals not found. Unable to validate task ID.")
            return task_id  # Return as-is if we can't validate
        
        try:
            # Load task proposals directly with JSON
            with open(proposals_path, 'r') as f:
                proposals_data = json.load(f)
            
            available_tasks = proposals_data.get("data", {}).get("proposals", [])
            
            # Build map of task IDs
            task_ids = {task["id"] for task in available_tasks}
            
            # Direct match
            if task_id in task_ids:
                logger.info(f"Task {task_id} validated")
                return task_id
            
            # Try alternate formats
            alternates = [
                f"TSK-{task_num}",  # Raw number
                f"TSK-{int(task_num)}",  # Unpadded
            ]
            
            for alt in alternates:
                if alt in task_ids:
                    logger.info(f"Task {task_num} matched to {alt}")
                    return alt
            
            # No match found
            logger.warning(f"Task {task_id} not found. Available: {sorted(task_ids)}")
            return None
            
        except Exception as e:
            logger.error(f"Error validating task ID: {e}")
            return task_id  # Return as-is on error

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return [
            {"role": m.role, "content": m.content}
            for m in self.context.messages
        ]

    def save_session(self):
        """Save session state to disk."""
        try:
            session_path = CONVERSATION_STATE_DIR / f"{self.session_id}.json"
            DataPassingManager.save_artifact(
                data=self.context.model_dump(),
                output_dir=CONVERSATION_STATE_DIR,
                filename=f"{self.session_id}.json"
            )
        except Exception as e:
            logger.error(f"Failed to save session: {e}")

    @classmethod
    def load_session(cls, session_id: str) -> "ConversationHandler":
        """Load an existing session."""
        handler = cls(session_id=session_id)

        session_path = CONVERSATION_STATE_DIR / f"{session_id}.json"
        if session_path.exists():
            try:
                data = DataPassingManager.load_artifact(session_path)
                handler.context = ConversationContext(**data)
            except Exception as e:
                logger.error(f"Failed to load session: {e}")

        return handler


# ============================================================================
# QUICK RESPONSE FUNCTIONS
# ============================================================================

def get_quick_summary() -> str:
    """Get a quick summary of available data and tasks."""
    from tools.conversation_tools import get_available_data, get_task_proposals, check_pipeline_status

    parts = []

    # Get data summary
    data_result = get_available_data.invoke({})
    parts.append("**Available Data:**")
    parts.append(data_result)
    parts.append("")

    # Get pipeline status
    status_result = check_pipeline_status.invoke({})
    parts.append("**Pipeline Status:**")
    parts.append(status_result)
    parts.append("")

    # Get task proposals if available
    tasks_result = get_task_proposals.invoke({})
    if "No task proposals" not in tasks_result:
        parts.append("**Available Tasks:**")
        parts.append(tasks_result)

    return "\n".join(parts)


def evaluate_forecasting_query(query: str) -> Dict[str, Any]:
    """
    Evaluate if a user's forecasting query is feasible.

    Returns feasibility assessment and suggested approach.
    """
    from tools.conversation_tools import evaluate_user_query

    result = evaluate_user_query.invoke({"query": query})

    # Parse the result to extract key info
    is_feasible = "FEASIBLE" in result.upper() and "INFEASIBLE" not in result.upper()

    return {
        "feasible": is_feasible,
        "analysis": result,
        "recommendation": "proceed" if is_feasible else "review_data"
    }


# ============================================================================
# MAIN CONVERSATION LOOP
# ============================================================================

def run_conversation_loop():
    """
    Run an interactive conversation loop.
    """
    print("\n" + "="*60)
    print("Welcome to the Conversational AI Pipeline!")
    print("="*60)
    print("\nI can help you:")
    print("  - Explore available datasets")
    print("  - View proposed analytical tasks")
    print("  - Create custom forecasting queries")
    print("  - Run the full analysis pipeline")
    print("\nType 'quit' or 'exit' to end the session.")
    print("="*60 + "\n")

    handler = ConversationHandler()

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nAssistant: Goodbye! Your session has been saved.")
                handler.save_session()
                break

            # Process the message
            result = handler.process_message(user_input)

            print(f"\nAssistant: {result['response']}")

            # If there's a pipeline action, notify the user
            if result['action']:
                print(f"\n[System: Pipeline action detected - {result['action']}]")
                if result['task_id']:
                    print(f"[Task ID: {result['task_id']}]")
                if result['stages']:
                    print(f"[Stages: {', '.join(result['stages'])}]")

        except KeyboardInterrupt:
            print("\n\nAssistant: Session interrupted. Saving...")
            handler.save_session()
            break
        except Exception as e:
            print(f"\nAssistant: I encountered an error: {e}")
            continue


if __name__ == "__main__":
    run_conversation_loop()
