#!/usr/bin/env python3
"""
UI Server for Conversational AI Pipeline

Provides a web interface for:
- Chatting with the AI assistant
- Viewing pipeline progress in real-time
- Inspecting stage outputs and model thoughts
"""

import os
import sys
import json
import logging
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from code.config import (
    SUMMARIES_DIR, STAGE2_OUT_DIR, STAGE3_OUT_DIR, STAGE3B_OUT_DIR,
    STAGE3_5A_OUT_DIR, STAGE3_5B_OUT_DIR, STAGE4_OUT_DIR, STAGE5_OUT_DIR,
    logger as pipeline_logger
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ui_server")

# ============================================================================
# GLOBAL STATE
# ============================================================================

class PipelineTracker:
    """Tracks pipeline execution state for the UI."""
    
    def __init__(self):
        self.current_task_id: Optional[str] = None
        self.is_running: bool = False
        self.current_stage: Optional[str] = None
        self.stage_status: Dict[str, str] = {}
        self.errors: List[str] = []
        self.lock = threading.Lock()
    
    def start_pipeline(self, task_id: str):
        with self.lock:
            self.current_task_id = task_id
            self.is_running = True
            self.current_stage = "stage1"
            self.stage_status = {}
            self.errors = []
    
    def update_stage(self, stage: str, status: str):
        with self.lock:
            self.stage_status[stage] = status
            if status == "running":
                self.current_stage = stage
    
    def finish_pipeline(self, success: bool = True):
        with self.lock:
            self.is_running = False
            if not success:
                self.errors.append("Pipeline execution failed")
    
    def get_state(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "task_id": self.current_task_id,
                "is_running": self.is_running,
                "current_stage": self.current_stage,
                "stage_status": self.stage_status.copy(),
                "errors": self.errors.copy()
            }

tracker = PipelineTracker()

# ============================================================================
# API MODELS
# ============================================================================

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    pipeline_started: bool = False
    task_id: Optional[str] = None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_stage_output_path(stage: str, task_id: str = None) -> Optional[Path]:
    """Get the output file path for a stage."""
    if not task_id:
        task_id = tracker.current_task_id
    
    if not task_id:
        return None
    
    # Ensure task_id has PLAN- prefix for stages that need it
    plan_id = f"PLAN-{task_id}" if not task_id.startswith("PLAN-") else task_id
    
    paths = {
        "stage1": SUMMARIES_DIR,  # Multiple files
        "stage2": STAGE2_OUT_DIR / "task_proposals.json",
        "stage3": STAGE3_OUT_DIR / f"{plan_id}.json",
        "stage3b": STAGE3B_OUT_DIR / f"prepared_{plan_id}.parquet",
        "stage3_5a": STAGE3_5A_OUT_DIR / f"method_proposal_{plan_id}.json",
        "stage3_5b": STAGE3_5B_OUT_DIR / f"tester_{plan_id}.json",
        "stage4": STAGE4_OUT_DIR / f"execution_result_{plan_id}.json",
        "stage5": STAGE5_OUT_DIR / f"visualization_report_{plan_id}.json",
    }
    
    return paths.get(stage)

def load_stage_output(stage: str, task_id: str = None) -> Optional[Dict[str, Any]]:
    """Load the output for a specific stage."""
    path = get_stage_output_path(stage, task_id)
    
    if path is None:
        return None
    
    # Stage 1 has multiple summary files
    if stage == "stage1":
        summaries = []
        if path.exists():
            for f in path.glob("*.summary.json"):
                try:
                    with open(f, 'r') as fp:
                        data = json.load(fp)
                        # Handle wrapped format
                        if "data" in data:
                            summaries.append(data["data"])
                        else:
                            summaries.append(data)
                except Exception as e:
                    logger.error(f"Failed to load {f}: {e}")
        if summaries:
            return {"summaries": summaries, "count": len(summaries)}
        return None
    
    # Other stages have single JSON files
    if isinstance(path, Path) and path.exists():
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                # Handle wrapped format
                if "data" in data and "_meta" in data:
                    return data["data"]
                return data
        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
    
    return None

def infer_stage_status(stage: str, task_id: str = None) -> str:
    """Infer the status of a stage based on file existence."""
    output = load_stage_output(stage, task_id)
    
    if output is not None:
        return "completed"
    
    # Check if this is the current running stage
    state = tracker.get_state()
    if state["is_running"] and state["current_stage"] == stage:
        return "running"
    
    return "pending"

def get_all_stages_status(task_id: str = None) -> Dict[str, Dict[str, Any]]:
    """Get status for all stages."""
    stages = ["stage1", "stage2", "stage3", "stage3b", "stage3_5a", "stage3_5b", "stage4", "stage5"]
    result = {}
    
    for stage in stages:
        status = infer_stage_status(stage, task_id)
        result[stage] = {
            "stage_name": stage,
            "status": status,
            "has_output": status == "completed"
        }
    
    return result

# ============================================================================
# PIPELINE EXECUTION
# ============================================================================

def run_pipeline_background(task_id: str):
    """Run the pipeline in a background thread."""
    from code.master_orchestrator import run_forecasting_pipeline
    
    logger.info(f"Starting background pipeline for task {task_id}")
    tracker.start_pipeline(task_id)
    
    try:
        # Run the pipeline
        state = run_forecasting_pipeline(task_id)
        
        # Update tracker with final state
        if state:
            for stage_name, stage_state in state.stages.items():
                tracker.update_stage(stage_name, stage_state.status.value)
        
        tracker.finish_pipeline(success=True)
        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        tracker.finish_pipeline(success=False)

# ============================================================================
# FASTAPI APP
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting UI Server...")
    yield
    logger.info("Shutting down UI Server...")

app = FastAPI(title="Conversational AI Pipeline UI", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="ui/static"), name="static")

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Serve the main UI."""
    return FileResponse("ui/static/index.html")

@app.post("/api/chat")
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    """Handle chat messages."""
    from code.master_orchestrator import ConversationalOrchestrator
    
    orchestrator = ConversationalOrchestrator()
    result = orchestrator.conversation.process_message(request.message)
    
    response_text = result.get("response", "I couldn't process that request.")
    pipeline_started = False
    task_id = None

    # Check if pipeline needs to run
    if result.get("action") == "run_pipeline" and result.get("task_id"):
        task_id = result["task_id"]
        if task_id:
            pipeline_started = True
            response_text += f"\n\n🚀 Starting pipeline execution for {task_id}..."
            background_tasks.add_task(run_pipeline_background, task_id)

    return ChatResponse(
        response=response_text,
        pipeline_started=pipeline_started,
        task_id=task_id
    )

@app.get("/api/state")
async def get_state():
    """Get the current pipeline state."""
    state = tracker.get_state()
    stages = get_all_stages_status(state.get("task_id"))
    
    return {
        "task_id": state["task_id"],
        "is_running": state["is_running"],
        "current_stage": state["current_stage"],
        "stages": stages,
        "errors": state["errors"]
    }

@app.get("/api/stage/{stage_name}")
async def get_stage_details(stage_name: str):
    """Get detailed output for a specific stage."""
    valid_stages = ["stage1", "stage2", "stage3", "stage3b", "stage3_5a", "stage3_5b", "stage4", "stage5"]
    
    if stage_name not in valid_stages:
        raise HTTPException(status_code=400, detail=f"Invalid stage: {stage_name}")
    
    task_id = tracker.current_task_id
    status = infer_stage_status(stage_name, task_id)
    output = load_stage_output(stage_name, task_id)
    
    return {
        "stage_name": stage_name,
        "status": status,
        "output": output,
        "has_output": output is not None
    }

@app.get("/api/tasks")
async def get_available_tasks():
    """Get list of available tasks from stage 2 output."""
    proposals_path = STAGE2_OUT_DIR / "task_proposals.json"
    
    if not proposals_path.exists():
        return {"tasks": [], "message": "No tasks available. Run stage 1 and 2 first."}
    
    try:
        with open(proposals_path, 'r') as f:
            data = json.load(f)
            
        # Handle wrapped format
        if "data" in data:
            data = data["data"]
        
        proposals = data.get("proposals", [])
        tasks = []
        for p in proposals:
            tasks.append({
                "id": p.get("id"),
                "title": p.get("title"),
                "category": p.get("category"),
                "target_column": p.get("target_column"),
                "feasibility_score": p.get("feasibility_score")
            })
        
        return {"tasks": tasks}
        
    except Exception as e:
        logger.error(f"Failed to load tasks: {e}")
        return {"tasks": [], "error": str(e)}

@app.get("/api/visualizations/{task_id}")
async def get_visualizations(task_id: str):
    """Get visualization files for a task."""
    plan_id = f"PLAN-{task_id}" if not task_id.startswith("PLAN-") else task_id
    viz_dir = STAGE5_OUT_DIR
    
    # Look for visualization report
    report_path = viz_dir / f"visualization_report_{plan_id}.json"
    
    if not report_path.exists():
        return {"visualizations": [], "message": "No visualizations available for this task."}
    
    try:
        with open(report_path, 'r') as f:
            data = json.load(f)
        
        if "data" in data:
            data = data["data"]
        
        return {
            "visualizations": data.get("visualizations", []),
            "insights": data.get("insights", []),
            "summary": data.get("summary", ""),
            "task_answer": data.get("task_answer", "")
        }
        
    except Exception as e:
        logger.error(f"Failed to load visualizations: {e}")
        return {"visualizations": [], "error": str(e)}

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
