"""
Unified Agentic AI Pipeline

A multi-stage autonomous data analytics system using LangGraph and LangChain.
"""

__version__ = "1.0.0"

from .master_agent import (
    run_full_pipeline,
    run_partial_pipeline,
    master_app,
    PipelineState,
)

from .stage1_agent import run_stage1
from .stage2_agent import run_stage2
from .stage3_agent import run_stage3
from .stage4_agent import run_stage4
from .stage5_agent import run_stage5

from .config import print_config

__all__ = [
    # Main pipeline functions
    "run_full_pipeline",
    "run_partial_pipeline",
    "master_app",
    "PipelineState",
    
    # Individual stage runners
    "run_stage1",
    "run_stage2",
    "run_stage3",
    "run_stage4",
    "run_stage5",
    
    # Utilities
    "print_config",
]
