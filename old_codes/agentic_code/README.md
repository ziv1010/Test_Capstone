# Unified Agentic AI Pipeline

A comprehensive multi-stage agentic AI system for automated data analytics using LangGraph and LangChain.

## Overview

This pipeline unifies 5 distinct stages into a single cohesive agentic AI workflow:

1. **Stage 1: Dataset Summarization** - Profiles CSV files and generates structured summaries
2. **Stage 2: Task Proposal Generation** - Explores summaries and proposes analytical tasks
3. **Stage 3: Execution Planning** - Creates detailed execution plans for selected tasks
4. **Stage 4: Execution** - Executes plans by generating and running data processing code
5. **Stage 5: Visualization** - Creates comprehensive visualizations and reports

All stages share a unified `PipelineState` that flows through the system, accumulating results at each step.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Master Agent (LangGraph)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Stage 1        Stage 2         Stage 3         Stage 4        Stage 5
│  ┌──────┐     ┌──────┐       ┌──────┐       ┌──────┐       ┌──────┐
│  │ Sum  │────▶│ Task │──────▶│ Plan │──────▶│ Exec │──────▶│ Viz  │
│  │mariz.│     │ Prop.│       │      │       │      │       │      │
│  └──────┘     └──────┘       └──────┘       └──────┘       └──────┘
│     │             │              │              │              │
│     └─────────────┴──────────────┴──────────────┴──────────────┘
│                        Shared Pipeline State
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
agentic_code/
├── config.py           # Configuration and paths
├── models.py           # Pydantic models for all stages
├── utils.py            # Shared utility functions
├── tools.py            # Centralized tool definitions
├── stage1_agent.py     # Dataset summarization
├── stage2_agent.py     # Task proposal generation
├── stage3_agent.py     # Execution planning
├── stage4_agent.py     # Execution
├── stage5_agent.py     # Visualization
├── master_agent.py     # Master orchestration agent
├── run_pipeline.py     # Main entry point script
└── README.md           # This file
```

## Installation

```bash
# Required packages (already installed based on notebooks)
pip install transformers accelerate sentencepiece huggingface_hub \
           langchain-core langchain-openai langgraph pydantic \
           pandas pyarrow numpy scikit-learn matplotlib seaborn
```

## Usage

### Running the Full Pipeline

```bash
# Run all 5 stages
python run_pipeline.py

# Run with specific task ID (auto-selected if not specified)
python run_pipeline.py --task TSK-001
```

### Running Individual Stages

```bash
# Stage 1: Dataset Summarization
python run_pipeline.py --stage 1

# Stage 2: Task Proposal Generation
python run_pipeline.py --stage 2

# Stage 3: Execution Planning (requires task ID)
python run_pipeline.py --stage 3 --task TSK-001

# Stage 4: Execution (requires plan ID)
python run_pipeline.py --stage 4 --task PLAN-TSK-001

# Stage 5: Visualization (requires plan ID)
python run_pipeline.py --stage 5 --task PLAN-TSK-001
```

### Running Partial Pipelines

```bash
# Run stages 1-3
python run_pipeline.py --start 1 --end 3

# Run stages 1-2
python run_pipeline.py --start 1 --end 2
```

### View Configuration

```bash
python run_pipeline.py --config
```

## Stage Details

### Stage 1: Dataset Summarization
- **Input**: CSV files in `data/`
- **Output**: Structured summaries in `summaries/`
- **Tech**: Direct LLM inference (no LangGraph)
- **Purpose**: Profile datasets and extract metadata

### Stage 2: Task Proposal Generation
- **Input**: Dataset summaries from Stage 1
- **Output**: Task proposals in `stage2_out/task_proposals.json`
- **Tech**: LangGraph exploration loop
- **Purpose**: Generate analytical task proposals

### Stage 3: Execution Planning
- **Input**: Selected task proposal
- **Output**: Detailed plan in `stage3_out/PLAN-{task_id}.json`
- **Tech**: LangGraph with tools
- **Purpose**: Create executable data pipeline plans

### Stage 4: Execution
- **Input**: Stage 3 plan
- **Output**: Execution results in `stage4_out/`
- **Tech**: LangGraph with code execution
- **Purpose**: Execute plans and generate results

### Stage 5: Visualization
- **Input**: Stage 4 results
- **Output**: Visualizations and reports in `stage5_out/`
- **Tech**: LangGraph with visualization tools
- **Purpose**: Create insights and visualizations

## LLM Configuration

The system uses two LLM endpoints:

- **Primary**: Meta-Llama-3.1-8B-Instruct on port 8000 (Stages 1-2)
- **Secondary**: Qwen2.5-32B-Instruct on port 8001 (Stages 3-5)

Configure in `config.py`.

## Output Directories

```
/scratch/ziv_baretto/llmserve/
├── data/                  # Input CSV files
├── summaries/             # Stage 1 output
├── stage2_out/            # Stage 2 output
├── stage3_out/            # Stage 3 output
├── stage4_out/            # Stage 4 output
│   └── code_workspace/    # Working directory for code execution
└── stage5_out/            # Stage 5 output
    └── viz_workspace/     # Working directory for visualizations
```

## Key Features

✅ **Fully Autonomous** - Each stage operates independently with LLM-driven decision making

✅ **Dataset Agnostic** - Works with any tabular data without domain-specific assumptions

✅ **Modular Design** - Run individual stages or the full pipeline

✅ **State Management** - Unified state flows through all stages

✅ **Error Handling** - Tracks errors and continues when possible

✅ **Extensible** - Easy to add new stages or modify existing ones

## Development

### Running Individual Stage Agents

Each stage can be run standalone:

```bash
# Stage 1
python -m agentic_code.stage1_agent

# Stage 2
python -m agentic_code.stage2_agent

# Stage 3
python -m agentic_code.stage3_agent TSK-001

# Stage 4
python -m agentic_code.stage4_agent PLAN-TSK-001

# Stage 5
python -m agentic_code.stage5_agent PLAN-TSK-001
```

### Importing in Python

```python
from agentic_code.master_agent import run_full_pipeline, master_app
from agentic_code.stage1_agent import run_stage1
from agentic_code.config import print_config

# Run full pipeline
result = run_full_pipeline(selected_task_id="TSK-001")

# Access results
print(f"Completed stages: {result['completed_stages']}")
print(f"Task proposals: {len(result['task_proposals'])}")
```

## Troubleshooting

### LLM Connection Issues
- Ensure vLLM servers are running on ports 8000 and 8001
- Check `config.py` for correct endpoints

### Missing Dependencies
```bash
pip install -r requirements.txt  # (if created)
```

### Stage Failures
- Check output directories for partial results
- Review error messages in console output
- Errors are tracked in `state['errors']`

## Credits

Consolidated from original Jupyter notebooks:
- `stage.ipynb` - Stage 1
- `stage_2_graph.ipynb` - Stage 2
- `stage3_final.ipynb` - Stages 3, 4, 5

---

**Note**: This is an autonomous agentic AI system. The agents make independent decisions about data processing, feature engineering, and visualization. Review outputs before using in production.
