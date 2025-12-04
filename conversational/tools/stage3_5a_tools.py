"""
Stage 3.5A Tools: Method Proposal

Tools for proposing forecasting methods based on the data and task.
"""

import json
from pathlib import Path
from typing import Optional
from langchain_core.tools import tool
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from code.config import (
    DATA_DIR, STAGE3_OUT_DIR, STAGE3B_OUT_DIR, STAGE3_5A_OUT_DIR,
    DataPassingManager, logger, DEBUG
)
from code.utils import load_dataframe, execute_python_sandbox, safe_json_dumps


# State tracking for ReAct framework
_stage3_5a_thoughts = []
_stage3_5a_observations = []


@tool
def load_plan_and_data(plan_id: str = None) -> str:
    """
    Load the execution plan and prepared data for method proposal.

    Args:
        plan_id: Plan ID. If not provided, loads most recent.

    Returns:
        Summary of plan and data characteristics
    """
    try:
        # Find plan
        if plan_id:
            plan_path = STAGE3_OUT_DIR / f"{plan_id}.json"
        else:
            plans = list(STAGE3_OUT_DIR.glob("PLAN-*.json"))
            if not plans:
                return "No execution plans found."
            plan_path = max(plans, key=lambda p: p.stat().st_mtime)
            plan_id = plan_path.stem

        plan = DataPassingManager.load_artifact(plan_path)

        result = [
            f"=== Plan & Data Summary: {plan_id} ===",
            f"Task: {plan.get('selected_task_id')}",
            f"Category: {plan.get('task_category')}",
            f"Target: {plan.get('target_column')}",
            f"Date Column: {plan.get('date_column')}",
            "",
        ]

        # Try to load prepared data
        prepared_path = STAGE3B_OUT_DIR / f"prepared_{plan_id}.parquet"
        logger.debug(f"Looking for prepared data at: {prepared_path}")

        if prepared_path.exists():
            df = pd.read_parquet(prepared_path)
            result.append(f"Prepared Data: {df.shape[0]} rows x {df.shape[1]} columns")
            result.append(f"Columns: {list(df.columns)}")

            # Date range
            date_col = plan.get('date_column')
            if date_col and date_col in df.columns:
                dates = pd.to_datetime(df[date_col], errors='coerce')
                result.append(f"Date Range: {dates.min()} to {dates.max()}")

            # Target stats
            target_col = plan.get('target_column')
            if target_col and target_col in df.columns:
                target = df[target_col]
                result.append(f"Target Stats: min={target.min():.2f}, max={target.max():.2f}, mean={target.mean():.2f}")
        else:
            result.append("Prepared data not found - will need to use raw files")

        result.append("\n\nFull plan JSON:")
        result.append(json.dumps(plan, indent=2, default=str))

        return "\n".join(result)

    except Exception as e:
        logger.error(f"Error loading plan/data: {e}")
        return f"Error loading plan/data: {e}"


@tool
def analyze_time_series(plan_id: str = None) -> str:
    """
    Analyze the time series characteristics of the data.

    Determines frequency, seasonality, trend, and stationarity.

    Args:
        plan_id: Plan ID to analyze

    Returns:
        Time series analysis report
    """
    try:
        if not plan_id:
            plans = list(STAGE3_OUT_DIR.glob("PLAN-*.json"))
            if plans:
                plan_id = max(plans, key=lambda p: p.stat().st_mtime).stem

        # Load plan
        plan = DataPassingManager.load_artifact(STAGE3_OUT_DIR / f"{plan_id}.json")
        date_col = plan.get('date_column')
        target_col = plan.get('target_column')

        # Load data
        prepared_path = STAGE3B_OUT_DIR / f"prepared_{plan_id}.parquet"
        if not prepared_path.exists():
            return "Prepared data not found. Run Stage 3B first."

        df = pd.read_parquet(prepared_path)

        result = ["=== Time Series Analysis ===\n"]

        if not date_col or date_col not in df.columns:
            result.append("WARNING: No date column found or specified")
            return "\n".join(result)

        if not target_col or target_col not in df.columns:
            result.append("WARNING: No target column found or specified")
            return "\n".join(result)

        # Parse dates and sort
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col]).sort_values(date_col)

        # Frequency analysis
        diffs = df[date_col].diff().dropna()
        median_diff = diffs.median()

        result.append(f"Date Column: {date_col}")
        result.append(f"Date Range: {df[date_col].min()} to {df[date_col].max()}")
        result.append(f"Number of observations: {len(df)}")

        if median_diff.days == 1:
            freq = "daily"
        elif median_diff.days == 7:
            freq = "weekly"
        elif 28 <= median_diff.days <= 31:
            freq = "monthly"
        elif 365 <= median_diff.days <= 366:
            freq = "yearly"
        else:
            freq = f"irregular ({median_diff.days} days median)"

        result.append(f"Detected Frequency: {freq}")

        # Target analysis
        target = df[target_col]
        result.append(f"\nTarget Column: {target_col}")
        result.append(f"  Mean: {target.mean():.2f}")
        result.append(f"  Std: {target.std():.2f}")
        result.append(f"  Min: {target.min():.2f}")
        result.append(f"  Max: {target.max():.2f}")

        # Simple trend check
        first_half_mean = target.iloc[:len(target)//2].mean()
        second_half_mean = target.iloc[len(target)//2:].mean()
        if second_half_mean > first_half_mean * 1.1:
            trend = "upward"
        elif second_half_mean < first_half_mean * 0.9:
            trend = "downward"
        else:
            trend = "stable"

        result.append(f"  Trend: {trend}")

        # Seasonality hint (if enough data)
        if len(df) > 24 and freq in ["daily", "monthly"]:
            result.append("  Seasonality: Possible (needs further analysis)")
        else:
            result.append("  Seasonality: Insufficient data to determine")

        # Method recommendations
        result.append("\n--- Method Recommendations ---")
        result.append("Based on data characteristics:")

        if len(df) < 50:
            result.append("  - Simple methods recommended (moving average, naive)")
            result.append("  - Avoid complex models due to limited data")
        elif len(df) < 200:
            result.append("  - Statistical methods suitable (ARIMA, exponential smoothing)")
            result.append("  - Simple ML models may work (linear regression)")
        else:
            result.append("  - All method types suitable")
            result.append("  - Consider complex ML models (random forest, gradient boosting)")

        return "\n".join(result)

    except Exception as e:
        return f"Error analyzing time series: {e}"


@tool
def record_thought_3_5a(thought: str, next_action: str) -> str:
    """
    Record a thought before taking an action (ReAct framework).

    Args:
        thought: Current reasoning
        next_action: Planned next step

    Returns:
        Confirmation
    """
    global _stage3_5a_thoughts
    entry = {"thought": thought, "next_action": next_action, "step": len(_stage3_5a_thoughts) + 1}
    _stage3_5a_thoughts.append(entry)
    return f"Thought #{entry['step']} recorded."


@tool
def record_observation_3_5a(what_happened: str, insight: str, next_step: str) -> str:
    """
    Record an observation after an action (ReAct framework).

    Args:
        what_happened: Result of action
        insight: Key learning
        next_step: What to do next

    Returns:
        Confirmation
    """
    global _stage3_5a_observations
    entry = {"what_happened": what_happened, "insight": insight, "next_step": next_step, "step": len(_stage3_5a_observations) + 1}
    _stage3_5a_observations.append(entry)
    return f"Observation #{entry['step']} recorded."


@tool
def python_sandbox_stage3_5a(code: str, description: str = "") -> str:
    """
    Execute Python code for method proposal analysis.

    Available: pd, np, DATA_DIR, STAGE3B_OUT_DIR, load_dataframe()

    Args:
        code: Python code to execute
        description: What the code does

    Returns:
        Execution output
    """
    additional = {
        'DATA_DIR': DATA_DIR,
        'STAGE3B_OUT_DIR': STAGE3B_OUT_DIR,
        'STAGE3_OUT_DIR': STAGE3_OUT_DIR,
    }
    return execute_python_sandbox(code, additional, description)


@tool
def get_method_templates() -> str:
    """
    Get implementation PATTERNS for algorithms (not specific algorithms).

    Returns code structure templates that the agent should adapt based on
    its data analysis. The agent chooses which algorithms to use.
    """
    templates = {
        "code_patterns": {
            "baseline_pattern": {
                "description": "Pattern for simple baseline methods",
                "template": """
def predict_{METHOD_NAME}(train_df, test_df, target_col, **params):
    import pandas as pd
    import numpy as np
    
    # TODO: Implement your chosen baseline approach
    # Examples: mean prediction, last value, moving average, most frequent class
    
    baseline_prediction = ...  # YOUR BASELINE LOGIC HERE
    
    predictions = pd.DataFrame({
        'predicted': [baseline_prediction] * len(test_df)
    }, index=test_df.index)
    
    return predictions
""",
                "notes": "Baseline should be simple with minimal assumptions. Good for comparison."
            },
            "statistical_pattern": {
                "description": "Pattern for statistical/traditional methods",
                "template": """
def predict_{METHOD_NAME}(train_df, test_df, target_col, feature_cols=None, **params):
    import pandas as pd
    import numpy as np
    # Import your chosen library (sklearn, statsmodels, etc.)
    
    # Prepare data
    if feature_cols:
        X_train = train_df[feature_cols].fillna(0)
        X_test = test_df[feature_cols].fillna(0)
    y_train = train_df[target_col]
    
    # TODO: Initialize and fit your chosen model
    model = ...  # YOUR MODEL HERE
    model.fit(...)
    
    # Generate predictions
    predictions = model.predict(...)
    
    return pd.DataFrame({'predicted': predictions}, index=test_df.index)
""",
                "notes": "Use interpretable, well-established algorithms appropriate for the data."
            },
            "ml_pattern": {
                "description": "Pattern for machine learning methods",
                "template": """
def predict_{METHOD_NAME}(train_df, test_df, target_col, feature_cols=None, **params):
    import pandas as pd
    import numpy as np
    # Import your chosen ML library
    
    # Prepare features
    if feature_cols:
        X_train = train_df[feature_cols].fillna(0)
        X_test = test_df[feature_cols].fillna(0)
    y_train = train_df[target_col]
    
    # Optional: Feature scaling
    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)
    
    # TODO: Initialize your chosen ML model with hyperparameters
    model = ...  # YOUR ML MODEL HERE
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    
    return pd.DataFrame({'predicted': predictions}, index=test_df.index)
""",
                "notes": "Choose ML algorithm based on data size, patterns, and task requirements."
            },
            "timeseries_pattern": {
                "description": "Pattern for time series forecasting",
                "template": """
def predict_{METHOD_NAME}(train_df, test_df, target_col, date_col=None, **params):
    import pandas as pd
    import numpy as np
    # Import your chosen time series library
    
    y_train = train_df[target_col].values
    n_forecast = len(test_df)
    
    # TODO: Implement your time series approach
    # Options: ARIMA, exponential smoothing, Prophet, lag-based ML, etc.
    
    # For statistical TS:
    # model = YOUR_TS_MODEL(y_train, ...)
    # fitted = model.fit()
    # forecast = fitted.forecast(steps=n_forecast)
    
    # For ML with lags:
    # Create lag features, train, predict iteratively
    
    predictions = ...  # YOUR FORECAST HERE
    
    return pd.DataFrame({'predicted': predictions}, index=test_df.index)
""",
                "notes": "Consider data frequency, trend, seasonality when choosing approach."
            },
            "clustering_pattern": {
                "description": "Pattern for clustering methods",
                "template": """
def predict_{METHOD_NAME}(train_df, test_df, feature_cols, **params):
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    # Import your chosen clustering library
    
    X = train_df[feature_cols].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # TODO: Initialize your chosen clustering algorithm
    model = ...  # YOUR CLUSTERING MODEL HERE
    labels = model.fit_predict(X_scaled)
    
    return pd.DataFrame({'predicted': labels}, index=train_df.index)
""",
                "notes": "Consider cluster shapes, number of clusters, and outlier handling."
            }
        },
        "method_structure": {
            "required_fields": [
                "method_id (M1, M2, or M3)",
                "name (descriptive name you choose)",
                "category (baseline, statistical, or ml)",
                "description (what it does and why you chose it)",
                "implementation_code (complete, runnable function)",
                "required_libraries (list of imports)",
                "hyperparameters (key parameters with values)",
                "expected_strengths (list)",
                "expected_weaknesses (list)"
            ],
            "complexity_progression": "M1 (simplest) < M2 (interpretable) < M3 (most complex)"
        },
        "selection_guidance": {
            "analyze_first": [
                "Data size (rows, columns)",
                "Feature types (numeric, categorical, temporal)",
                "Target distribution",
                "Missing values",
                "Patterns (linear, non-linear, seasonal)"
            ],
            "then_choose": [
                "Algorithm family based on task type",
                "Specific algorithm based on data characteristics",
                "Hyperparameters based on data size and complexity"
            ]
        }
    }

    return "Implementation Patterns (adapt based on YOUR data analysis):\n\n" + json.dumps(templates, indent=2)


@tool
def save_method_proposal(proposal_json: str) -> str:
    """
    Save the method proposal to Stage 3.5A output directory.

    Args:
        proposal_json: JSON string with MethodProposalOutput structure

    Returns:
        Confirmation with saved path
    """
    try:
        if DEBUG:
            logger.debug(f"Saving proposal. Input type: {type(proposal_json)}")
            logger.debug(f"Input preview: {str(proposal_json)[:500]}...")

        if isinstance(proposal_json, dict):
            proposal = proposal_json
        else:
            # Clean up the JSON string
            cleaned_json = str(proposal_json).strip()
            if cleaned_json.startswith("```json"):
                cleaned_json = cleaned_json[7:]
            if cleaned_json.startswith("```"):
                cleaned_json = cleaned_json[3:]
            if cleaned_json.endswith("```"):
                cleaned_json = cleaned_json[:-3]
            cleaned_json = cleaned_json.strip()

            try:
                proposal = json.loads(cleaned_json)
                # Handle double-encoded JSON (string inside string)
                if isinstance(proposal, str):
                    if DEBUG:
                        logger.debug("Detected double-encoded JSON string, parsing again...")
                    proposal = json.loads(proposal)
            except json.JSONDecodeError:
                # Try one more time with relaxed parsing if needed, or just fail
                # Sometimes LLM escapes quotes weirdly
                if cleaned_json.startswith('"') and cleaned_json.endswith('"'):
                     cleaned_json = cleaned_json[1:-1].replace('\\"', '"')
                proposal = json.loads(cleaned_json)
                if isinstance(proposal, str):
                    proposal = json.loads(proposal)

        # Validate structure - be more lenient with required fields
        required = ['plan_id', 'methods_proposed']
        missing = [f for f in required if f not in proposal]
        if missing:
            return f"Error: Missing required fields: {missing}"

        methods = proposal.get('methods_proposed', [])
        if not methods:
            return "Error: No methods proposed"

        # Accept 1-3 methods instead of exactly 3 (be lenient)
        if len(methods) > 3:
            logger.warning(f"More than 3 methods proposed ({len(methods)}), using first 3")
            proposal['methods_proposed'] = methods[:3]

        # Ensure data_split_strategy exists with defaults
        if 'data_split_strategy' not in proposal:
            proposal['data_split_strategy'] = {
                'strategy_type': 'temporal',
                'train_size': 0.7,
                'validation_size': 0.15,
                'test_size': 0.15
            }
            logger.warning("Added default data_split_strategy")

        # Ensure date_column and target_column are consistent with data_split_strategy
        # Priority: data_split_strategy > top-level > load from plan
        split_strategy = proposal.get('data_split_strategy', {})

        # Get values from split strategy first
        split_target = split_strategy.get('target_column')
        split_date = split_strategy.get('date_column')

        # If not in proposal but in split_strategy, copy up
        if 'date_column' not in proposal or proposal.get('date_column') == 'date':
            if split_date:
                proposal['date_column'] = split_date
            else:
                # Try to load from plan
                try:
                    plan_id = proposal['plan_id']
                    plan_path = STAGE3_OUT_DIR / f"{plan_id}.json"
                    if plan_path.exists():
                        plan = DataPassingManager.load_artifact(plan_path)
                        proposal['date_column'] = plan.get('date_column')
                        logger.info(f"Loaded date_column from plan: {proposal['date_column']}")
                except Exception:
                    proposal['date_column'] = None
                    logger.warning("date_column set to None (not found in plan)")

        if 'target_column' not in proposal or proposal.get('target_column') == 'target':
            if split_target:
                proposal['target_column'] = split_target
                logger.info(f"Using target_column from data_split_strategy: {split_target}")
            else:
                # Try to load from plan
                try:
                    plan_id = proposal['plan_id']
                    plan_path = STAGE3_OUT_DIR / f"{plan_id}.json"
                    if plan_path.exists():
                        plan = DataPassingManager.load_artifact(plan_path)
                        proposal['target_column'] = plan.get('target_column')
                        logger.info(f"Loaded target_column from plan: {proposal['target_column']}")
                except Exception:
                    proposal['target_column'] = None
                    logger.warning("target_column set to None (not found in plan)")

        # Also ensure data_split_strategy has consistent values
        if 'data_split_strategy' in proposal:
            if proposal.get('target_column') and not proposal['data_split_strategy'].get('target_column'):
                proposal['data_split_strategy']['target_column'] = proposal['target_column']
            if proposal.get('date_column') and not proposal['data_split_strategy'].get('date_column'):
                proposal['data_split_strategy']['date_column'] = proposal['date_column']

        plan_id = proposal['plan_id']
        filename = f"method_proposal_{plan_id}.json"

        # Ensure output directory exists
        STAGE3_5A_OUT_DIR.mkdir(parents=True, exist_ok=True)

        output_path = DataPassingManager.save_artifact(
            data=proposal,
            output_dir=STAGE3_5A_OUT_DIR,
            filename=filename,
            metadata={"stage": "stage3_5a", "type": "method_proposal"}
        )

        if DEBUG:
            logger.debug(f"Successfully saved to {output_path}")

        # Verify the file was actually saved
        if output_path.exists():
            logger.info(f"Verified file exists at: {output_path}")
            return f"SUCCESS: Method proposal saved to: {output_path}"
        else:
            # Fallback: direct write
            logger.warning("DataPassingManager save may have failed, attempting direct write...")
            with open(STAGE3_5A_OUT_DIR / filename, 'w') as f:
                json.dump(proposal, f, indent=2, default=str)
            return f"SUCCESS (fallback): Method proposal saved to: {STAGE3_5A_OUT_DIR / filename}"

    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        logger.error(f"Input was: {str(proposal_json)[:1000]}")
        return f"Error: Invalid JSON - {e}"
    except Exception as e:
        logger.error(f"Error saving proposal: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return f"Error saving proposal: {e}"


def reset_react_state():
    """Reset ReAct tracking state."""
    global _stage3_5a_thoughts, _stage3_5a_observations
    _stage3_5a_thoughts = []
    _stage3_5a_observations = []


@tool
def finish_method_proposal() -> str:
    """
    Signal that method proposal is complete.
    
    Call this ONLY after save_method_proposal returns success.
    
    Returns:
        Completion message
    """
    return "Stage 3.5A Complete. You may now stop."


@tool
def inspect_data_sample(plan_id: str = None, n_rows: int = 10) -> str:
    """
    Get a sample of the prepared data for inspection.

    Args:
        plan_id: Plan ID to inspect
        n_rows: Number of rows to display

    Returns:
        Data sample with statistics
    """
    try:
        if not plan_id:
            plans = list(STAGE3_OUT_DIR.glob("PLAN-*.json"))
            if plans:
                plan_id = max(plans, key=lambda p: p.stat().st_mtime).stem

        prepared_path = STAGE3B_OUT_DIR / f"prepared_{plan_id}.parquet"
        if not prepared_path.exists():
            return f"Prepared data not found at: {prepared_path}"

        df = pd.read_parquet(prepared_path)

        result = [
            f"=== Data Sample: {plan_id} ===",
            f"Shape: {df.shape[0]} rows x {df.shape[1]} columns",
            f"Columns: {list(df.columns)}",
            "",
            "Data Types:",
        ]

        for col in df.columns:
            result.append(f"  {col}: {df[col].dtype}")

        result.append(f"\nFirst {n_rows} rows:")
        result.append(df.head(n_rows).to_string())

        result.append(f"\nLast {n_rows} rows:")
        result.append(df.tail(n_rows).to_string())

        result.append("\nNumeric statistics:")
        result.append(df.describe().to_string())

        return "\n".join(result)

    except Exception as e:
        return f"Error inspecting data: {e}"


@tool
def test_method_code(code: str, plan_id: str = None) -> str:
    """
    Test method implementation code before proposing it.

    This allows you to verify that the implementation works correctly
    before including it in the method proposal.

    Args:
        code: Python code containing the method function
        plan_id: Plan ID for loading test data

    Returns:
        Test results showing if the code executes correctly
    """
    import sys
    from io import StringIO

    try:
        if not plan_id:
            plans = list(STAGE3_OUT_DIR.glob("PLAN-*.json"))
            if plans:
                plan_id = max(plans, key=lambda p: p.stat().st_mtime).stem

        # Load data
        prepared_path = STAGE3B_OUT_DIR / f"prepared_{plan_id}.parquet"
        if not prepared_path.exists():
            return f"Prepared data not found: {prepared_path}"

        df = pd.read_parquet(prepared_path)

        # Load plan for column info
        plan_path = STAGE3_OUT_DIR / f"{plan_id}.json"
        plan = DataPassingManager.load_artifact(plan_path)
        target_col = plan.get('target_column', df.columns[-1])
        date_col = plan.get('date_column', df.columns[0])

        # Setup test namespace
        namespace = {
            'pd': pd,
            'np': np,
            'df': df,
            'target_col': target_col,
            'date_col': date_col,
        }

        # Common ML imports
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.linear_model import LinearRegression
            namespace['RandomForestRegressor'] = RandomForestRegressor
            namespace['LinearRegression'] = LinearRegression
        except ImportError:
            pass

        try:
            from statsmodels.tsa.arima.model import ARIMA
            namespace['ARIMA'] = ARIMA
        except ImportError:
            pass

        # Execute code
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            exec(code, namespace)
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        # Check if a predict function was defined
        predict_funcs = [k for k in namespace.keys() if k.startswith('predict_') or 'method' in k.lower()]

        result = [
            "=== Method Code Test ===",
            f"Code executed successfully!",
            f"Output: {output[:500] if output else '(no output)'}",
            "",
            f"Functions defined: {predict_funcs}",
        ]

        # Try to run the method with a small sample
        if predict_funcs:
            func_name = predict_funcs[0]
            func = namespace.get(func_name)
            if callable(func):
                result.append(f"\nTesting {func_name} with sample data...")
                train_size = int(len(df) * 0.8)
                train_df = df.iloc[:train_size].copy()
                test_df = df.iloc[train_size:train_size+5].copy()

                try:
                    predictions = func(train_df, test_df, target_col, date_col)
                    result.append(f"Test run successful!")
                    result.append(f"Predictions shape: {predictions.shape if hasattr(predictions, 'shape') else len(predictions)}")
                    if hasattr(predictions, 'head'):
                        result.append(f"Sample predictions:\n{predictions.head()}")
                except Exception as e:
                    result.append(f"Test run failed: {e}")

        return "\n".join(result)

    except Exception as e:
        import traceback
        return f"Error testing code: {e}\n{traceback.format_exc()}"


@tool
def debug_pipeline_state(plan_id: str = None) -> str:
    """
    Debug the current pipeline state by checking all stage outputs.

    Args:
        plan_id: Plan ID to check

    Returns:
        Comprehensive status of all pipeline artifacts
    """
    try:
        result = ["=== Pipeline Debug State ===\n"]

        # Check Stage 1
        summaries = list(SUMMARIES_DIR.glob("*.summary.json"))
        result.append(f"Stage 1 (Summaries): {len(summaries)} files")
        for s in summaries[:3]:
            result.append(f"  - {s.name}")

        # Check Stage 2
        proposals_path = STAGE2_OUT_DIR / "task_proposals.json"
        result.append(f"\nStage 2 (Proposals): {'EXISTS' if proposals_path.exists() else 'MISSING'}")

        if plan_id:
            # Check Stage 3
            plan_path = STAGE3_OUT_DIR / f"{plan_id}.json"
            result.append(f"\nStage 3 (Plan): {'EXISTS' if plan_path.exists() else 'MISSING'}")

            # Check Stage 3B
            prepared_path = STAGE3B_OUT_DIR / f"prepared_{plan_id}.parquet"
            result.append(f"Stage 3B (Data): {'EXISTS' if prepared_path.exists() else 'MISSING'}")
            if prepared_path.exists():
                df = pd.read_parquet(prepared_path)
                result.append(f"  Shape: {df.shape}")

            # Check Stage 3.5A
            method_path = STAGE3_5A_OUT_DIR / f"method_proposal_{plan_id}.json"
            result.append(f"Stage 3.5A (Methods): {'EXISTS' if method_path.exists() else 'MISSING'}")
            if method_path.exists():
                methods = DataPassingManager.load_artifact(method_path)
                result.append(f"  Methods: {len(methods.get('methods_proposed', []))}")

            # Check Stage 3.5B
            tester_path = STAGE3_5B_OUT_DIR / f"tester_{plan_id}.json"
            result.append(f"Stage 3.5B (Tester): {'EXISTS' if tester_path.exists() else 'MISSING'}")
            if tester_path.exists():
                tester = DataPassingManager.load_artifact(tester_path)
                result.append(f"  Selected: {tester.get('selected_method_id')}")

            # Check Stage 4
            results_path = STAGE4_OUT_DIR / f"results_{plan_id}.parquet"
            exec_path = STAGE4_OUT_DIR / f"execution_result_{plan_id}.json"
            result.append(f"Stage 4 (Execution): {'EXISTS' if exec_path.exists() else 'MISSING'}")
            result.append(f"Stage 4 (Results): {'EXISTS' if results_path.exists() else 'MISSING'}")

            # Check Stage 5
            viz_path = STAGE5_OUT_DIR / f"visualization_report_{plan_id}.json"
            result.append(f"Stage 5 (Visualization): {'EXISTS' if viz_path.exists() else 'MISSING'}")

        return "\n".join(result)

    except Exception as e:
        return f"Error debugging pipeline: {e}"


@tool
def get_react_summary_3_5a() -> str:
    """
    Get a summary of all recorded thoughts and observations.

    Returns:
        Summary of ReAct reasoning trail
    """
    global _stage3_5a_thoughts, _stage3_5a_observations

    result = ["=== ReAct Summary (Stage 3.5A) ===\n"]

    if _stage3_5a_thoughts:
        result.append("Thoughts:")
        for t in _stage3_5a_thoughts:
            result.append(f"  {t['step']}. {t['thought'][:100]}...")
            result.append(f"     Next: {t['next_action'][:50]}...")
    else:
        result.append("No thoughts recorded yet.")

    result.append("")

    if _stage3_5a_observations:
        result.append("Observations:")
        for o in _stage3_5a_observations:
            result.append(f"  {o['step']}. {o['what_happened'][:100]}...")
            result.append(f"     Insight: {o['insight'][:50]}...")
    else:
        result.append("No observations recorded yet.")

    return "\n".join(result)


@tool
def get_actual_columns(plan_id: str = None) -> str:
    """
    Get the ACTUAL column names from the prepared data.
    
    CRITICAL: Use this to prevent column hallucination. Only use columns
    that are returned by this tool - do not assume or invent column names.
    
    Args:
        plan_id: Plan ID to check
    
    Returns:
        List of actual columns with their data types
    """
    try:
        if not plan_id:
            plans = list(STAGE3_OUT_DIR.glob("PLAN-*.json"))
            if plans:
                plan_id = max(plans, key=lambda p: p.stat().st_mtime).stem
        
        # Load prepared data
        prepared_path = STAGE3B_OUT_DIR / f"prepared_{plan_id}.parquet"
        if not prepared_path.exists():
            return f"ERROR: Prepared data not found at {prepared_path}"
        
        df = pd.read_parquet(prepared_path)
        
        # Load plan to show what was expected vs actual
        plan_path = STAGE3_OUT_DIR / f"{plan_id}.json"
        plan = DataPassingManager.load_artifact(plan_path) if plan_path.exists() else {}
        
        result = [
            f"=== ACTUAL COLUMNS in prepared_{plan_id}.parquet ===",
            f"Total columns: {len(df.columns)}",
            f"Data shape: {df.shape}",
            "",
            "Column Name | Data Type",
            "-" * 40,
        ]
        
        for col in df.columns:
            result.append(f"{col} | {df[col].dtype}")
        
        result.append("")
        result.append("=== Plan Expectations vs Reality ===")
        
        expected_date = plan.get('date_column')
        expected_target = plan.get('target_column')
        
        if expected_date:
            status = "✓ EXISTS" if expected_date in df.columns else "✗ MISSING"
            result.append(f"Expected date_column: {expected_date} ... {status}")
            if expected_date not in df.columns:
                result.append(f"  WARNING: Use df.index or set date_col=None in your code!")
        
        if expected_target:
            status = "✓ EXISTS" if expected_target in df.columns else "✗ MISSING"
            result.append(f"Expected target_column: {expected_target} ... {status}")
        
        result.append("")
        result.append("⚠️  CRITICAL: Use ONLY the columns listed above!")
        result.append("⚠️  Do NOT assume or invent column names like 'Year', 'date', etc.")
        
        return "\n".join(result)
        
    except Exception as e:
        return f"Error getting actual columns: {e}"


# Import SUMMARIES_DIR and STAGE2_OUT_DIR for debug tool
from code.config import SUMMARIES_DIR, STAGE2_OUT_DIR, STAGE4_OUT_DIR, STAGE5_OUT_DIR


# Export tools list
STAGE3_5A_TOOLS = [
    load_plan_and_data,
    get_actual_columns,  # NEW: Prevent column hallucination
    analyze_time_series,
    record_thought_3_5a,
    record_observation_3_5a,
    python_sandbox_stage3_5a,
    get_method_templates,
    save_method_proposal,
    finish_method_proposal,
    # New debugging tools
    inspect_data_sample,
    test_method_code,
    debug_pipeline_state,
    get_react_summary_3_5a,
]
