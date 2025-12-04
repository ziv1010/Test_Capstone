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
    DataPassingManager, logger
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
        if DEBUG:
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
    Get templates for the three required forecasting methods.

    Returns templates for baseline, statistical, and ML methods.
    """
    templates = {
        "baseline": {
            "method_id": "M1",
            "name": "Moving Average",
            "category": "baseline",
            "description": "Simple moving average baseline",
            "implementation_code": """
def predict_moving_average(train_df, test_df, target_col, window=7):
    import pandas as pd
    import numpy as np

    # Calculate moving average from training data
    last_values = train_df[target_col].tail(window)
    prediction = last_values.mean()

    # Create predictions DataFrame
    predictions = pd.DataFrame({
        'predicted': [prediction] * len(test_df)
    }, index=test_df.index)

    return predictions
""",
            "required_libraries": ["pandas", "numpy"],
            "hyperparameters": {"window": 7},
            "expected_strengths": ["Simple", "Robust to outliers", "No training needed"],
            "expected_weaknesses": ["Cannot capture trends", "Ignores seasonality"]
        },

        "statistical": {
            "method_id": "M2",
            "name": "ARIMA",
            "category": "statistical",
            "description": "Auto-regressive integrated moving average",
            "implementation_code": """
def predict_arima(train_df, test_df, target_col, date_col, order=(1,1,1)):
    import pandas as pd
    import numpy as np
    from statsmodels.tsa.arima.model import ARIMA

    # Prepare training data
    y_train = train_df[target_col].values

    # Fit ARIMA model
    model = ARIMA(y_train, order=order)
    fitted = model.fit()

    # Forecast
    n_forecast = len(test_df)
    forecast = fitted.forecast(steps=n_forecast)

    # Create predictions DataFrame
    predictions = pd.DataFrame({
        'predicted': forecast
    }, index=test_df.index)

    return predictions
""",
            "required_libraries": ["pandas", "numpy", "statsmodels"],
            "hyperparameters": {"order": [1, 1, 1]},
            "expected_strengths": ["Captures trends", "Well-understood theory", "Good for stationary data"],
            "expected_weaknesses": ["Assumes linear relationships", "Sensitive to parameter tuning"]
        },

        "ml": {
            "method_id": "M3",
            "name": "Random Forest",
            "category": "ml",
            "description": "Random forest with lag features",
            "implementation_code": """
def predict_random_forest(train_df, test_df, target_col, date_col, n_lags=7):
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor

    def create_lag_features(df, target_col, n_lags):
        result = df.copy()
        for i in range(1, n_lags + 1):
            result[f'lag_{i}'] = result[target_col].shift(i)
        return result.dropna()

    # Create features
    train_with_lags = create_lag_features(train_df, target_col, n_lags)
    feature_cols = [f'lag_{i}' for i in range(1, n_lags + 1)]

    X_train = train_with_lags[feature_cols]
    y_train = train_with_lags[target_col]

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict (using last known values for lags)
    predictions = []
    last_values = list(train_df[target_col].tail(n_lags).values)

    for _ in range(len(test_df)):
        X_pred = pd.DataFrame([last_values[::-1]], columns=feature_cols)
        pred = model.predict(X_pred)[0]
        predictions.append(pred)
        last_values = [pred] + last_values[:-1]

    return pd.DataFrame({'predicted': predictions}, index=test_df.index)
""",
            "required_libraries": ["pandas", "numpy", "scikit-learn"],
            "hyperparameters": {"n_lags": 7, "n_estimators": 100},
            "expected_strengths": ["Handles non-linear patterns", "Feature importance", "Robust to outliers"],
            "expected_weaknesses": ["May overfit", "Cannot extrapolate beyond training range"]
        }
    }

    return "Method Templates:\n\n" + json.dumps(templates, indent=2)


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
            logger.debug(f"Input preview: {str(proposal_json)[:200]}...")

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

        # Validate structure
        required = ['plan_id', 'methods_proposed', 'data_split_strategy', 'date_column', 'target_column']
        missing = [f for f in required if f not in proposal]
        if missing:
            return f"Error: Missing required fields: {missing}"

        if len(proposal.get('methods_proposed', [])) != 3:
            return "Error: Exactly 3 methods must be proposed"

        plan_id = proposal['plan_id']
        filename = f"method_proposal_{plan_id}.json"

        output_path = DataPassingManager.save_artifact(
            data=proposal,
            output_dir=STAGE3_5A_OUT_DIR,
            filename=filename,
            metadata={"stage": "stage3_5a", "type": "method_proposal"}
        )

        if DEBUG:
            logger.debug(f"Successfully saved to {output_path}")

        return f"Method proposal saved to: {output_path}"

    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON - {e}"
    except Exception as e:
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


# Export tools list
STAGE3_5A_TOOLS = [
    load_plan_and_data,
    analyze_time_series,
    record_thought_3_5a,
    record_observation_3_5a,
    python_sandbox_stage3_5a,
    get_method_templates,
    save_method_proposal,
    finish_method_proposal,
]
