# Fixed: Max Token Error Handling and Automatic Retry

## Problem

When the LLM agent hit max_tokens errors in Stage 3.5a or 3.5b:
1. The agent would crash without saving proper output
2. **Fallback logic** would create **dummy/placeholder data** with fake metrics
3. The pipeline would continue to the next stage with this bad data
4. Results were meaningless (all methods had identical metrics: MAE=100, RMSE=120, MAPE=10)

## Root Cause

Both `stage3_5a_agent.py` and `stage3_5b_agent.py` had fallback functions:
- `_create_default_method_proposal()` in Stage 3.5a
- `_create_default_tester_output()` in Stage 3.5b

These functions were called when the agent failed to save output, creating dummy data to allow the pipeline to continue.

## Solution

### 1. Removed Fallback Logic ❌

**File: `code/stage3_5a_agent.py`**
- Removed `_create_default_method_proposal()` function
- Modified `run_stage3_5a()` to raise `RuntimeError` instead of creating fallback data
- Added detection for max_tokens errors with clear error messages

**File: `code/stage3_5b_agent.py`**
- Removed `_create_default_tester_output()` function  
- Modified `run_stage3_5b()` to raise `RuntimeError` instead of creating fallback data
- Added detection for max_tokens errors with clear error messages

### 2. Added Retry Configuration ⚙️

**File: `code/config.py`**
```python
# Retry parameters
MAX_RETRIES = 3  # Maximum number of retries for failed stages
RETRY_STAGES = ["stage3_5a", "stage3_5b"]  # Stages that support retry
```

### 3. Implemented Automatic Retry Logic 🔄

**Modified Node Functions:**
- `stage3_5a_node()` - Added retry loop with intelligent error detection
- `stage3_5b_node()` - Added retry loop with intelligent error detection

**Retry Logic Features:**
- Automatically retries up to 3 times when max_tokens or other transient errors occur
- Cleans up partial outputs before each retry to ensure fresh start
- Identifies retryable errors:
  - `max_tokens` errors
  - Token-related errors
  - "did not save" errors
- Stops retrying for non-transient errors
- Provides clear logging for each attempt

## How It Works Now

```
Stage 3.5a/3.5b Execution:
├─ Attempt 1
│  └─ ❌ Fails with max_tokens error
│     ├─ Detects retryable error
│     ├─ Cleans up partial output
│     └─ Continues to Attempt 2
├─ Attempt 2
│  └─ ❌ Fails again
│     ├─ Cleans up partial output
│     └─ Continues to Attempt 3
└─ Attempt 3
   └─ ✅ Succeeds!
      └─ Marks stage as completed
```

If all 3 attempts fail:
```
❌ Stage 3.5B failed after 3 attempts. Last error: ...
Pipeline STOPS (does not continue with bad data)
```

## Benefits

✅ **No More Fake Data**: Pipeline will never proceed with placeholder metrics  
✅ **Automatic Recovery**: Transient errors are automatically retried  
✅ **Clean Retries**: Partial outputs are removed before each retry  
✅ **Clear Logging**: Each attempt is logged with attempt number  
✅ **Fail Fast**: Non-retryable errors stop immediately  
✅ **Configurable**: Retry count and retry-enabled stages can be adjusted in config

## Example Log Output

```
2025-12-04 13:45:00 - INFO - Stage 3.5B attempt 1/3
2025-12-04 13:46:30 - ERROR - Stage 3.5B failed with token error: max_tokens exceeded
⚠️  Stage 3.5B attempt 1 failed with retryable error. Retrying... (1/3)
2025-12-04 13:46:30 - INFO - Removing partial output: tester_PLAN-TSK-4596.json
2025-12-04 13:46:31 - INFO - Stage 3.5B attempt 2/3
2025-12-04 13:48:15 - INFO - ✅ Stage 3.5B succeeded on attempt 2
```

## Testing

To test the retry mechanism:
1. Run a task that previously failed with max_tokens
2. Check logs for retry attempts
3. Verify that the pipeline either:
   - Succeeds after retrying, OR
   - Stops cleanly after max retries (no fallback data created)

## Related Changes

Also fixed in this session:
- `tools/stage3_tools.py`: Added Pydantic validation to `save_stage3_plan` to catch schema errors early
