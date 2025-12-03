#!/usr/bin/env python3
"""
Test the fixed conversation flow with various task ID formats.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from code.conversation_agent import ConversationHandler

print("=" * 70)
print("TESTING TASK ID PARSING AND VALIDATION")
print("=" * 70)

handler = ConversationHandler()

# Test cases
test_inputs = [
    ("run tsk 1", "Should match TSK-001"),
    ("run task TSK-001", "Should match TSK-001 directly"),
    ("execute tsk 2", "Should match TSK-002"),
    ("run tsk 9586", "Should match TSK-9586"),
    ("run task TSK-999", "Should fail - task doesn't exist"),
]

for user_input, expected in test_inputs:
    print(f"\n{'=' * 70}")
    print(f"Input: '{user_input}'")
    print(f"Expected: {expected}")
    print("-" * 70)
    
    result = handler._detect_pipeline_action(user_input, "")
    
    action = result.get("action")
    task_id = result.get("task_id")
    
    if action == "run_pipeline":
        print(f"✓ Action detected: {action}")
        print(f"✓ Task ID: {task_id}")
        if task_id is None:
            print("✗ Task validation failed (as expected for invalid tasks)")
        else:
            print(f"✓ Task validated successfully!")
    else:
        print(f"✗ No pipeline action detected")

print(f"\n{'=' * 70}")
print("TEST COMPLETE")
print("=" * 70)
