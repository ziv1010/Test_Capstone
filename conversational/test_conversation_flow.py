#!/usr/bin/env python3
"""
Test script to diagnose conversation flow issues.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from code.master_orchestrator import ConversationalOrchestrator

# Simulate what happens when user types 'run tsk 1'
print("=" * 60)
print("Testing: 'run tsk 1 analysis'")
print("=" * 60)

orch = ConversationalOrchestrator()
result = orch.process_user_input('run tsk 1 analysis')

print(f"\nResponse: {result.get('response')}")
print(f"Action: {result.get('action')}")
print(f"Task ID: {result.get('task_id')}")
print(f"Pipeline Started: {result.get('pipeline_started')}")
print(f"Pipeline Completed: {result.get('pipeline_completed')}")

if result.get('task_id'):
    print(f"\n--- Task ID was detected: {result['task_id']} ---")
    print("Checking if pipeline was called...")
else:
    print("\n!!! Task ID was NOT detected !!!")

print("\n" + "=" * 60)
print("Testing: 'run task TSK-001'")
print("=" * 60)

result2 = orch.process_user_input('run task TSK-001')
print(f"\nResponse: {result2.get('response')}")
print(f"Action: {result2.get('action')}")
print(f"Task ID: {result2.get('task_id')}")
print(f"Pipeline Started: {result2.get('pipeline_started')}")
print(f"Pipeline Completed: {result2.get('pipeline_completed')}")
