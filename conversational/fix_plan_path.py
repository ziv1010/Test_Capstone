"""
Fix the Stage 3 Plan JSON to match the expected schema.

This script fixes two issues:
1. Converts filters from string to list
2. Converts validation_strategy from dict to string
"""

import json
from pathlib import Path

plan_path = Path("/scratch/ziv_baretto/conversational_agent/Test_Capstone/conversational/output/stage3_out/PLAN-TSK-4106.json")

# Load the plan
with open(plan_path, 'r') as f:
    plan_data = json.load(f)

# Fix issue 1: Convert filters from string to list
if 'data' in plan_data and 'file_instructions' in plan_data['data']:
    for file_inst in plan_data['data']['file_instructions']:
        if isinstance(file_inst.get('filters'), str):
            # Convert string to list
            file_inst['filters'] = [file_inst['filters']]
            print(f"✓ Fixed filters: {file_inst['filters']}")

# Fix issue 2: Convert validation_strategy from dict to string
if 'data' in plan_data and 'validation_strategy' in plan_data['data']:
    if isinstance(plan_data['data']['validation_strategy'], dict):
        # Extract just the type
        plan_data['data']['validation_strategy'] = plan_data['data']['validation_strategy'].get('type', 'temporal')
        print(f"✓ Fixed validation_strategy: {plan_data['data']['validation_strategy']}")

# Save the fixed plan
with open(plan_path, 'w') as f:
    json.dump(plan_data, f, indent=2)

print(f"\n✅ Fixed plan saved to: {plan_path}")
print("\nYou can now re-run the pipeline and it should proceed to stage 3B!")
