
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from code.config import STAGE3_OUT_DIR, DataPassingManager

plan_path = STAGE3_OUT_DIR / "PLAN-TSK-001.json"
print(f"Updating plan: {plan_path}")

# Load plan (ignoring checksum for now since we want to fix it)
data = DataPassingManager.load_artifact(plan_path, verify_checksum=False)

# Fix filepath
for instruction in data.get("file_instructions", []):
    filename = instruction.get("filename")
    if filename:
        # Remove the incorrect /data/ prefix and make it relative or use DATA_DIR in logic
        # We'll just set filepath to filename so agent relies on DATA_DIR
        instruction["filepath"] = filename
        print(f"Fixed filepath for {filename}")

# Save with new checksum
DataPassingManager.save_artifact(
    data=data,
    output_dir=STAGE3_OUT_DIR,
    filename="PLAN-TSK-001.json",
    metadata={"stage": "stage3", "type": "execution_plan", "note": "Fixed filepath"}
)

print("Plan updated successfully.")
