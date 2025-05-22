import subprocess
import sys
import time
from pathlib import Path

# Set working directory to the script's own directory
base_dir = Path(__file__).resolve().parent
data_dir = base_dir.parent.parent / "data"

steps = [
    ("🧹 Step 1: Clean evolab JSON", ["python", "evo.1.clean.py"]),
    ("📊 Step 2: Prepare prediction-ready CSV", ["python", "evo.2.prep.py"]),
    ("🧠 Step 3: Predict meta ratings", ["python", "evo.3.predict.py"]),
    ("📦 Step 4: Inject predicted meta into JSON", ["python", "evo.4.get.meta.py"]),
]

start = time.time()

for label, command in steps:
    print(f"\n{label}")
    result = subprocess.run(command, cwd=base_dir)

    if result.returncode != 0:
        print(f"\n❌ Failed at: {label}")
        sys.exit(result.returncode)

print(f"\n✅ Evo pipeline completed successfully in {round(time.time() - start, 2)} seconds!")