import subprocess
import sys
import time
from pathlib import Path

# Set working directory to the script's own directory
base_dir = Path(__file__).resolve().parent
data_dir = base_dir.parent.parent / "data"

steps = [
    ("🔎 Step 1: Extract CLUB player IDs", ["python", "club.1.get.ids.py"]),
    ("📦 Step 2: Scrape player data", ["node", "club.2.get.gg.js"]),
    ("🧹 Step 3: Clean player data", ["python", "club.3.clean.gg.py"]),
    ("🔍 Step 4: Enrich with meta ratings", ["node", "club.4.get.meta.js"]),
]

start = time.time()

for label, command in steps:
    print(f"\n{label}")
    result = subprocess.run(command, cwd=base_dir)

    if result.returncode != 0:
        print(f"\n❌ Failed at: {label}")
        sys.exit(result.returncode)

print(f"\n✅ Club pipeline completed successfully in {round(time.time() - start, 2)} seconds!")