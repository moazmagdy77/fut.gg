import subprocess
import sys
import time
from pathlib import Path

script_dir = Path(__file__).parent

steps = [
    ("🔎 Step 1: Scraping player IDs", ["node", str(script_dir / "model.1.get_ids.js")]),
    ("📦 Step 2: Scraping player details", ["node", str(script_dir / "model.2.get.gg.js")]),
    ("🧹 Step 3: Cleaning raw player data", ["python", str(script_dir / "model.3.clean.gg.py")]),
    ("🔍 Step 4: Enriching with meta ratings", ["node", str(script_dir / "model.4.get.meta.js")]),
    ("🧼 Step 5: Final data clean & flatten", ["python", str(script_dir / "model.5.clean.meta.py")]),
    ("🤖 Step 6: Training regression models", ["python", str(script_dir / "model.6.train.py")]),
]

start = time.time()

for label, command in steps:
    print(f"\n{label}")
    result = subprocess.run(command)

    if result.returncode != 0:
        print(f"\n❌ Failed at: {label}")
        sys.exit(result.returncode)

print(f"\n✅ All steps completed successfully in {round(time.time() - start, 2)} seconds!")