import subprocess
import sys
import time
from pathlib import Path

# Set working directory to the script's own directory
base_dir = Path(__file__).resolve().parent
data_dir = base_dir.parent.parent / "data"

steps = [
    ("🔎 Step 1: Scrape FUT.GG to get R objects", ["node", "model.1.get_ids.js"]),
    ("📦 Step 2: Clean R objects to get player IDs", ["python", "model.2.parse_r_objects.py"]),
    ("📦 Step 3: Scrape player details and metaratings", ["node", "model.3.fetch.data.js"]),
    ("🔍 Step 4: Build esMeta training dataset", ["python", "model.4.build_es_training_dataset.py"]),
    ("🔍 Step 5: Train esMeta and esMetaSub models", ["python", "model.5.train_es_model.py"]),
    ("🔍 Step 6: Build ggMeta training dataset", ["python", "model.6.build_gg_training_dataset.py"]),
    ("🔍 Step 7: Train ggMeta and ggMetaSub models", ["python", "model.7.train_gg_models.py"])
]

start = time.time()

for label, command in steps:
    print(f"\n{label}")
    result = subprocess.run(command, cwd=base_dir)

    if result.returncode != 0:
        print(f"\n❌ Failed at: {label}")
        sys.exit(result.returncode)

print(f"\n✅ Club pipeline completed successfully in {round(time.time() - start, 2)} seconds!")