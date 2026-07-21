import subprocess
import sys
import time
from pathlib import Path

# Set working directory to the script's own directory
base_dir = Path(__file__).resolve().parent

# Define the initial sequential steps (Fetching & Parsing)
# NOTE: We use sys.executable to ensure we use the ACTIVE venv python, not the system python
initial_steps = [
    ("🔎 Step 1: Scrape FUT.GG to get Player IDs", ["node", "model.1.get_ids.js"]),
    ("📦 Step 2: Scrape player details and metaratings", ["node", "shared.fetch.data.js", "--file", "../data/player_ids.json", "--mode", "model"]),
]

start = time.time()

# --- Part 1: Sequential Data Fetching ---
for label, command in initial_steps:
    print(f"\n{label}")
    # We pass env=None so it inherits the current environment (with VIRTUAL_ENV variables)
    result = subprocess.run(command, cwd=base_dir)

    if result.returncode != 0:
        print(f"\n❌ Failed at: {label}")
        sys.exit(result.returncode)

# --- Part 2: Build the ggMetaSub training dataset ---
# esMeta is fetched exact (base API + EasySBC evo), so no ES dataset/model is built here.
print("\n🔎 Step 6: Build ggMetaSub training dataset")
res_6 = subprocess.run([sys.executable, "model.6.build_gg_sub_dataset.py"], cwd=base_dir)
if res_6.returncode != 0:
    print("\n❌ Failed at: Step 6 (Build ggMeta dataset)")
    sys.exit(res_6.returncode)

# --- Part 3: Model Training ---
print("\n🔍 Step 7: Train ggMetaSub models")
res_7 = subprocess.run([sys.executable, "model.7.train_gg_sub_models.py"], cwd=base_dir)
if res_7.returncode != 0:
    print("\n❌ Failed at: Step 7 (Train ggMeta models)")
    sys.exit(res_7.returncode)

# --- Part 4: All-Players Summary ---
# MUST run AFTER Step 7: build_all_players_summary now predicts ggMetaSub with the
# freshly-trained models, so building it earlier (the old parallel-with-step-6 layout)
# would bake in stale or missing models.
print("\n📊 Step 8: Build All-Players Summary (with ggMetaSub)")
res_all = subprocess.run([sys.executable, "build_all_players_summary.py"], cwd=base_dir)
if res_all.returncode != 0:
    print("\n❌ Failed at: Build All-Players Summary")
    sys.exit(res_all.returncode)

print(f"\n✅ Club pipeline completed successfully in {round(time.time() - start, 2)} seconds!")