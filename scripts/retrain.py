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

# --- Part 2: Parallel Dataset Building ---
# We run Step 4, Step 6, and All-Players Summary at the same time to save time
print("\n🔎 Steps 4, 6 & All-Players Summary: Building in Parallel...")

# Use sys.executable here too!
p_es = subprocess.Popen([sys.executable, "model.4.build_es_training_dataset.py"], cwd=base_dir)
p_gg = subprocess.Popen([sys.executable, "model.6.build_gg_sub_dataset.py"], cwd=base_dir)
p_all = subprocess.Popen([sys.executable, "build_all_players_summary.py"], cwd=base_dir)

# Wait for both processes to finish
exit_code_es = p_es.wait()
exit_code_gg = p_gg.wait()
exit_code_all = p_all.wait()

# Check for errors
if exit_code_es != 0:
    print("\n❌ Failed at: Step 4 (Build esMeta dataset)")
    sys.exit(exit_code_es)

if exit_code_gg != 0:
    print("\n❌ Failed at: Step 6 (Build ggMeta dataset)")
    sys.exit(exit_code_gg)

if exit_code_all != 0:
    print("\n❌ Failed at: Build All-Players Summary")
    sys.exit(exit_code_all)

# --- Part 3: Sequential Model Training ---
# We run these sequentially to allow the heavy ML training (ElasticNet) to utilize full CPU cores 
# without fighting for resources. Parallelizing these usually slows down the total time.

print("\n🔍 Step 5: Train unified esMeta models")
res_5 = subprocess.run([sys.executable, "model.5.train_es_models.py"], cwd=base_dir)
if res_5.returncode != 0:
    print("\n❌ Failed at: Step 5 (Train unified esMeta models)")
    sys.exit(res_5.returncode)

print("\n🔍 Step 7: Train ggMeta and ggMetaSub models")
res_7 = subprocess.run([sys.executable, "model.7.train_gg_sub_models.py"], cwd=base_dir)
if res_7.returncode != 0:
    print("\n❌ Failed at: Step 7 (Train ggMeta models)")
    sys.exit(res_7.returncode)

print(f"\n✅ Club pipeline completed successfully in {round(time.time() - start, 2)} seconds!")