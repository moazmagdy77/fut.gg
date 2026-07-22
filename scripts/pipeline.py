import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# --- Directory Setup ---
# Set working directory to the script's own directory (fut.gg/scripts)
base_dir = Path(__file__).resolve().parent
# Define the repository root (the parent 'fut.gg' folder)
repo_root = base_dir.parent

steps = [
    ("🔎 Step 1: Extract CLUB player IDs", [sys.executable, "club.1.get.ids.py"]),
    ("🧹 Step 1.5: Invalidate upgraded/evolved players", [sys.executable, "invalidate_upgraded_players.py"]),
    ("📦 Step 2a: Scrape player data (≥75 OVR)", ["node", "shared.fetch.data.js", "--file", "../data/club_ids.json", "--mode", "club"]),
    ("📦 Step 2a+: Scrape player data (All IDs, for tall detection)", ["node", "shared.fetch.data.js", "--file", "../data/all_club_ids.json", "--mode", "club"]),
    ("💰 Step 2b: Fetch prices (Tradeables)", ["node", "club.2b.fetch.prices.js"]),
    # Step 3 command will be built dynamically to include --min-height
]

# --- User Prompts ---
run_club = input("\nDo you want to run the Club Update pipeline (Steps 1, 2, 3)? (y/n): ").strip().lower()

if run_club == 'y':
    run_prices = input("Do you want to run Step 2b: Fetch prices (Tradeables)? (y/n): ").strip().lower()
    if run_prices != 'y':
        print("Skipping Step 2b: Fetch prices...\n")
        steps = [s for s in steps if "club.2b.fetch.prices.js" not in s[1]]
    
    min_height_input = input("Minimum height for tall-player group (default 195 cm): ").strip()
    min_height_cm = int(min_height_input) if min_height_input.isdigit() else 195
    print(f"Tall-player threshold: {min_height_cm} cm")

    run_evo = input("Refresh evo data (fut.gg Evo Lab + EasySBC esMeta)? (y/n): ").strip().lower()

    # Append Step 3 with --min-height argument
    steps.append(("🔍 Step 3: Enrich with evo and all cleaning", [sys.executable, "club.3.clean.py", "--min-height", str(min_height_cm)]))
else:
    run_prices = 'n'
    run_evo = 'n'
    min_height_cm = 195
    print("Skipping Club Update pipeline...\n")
    steps = []

run_fodder = input("Do you want to update Fodder prices (fut.gg cheapest-by-rating)? (y/n): ").strip().lower()

start = time.time()

# --- Run Data Pipeline (fut.gg) ---
if run_club == 'y':
    # Refresh evo data first (authed fut.gg + EasySBC). Non-fatal: fall back to existing files.
    if run_evo == 'y':
        for label, command in [
            ("🧬 Fetch Evo Lab (fut.gg)", ["node", "fetch.evolab.js"]),
            ("🧪 Fetch evo esMeta (EasySBC)", ["node", "fetch.evo.esmeta.js"]),
        ]:
            print(f"\n{label}")
            if subprocess.run(command, cwd=base_dir).returncode != 0:
                print(f"⚠️ {label} failed — continuing with existing data.")

    for label, command in steps:
        print(f"\n{label}")
        print(command)
        # These scripts live in 'scripts/' so we run them from base_dir
        result = subprocess.run(command, cwd=base_dir)

        if result.returncode != 0:
            print(f"\n❌ Failed at: {label}")
            sys.exit(result.returncode)

# --- Fodder prices (fut.gg cheapest-by-rating) ---
if run_fodder == 'y':
    print(f"\n🍞 Updating fodder prices (fut.gg cheapest-by-rating)...")
    if subprocess.run(["node", "fetch.fodder.prices.js"], cwd=base_dir).returncode != 0:
        print("⚠️ Fodder price fetch failed — continuing (existing fodder_prices.json kept).")

# --- Run Git Automation (fut.gg) ---
if run_club == 'y' or run_fodder == 'y':
    print(f"\n🐙 Automating Git Push...")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parts = []
    if run_club == 'y':
        parts.append("new players + prices" if run_prices == 'y' else "new players (prices not updated)")
    if run_fodder == 'y':
        parts.append("fodder prices")
    commit_message = f"{', '.join(parts)} [{timestamp}]"

    print("Running: git add .")
    subprocess.run(["git", "add", "."], cwd=repo_root)
    print("Running: git commit")
    committed = subprocess.run(["git", "commit", "-m", commit_message], cwd=repo_root).returncode == 0
    if not committed:
        print("ℹ️ Nothing new to commit — skipping push.")
    else:
        print("Running: git push origin main")
        if subprocess.run(["git", "push", "origin", "main"], cwd=repo_root).returncode == 0:
            print("✅ Pushed to origin/main — the deployed app will update shortly.")
        else:
            print("⚠️ git push FAILED (see error above). Your commit is saved LOCALLY only.")
            print("   Fix GitHub auth (e.g. `gh auth login` as the repo owner) and run `git push origin main`.")

print(f"\n✅ Master pipeline & sync completed successfully in {round(time.time() - start, 2)} seconds!")