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
# Define the absolute path to your second repository
futbin_scraper_dir = Path(r"D:\Dev\futbin-scraper")

steps = [
    ("🔎 Step 1: Extract CLUB player IDs", [sys.executable, "club.1.get.ids.py"]),
    ("📦 Step 2: Scrape player data", ["node", "shared.fetch.data.js", "--file", "../data/club_ids.json", "--mode", "club"]),
    ("💰 Step 2b: Fetch prices (Tradeables)", ["node", "club.2b.fetch.prices.js"]),
    ("🔍 Step 3: Enrich with evo and all cleaning", [sys.executable, "club.3.clean.py"]),
]

# --- User Prompts ---
run_club = input("\nDo you want to run the Club Update pipeline (Steps 1, 2, 3)? (y/n): ").strip().lower()

if run_club == 'y':
    run_prices = input("Do you want to run Step 2b: Fetch prices (Tradeables)? (y/n): ").strip().lower()
    if run_prices != 'y':
        print("Skipping Step 2b: Fetch prices...\n")
        steps = [s for s in steps if "club.2b.fetch.prices.js" not in s[1]]
else:
    run_prices = 'n'
    print("Skipping Club Update pipeline...\n")
    steps = []

run_fodder = input("Do you want to update Fodder Prices (futbin-scraper)? (y/n): ").strip().lower()

start = time.time()

# --- Run Data Pipeline (fut.gg) ---
if run_club == 'y':
    for label, command in steps:
        print(f"\n{label}")
        print(command)
        # These scripts live in 'scripts/' so we run them from base_dir
        result = subprocess.run(command, cwd=base_dir)

        if result.returncode != 0:
            print(f"\n❌ Failed at: {label}")
            sys.exit(result.returncode)

# --- Run Git Automation (fut.gg) ---
if run_club == 'y':
    print(f"\n🐙 Automating Git Push...")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if run_prices == 'y':
        commit_message = f"new players + prices [{timestamp}]"
    else:
        commit_message = f"new players (prices not updated) [{timestamp}]"

    git_commands = [
        ["git", "add", "."],
        ["git", "commit", "-m", commit_message],
        ["git", "push", "origin", "main"]
    ]

    for cmd in git_commands:
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=repo_root)

# --- Run Fodder Scraper (futbin-scraper) ---
if run_fodder == 'y':
    print(f"\n📈 Updating Fodder Prices via futbin-scraper...")
    # We change the cwd to the other repository so Node finds the correct .env and index.js
    scraper_result = subprocess.run(["node", "index.js"], cwd=futbin_scraper_dir)
    
    if scraper_result.returncode != 0:
        print(f"\n⚠️ futbin-scraper encountered an error, but the main pipeline is safe.")
    else:
        print(f"\n✅ Fodder prices updated in Google Sheets!")
else:
    print("\nSkipping futbin-scraper...")

print(f"\n✅ Master pipeline & sync completed successfully in {round(time.time() - start, 2)} seconds!")