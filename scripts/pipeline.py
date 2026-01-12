import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Set working directory to the script's own directory
base_dir = Path(__file__).resolve().parent
# Define the repository root (the parent 'fut.gg' folder)
repo_root = base_dir.parent

steps = [
    ("🔎 Step 1: Extract CLUB player IDs", [sys.executable, "club.1.get.ids.py"]),
    ("📦 Step 2: Scrape player data", ["node", "club.2.fetch.data.js"]),
    ("💰 Step 2b: Fetch prices (Tradeables)", ["node", "club.2b.fetch.prices.js"]),
    ("🔍 Step 3: Enrich with evo and all cleaning", [sys.executable, "club.3.clean.py"]),
]

start = time.time()

# --- Run Data Pipeline ---
for label, command in steps:
    print(f"\n{label}")
    print(command)
    # These scripts live in 'scripts/' so we run them from base_dir
    result = subprocess.run(command, cwd=base_dir)

    if result.returncode != 0:
        print(f"\n❌ Failed at: {label}")
        sys.exit(result.returncode)

# --- Run Git Automation ---
print(f"\n🐙 Automating Git Push...")

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
commit_message = f"new players + prices [{timestamp}]"

git_commands = [
    ["git", "add", "."],
    ["git", "commit", "-m", commit_message],
    ["git", "push", "origin", "main"]
]

for cmd in git_commands:
    print(f"Running: {' '.join(cmd)}")
    # We run git commands from the repo_root (fut.gg folder) so 'git add .' captures everything
    subprocess.run(cmd, cwd=repo_root)

print(f"\n✅ Club pipeline & sync completed successfully in {round(time.time() - start, 2)} seconds!")