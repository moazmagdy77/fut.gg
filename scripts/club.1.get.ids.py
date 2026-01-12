from bs4 import BeautifulSoup
import json
from pathlib import Path

# Define data directory
try:
    data_dir = Path(__file__).resolve().parent.parent / "data"
except NameError:
    data_dir = Path(".").resolve().parent / "data"
    if not data_dir.exists():
        data_dir = Path(".").resolve()

# Load the HTML content
html_file_path = data_dir / "club-analyzer.html"
with open(html_file_path, "r", encoding="utf-8") as f:
    soup = BeautifulSoup(f, "html.parser")

# Parse the table rows
rows = soup.find_all("tr")

# Get the header to determine column indices
headers = [th.text.strip() for th in rows[0].find_all("th")]
id_idx = headers.index("Id")
location_idx = headers.index("Location")
rating_idx = headers.index("Rating")

# Attempt to find Untradeable column, default to 12 (Column 13 0-indexed) if not found by name
try:
    untradeable_idx = headers.index("Untradeable")
except ValueError:
    untradeable_idx = 12

# Extract IDs
club_ids = []
tradeable_ids = []

for row in rows[1:]:
    cols = row.find_all("td")
    if cols and cols[location_idx].text.strip() == "CLUB":
        ea_id = cols[id_idx].text.strip()
        club_ids.append(ea_id)
        
        # Check if Untradeable is False
        untradeable_text = cols[untradeable_idx].text.strip()
        # Checks for "False" string or empty/falsy values depending on HTML format
        if untradeable_text.lower() == "false":
            tradeable_ids.append(ea_id)

# Save CLUB IDs (Main list for data fetching)
output_file_path = data_dir / "club_ids.json"
with open(output_file_path, "w") as out:
    json.dump(club_ids, out, indent=2)

# Save TRADEABLE IDs (Subset for price fetching)
tradeable_file_path = data_dir / "tradeable_ids.json"
with open(tradeable_file_path, "w") as out:
    json.dump(tradeable_ids, out, indent=2)

print(f"Extracted {len(club_ids)} CLUB Player IDs -> {output_file_path}")
print(f"Extracted {len(tradeable_ids)} Tradeable Player IDs -> {tradeable_file_path}")