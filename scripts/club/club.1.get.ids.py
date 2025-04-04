from bs4 import BeautifulSoup
import json
from pathlib import Path

# Define data directory
data_dir = Path(__file__).resolve().parents[2] / "data"

# Load the HTML content
with open(data_dir / "club-analyzer.html", encoding="utf-8") as f:
    soup = BeautifulSoup(f, "html.parser")

# Parse the table rows
rows = soup.find_all("tr")

# Get the header to determine column indices
headers = [th.text.strip() for th in rows[0].find_all("th")]
id_idx = headers.index("Id")
location_idx = headers.index("Location")

# Extract IDs for CLUB entries
club_ids = []
for row in rows[1:]:
    cols = row.find_all("td")
    if cols and cols[location_idx].text.strip() == "CLUB":
        club_ids.append(cols[id_idx].text.strip())

# Save to JSON file
with open(data_dir / "club_ids.json", "w") as out:
    json.dump(club_ids, out, indent=2)

print("Extracted CLUB Player IDs and saved them to club_ids.json")