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

try: name_idx = headers.index("Name")
except ValueError: name_idx = -1

try: rarity_idx = headers.index("Rarity")
except ValueError: rarity_idx = -1

try: position_idx = headers.index("Position")
except ValueError: position_idx = -1

try: discard_idx = headers.index("Discard Value")
except ValueError: discard_idx = -1

# Extract IDs
club_ids = []
tradeable_ids = []
tradeable_details = []

for row in rows[1:]:
    cols = row.find_all("td")
    if cols and cols[location_idx].text.strip() == "CLUB":
        try:
            rating = int(cols[rating_idx].text.strip())
        except (ValueError, IndexError):
            rating = 0
            
        ea_id = cols[id_idx].text.strip()
        if rating >= 75:
            club_ids.append(ea_id)
            
        # Check if Untradeable is False
        untradeable_text = cols[untradeable_idx].text.strip()
        # Checks for "False" string or empty/falsy values depending on HTML format
        if untradeable_text.lower() == "false":
            tradeable_ids.append(ea_id)
            
            tradeable_details.append({
                "__true_player_id": ea_id,
                "commonName": cols[name_idx].text.strip() if name_idx != -1 else "Unknown",
                "overall": rating,
                "rarity": cols[rarity_idx].text.strip() if rarity_idx != -1 else "Unknown",
                "positions": cols[position_idx].text.strip() if position_idx != -1 else "Unknown",
                "avgMeta": 0.0,
                "isExtinct": False,
                "price": 0,
                "discardValue": int(cols[discard_idx].text.strip().replace(",", "").replace(".", "")) if discard_idx != -1 and cols[discard_idx].text.strip().replace(",", "").replace(".", "").isdigit() else 0
            })

# Save CLUB IDs (Main list for data fetching)
output_file_path = data_dir / "club_ids.json"
with open(output_file_path, "w") as out:
    json.dump(club_ids, out, indent=2)

# Save TRADEABLE IDs (Subset for price fetching)
tradeable_file_path = data_dir / "tradeable_ids.json"
with open(tradeable_file_path, "w") as out:
    json.dump(tradeable_ids, out, indent=2)

# Save TRADEABLE Details
tradeable_details_path = data_dir / "tradeable_details.json"
with open(tradeable_details_path, "w", encoding="utf-8") as out:
    json.dump(tradeable_details, out, indent=2)

print(f"Extracted {len(club_ids)} CLUB Player IDs -> {output_file_path}")
print(f"Extracted {len(tradeable_ids)} Tradeable Player IDs -> {tradeable_file_path}")
print(f"Extracted {len(tradeable_details)} Tradeable Player Details -> {tradeable_details_path}")