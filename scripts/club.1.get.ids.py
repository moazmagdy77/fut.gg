from bs4 import BeautifulSoup
import json
from pathlib import Path

# Define data directory - assuming the script is in a 'scripts' folder 
# and the data is in a parallel 'data' folder.
# Adjust the path if your directory structure is different.
try:
    data_dir = Path(__file__).resolve().parent.parent / "data"
except NameError:
    # Fallback for environments where __file__ is not defined (e.g., Jupyter)
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
# Get the index for the 'Rating' column
rating_idx = headers.index("Rating")

# Extract IDs for CLUB entries with a rating of 90 or higher
club_ids = []
for row in rows[1:]:
    cols = row.find_all("td")
    # Check if the row has columns and the location is 'CLUB'
    if cols and cols[location_idx].text.strip() == "CLUB":
        # Get the rating and convert it to an integer for comparison
        rating = int(cols[rating_idx].text.strip())
        # Process only players with a rating of 90 or greater
        if rating >= 0:
            club_ids.append(cols[id_idx].text.strip())

# Save to JSON file
output_file_path = data_dir / "club_ids.json"
with open(output_file_path, "w") as out:
    json.dump(club_ids, out, indent=2)

print(f"Extracted {len(club_ids)} CLUB Player IDs with rating >= 90 and saved them to {output_file_path}")