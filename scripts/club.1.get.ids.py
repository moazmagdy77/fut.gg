# club.1.get.ids.py
# Extracts CLUB player IDs, ALL CLUB IDs, and Tradeable IDs from club-analyzer.html.
# Optimized: Uses regex instead of BeautifulSoup to run in <0.5 seconds on Windows.

import json
import re
import sys
import io
from pathlib import Path

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
except Exception:
    pass

def main():
    # Define data directory
    try:
        data_dir = Path(__file__).resolve().parent.parent / "data"
    except NameError:
        data_dir = Path(".").resolve().parent / "data"
        if not data_dir.exists():
            data_dir = Path(".").resolve()

    # Load the HTML content
    html_file_path = data_dir / "club-analyzer.html"
    if not html_file_path.exists():
        print(f"❌ Error: {html_file_path.name} not found. Please place it in data/ folder first.")
        return

    print(f"📖 Reading and parsing {html_file_path.name} using regex...")
    try:
        html_content = html_file_path.read_text(encoding="utf-8")
        rows = re.findall(r'<tr>(.*?)</tr>', html_content, re.DOTALL)
    except Exception as e:
        print(f"❌ Error reading HTML file: {e}")
        return

    if not rows:
        print("❌ Error: No rows found in HTML table.")
        return

    # Get headers using regex
    headers = [re.sub(r'<.*?>', '', h).strip() for h in re.findall(r'<th.*?>(.*?)</th>', rows[0], re.DOTALL)]
    try:
        id_idx = headers.index("Id")
        location_idx = headers.index("Location")
        rating_idx = headers.index("Rating")
    except ValueError as e:
        print(f"❌ Error finding table headers: {e}")
        return

    # Attempt to find optional columns, default if not found
    try:
        untradeable_idx = headers.index("Untradeable")
    except ValueError:
        untradeable_idx = -1  # header absent -> skip tradeable extraction (don't guess a column)

    try: name_idx = headers.index("Name")
    except ValueError: name_idx = -1

    try: lastname_idx = headers.index("Lastname")
    except ValueError: lastname_idx = -1

    try: rarity_idx = headers.index("Rarity")
    except ValueError: rarity_idx = -1

    try: position_idx = headers.index("Position")
    except ValueError: position_idx = -1

    try: discard_idx = headers.index("Discard Value")
    except ValueError: discard_idx = -1

    # Extract IDs
    club_ids = []
    all_club_ids = []
    tradeable_ids = []
    tradeable_details = []

    for row in rows[1:]:
        cols = [re.sub(r'<.*?>', '', c).strip() for c in re.findall(r'<td.*?>(.*?)</td>', row, re.DOTALL)]
        if cols and len(cols) > max(id_idx, location_idx, rating_idx):
            if cols[location_idx] == "CLUB":
                try:
                    rating = int(cols[rating_idx])
                except ValueError:
                    rating = 0
                    
                ea_id = cols[id_idx]
                all_club_ids.append(ea_id)
                if rating >= 75:
                    club_ids.append(ea_id)
                    
                # Check if Untradeable is False (skip entirely if the column is absent)
                if 0 <= untradeable_idx < len(cols):
                    untradeable_text = cols[untradeable_idx]
                    if untradeable_text.lower() == "false":
                        tradeable_ids.append(ea_id)
                        
                        # Discard Value extraction
                        discard_val = 0
                        if discard_idx != -1 and discard_idx < len(cols):
                            discard_text = cols[discard_idx].replace(",", "").replace(".", "")
                            if discard_text.isdigit():
                                discard_val = int(discard_text)

                        # Name resolution
                        first = cols[name_idx] if name_idx != -1 and name_idx < len(cols) else ""
                        last = cols[lastname_idx] if lastname_idx != -1 and lastname_idx < len(cols) else ""
                        name = f"{first} {last}".strip() if first or last else "Unknown"

                        tradeable_details.append({
                            "__true_player_id": ea_id,
                            "commonName": name,
                            "overall": rating,
                            "rarity": cols[rarity_idx] if rarity_idx != -1 and rarity_idx < len(cols) else "Unknown",
                            "positions": cols[position_idx] if position_idx != -1 and position_idx < len(cols) else "Unknown",
                            "avgMeta": 0.0,
                            "isExtinct": False,
                            "price": 0,
                            "discardValue": discard_val
                        })

    # Save CLUB IDs (Main list for data fetching, rating >= 75)
    output_file_path = data_dir / "club_ids.json"
    with open(output_file_path, "w") as out:
        json.dump(club_ids, out, indent=2)

    # Save ALL CLUB IDs (No rating filter, used for tall-player identification)
    all_club_ids_path = data_dir / "all_club_ids.json"
    with open(all_club_ids_path, "w") as out:
        json.dump(all_club_ids, out, indent=2)

    # Save TRADEABLE IDs (Subset for price fetching)
    tradeable_file_path = data_dir / "tradeable_ids.json"
    with open(tradeable_file_path, "w") as out:
        json.dump(tradeable_ids, out, indent=2)

    # Save TRADEABLE Details
    tradeable_details_path = data_dir / "tradeable_details.json"
    with open(tradeable_details_path, "w", encoding="utf-8") as out:
        json.dump(tradeable_details, out, indent=2)

    print(f"Extracted {len(club_ids)} CLUB Player IDs (≥75 OVR) -> {output_file_path.name}")
    print(f"Extracted {len(all_club_ids)} ALL CLUB Player IDs -> {all_club_ids_path.name}")
    print(f"Extracted {len(tradeable_ids)} Tradeable Player IDs -> {tradeable_file_path.name}")
    print(f"Extracted {len(tradeable_details)} Tradeable Player Details -> {tradeable_details_path.name}")

if __name__ == "__main__":
    main()