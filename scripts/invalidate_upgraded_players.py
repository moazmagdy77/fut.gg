# invalidate_upgraded_players.py
# Compares overall ratings in club-analyzer.html with saved raw data.
# Deletes player files if they got upgraded/evolved, so the next pipeline run refetches them.

import json
import sys
import io
from pathlib import Path
from bs4 import BeautifulSoup
from shared_utils import load_json_file

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
except Exception:
    pass

def main():
    print("🧹 Starting Invalidator for Upgraded Players...")
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir.parent / "data"
    raw_dir = data_dir / "raw"
    html_file = data_dir / "club-analyzer.html"

    if not html_file.exists():
        print(f"❌ Error: {html_file.name} not found. Please place it in data/ folder first.")
        return

    # 1. Parse HTML to get actual club ratings
    print(f"📖 Parsing {html_file.name}...")
    try:
        with open(html_file, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
    except Exception as e:
        print(f"❌ Error reading HTML file: {e}")
        return

    rows = soup.find_all("tr")
    if not rows:
        print("❌ Error: No rows found in the HTML table.")
        return

    headers = [th.text.strip() for th in rows[0].find_all("th")]
    try:
        id_idx = headers.index("Id")
        rating_idx = headers.index("Rating")
        location_idx = headers.index("Location")
    except ValueError as e:
        print(f"❌ Error finding table headers: {e}")
        return

    try:
        name_idx = headers.index("Name")
    except ValueError:
        name_idx = -1

    # Map actual club player ea_id -> (rating, name)
    club_players = {}
    for row in rows[1:]:
        cols = row.find_all("td")
        if cols and cols[location_idx].text.strip() == "CLUB":
            ea_id = cols[id_idx].text.strip()
            name = cols[name_idx].text.strip() if name_idx != -1 else f"ID: {ea_id}"
            try:
                rating = int(cols[rating_idx].text.strip())
            except ValueError:
                rating = 0
            if ea_id:
                club_players[ea_id] = (rating, name)

    print(f"ℹ️ Found {len(club_players)} players in CLUB.")

    # 2. Check saved raw files and delete if rating mismatch
    categories = ['club - main', 'club - rest', 'training']
    deleted_count = 0

    for ea_id, (club_rating, name) in club_players.items():
        mismatch_detected = False
        saved_rating = None

        # Check across the categories to find existing files
        for cat in categories:
            gg_data_path = raw_dir / cat / "ggData" / f"{ea_id}_ggData.json"
            if gg_data_path.exists():
                gg_raw = load_json_file(gg_data_path)
                if gg_raw and "data" in gg_raw:
                    saved_rating = gg_raw["data"].get("overall")
                    if saved_rating is not None and int(saved_rating) != club_rating:
                        mismatch_detected = True
                        break

        if mismatch_detected:
            print(f"🔄 Upgraded Player Detected: {name} (ID: {ea_id})")
            print(f"   - Saved overall rating: {saved_rating}")
            print(f"   - Current club rating:  {club_rating}")
            print(f"   --> Deleting files to force refetch...")

            # Delete files from all categories
            for cat in categories:
                deleted_any = False
                for sub in ["ggData", "ggMeta", "esMeta"]:
                    file_to_del = raw_dir / cat / sub / f"{ea_id}_{sub}.json"
                    if file_to_del.exists():
                        try:
                            file_to_del.unlink()
                            deleted_any = True
                        except Exception as e:
                            print(f"      ⚠️ Failed to delete {file_to_del.name}: {e}")
                if deleted_any:
                    print(f"   Deleted files from folder: raw/{cat}/")
            deleted_count += 1

    print("\n--- Summary ---")
    if deleted_count > 0:
        print(f"✅ Invalidated/deleted files for {deleted_count} upgraded players.")
        print("👉 Run python scripts/pipeline.py to fetch their fresh data and update prices!")
    else:
        print("✅ No upgraded/mismatched players found in raw directories. All local files are up to date!")

if __name__ == "__main__":
    main()
