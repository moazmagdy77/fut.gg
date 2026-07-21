# invalidate_upgraded_players.py
# Compares overall ratings in club-analyzer.html with saved raw data.
# Deletes player files if they got upgraded/evolved, so the next pipeline run refetches them.
# Optimized: Uses all_players_summary.json as a single in-memory index to avoid thousands of disk I/O operations.
# Super-Optimized: Uses regex-based HTML parsing to avoid BeautifulSoup overhead, running in <0.5 seconds.

import json
import sys
import io
import re
from pathlib import Path
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
    summary_file = data_dir / "all_players_summary.json"

    if not html_file.exists():
        print(f"❌ Error: {html_file.name} not found. Please place it in data/ folder first.")
        return

    # 1. Parse HTML to get actual club ratings using ultra-fast regex
    print(f"📖 Parsing {html_file.name} using regex...")
    try:
        html_content = html_file.read_text(encoding="utf-8")
        rows = re.findall(r'<tr>(.*?)</tr>', html_content, re.DOTALL)
    except Exception as e:
        print(f"❌ Error reading HTML file: {e}")
        return

    if not rows:
        print("❌ Error: No rows found in the HTML table.")
        return

    # Extract headers
    headers = [re.sub(r'<.*?>', '', h).strip() for h in re.findall(r'<th.*?>(.*?)</th>', rows[0], re.DOTALL)]
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
        cols = [re.sub(r'<.*?>', '', c).strip() for c in re.findall(r'<td.*?>(.*?)</td>', row, re.DOTALL)]
        if cols and len(cols) > max(id_idx, rating_idx, location_idx):
            if cols[location_idx] == "CLUB":
                ea_id = cols[id_idx]
                name = cols[name_idx] if name_idx != -1 and name_idx < len(cols) else f"ID: {ea_id}"
                try:
                    rating = int(cols[rating_idx])
                except ValueError:
                    rating = 0
                if ea_id:
                    club_players[ea_id] = (rating, name)

    print(f"ℹ️ Found {len(club_players)} players in CLUB.")

    # 2. Build saved ratings index (try all_players_summary.json first for speed)
    saved_ratings = {}
    use_fallback = True

    if summary_file.exists():
        print(f"📖 Loading in-memory summary index from {summary_file.name}...")
        summary_data = load_json_file(summary_file)
        if summary_data and isinstance(summary_data, list):
            for p in summary_data:
                ea_id_val = p.get("eaId")
                ovr_val = p.get("overall")
                if ea_id_val is not None and ovr_val is not None:
                    saved_ratings[str(ea_id_val)] = int(ovr_val)
            use_fallback = False
            print(f"ℹ️ Indexed {len(saved_ratings)} players from summary file.")

    categories = ['club - main', 'club - rest', 'training']
    deleted_count = 0

    def raw_saved_rating(ea_id):
        """Authoritative overall rating read from the raw ggData files (the data
        that actually gets refetched). Returns None if no raw file exists yet."""
        for cat in categories:
            gg_data_path = raw_dir / cat / "ggData" / f"{ea_id}_ggData.json"
            if gg_data_path.exists():
                gg_raw = load_json_file(gg_data_path)
                if gg_raw and "data" in gg_raw:
                    ovr = gg_raw["data"].get("overall")
                    if ovr is not None:
                        try:
                            return int(ovr)
                        except (ValueError, TypeError):
                            return None
        return None

    # 3. Compare ratings and delete mismatching files
    for ea_id, (club_rating, name) in club_players.items():
        # Fast pre-filter via the summary index (may be stale between retrains,
        # since the club pipeline never rebuilds it). Only used to cheaply skip
        # the vast majority of players that clearly haven't changed.
        if not use_fallback:
            summary_rating = saved_ratings.get(ea_id)
            # Only skip when the summary clearly shows no change. If the player is absent
            # from the (possibly stale) summary index, fall through to the authoritative
            # raw check — skipping here missed players fetched since the last summary rebuild.
            if summary_rating is not None and summary_rating == club_rating:
                continue

        # Authoritative check against the raw ggData that would be refetched.
        # This prevents re-deleting/re-fetching players whose raw files are
        # already up to date when the summary index is stale.
        saved_rating = raw_saved_rating(ea_id)
        if saved_rating is None or saved_rating == club_rating:
            continue

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
        print("👉 The next pipeline step will fetch their fresh data and update prices!")
    else:
        print("✅ No upgraded/mismatched players found. All local files are up to date!")

if __name__ == "__main__":
    main()
