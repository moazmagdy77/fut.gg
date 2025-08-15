# 2-parse_r_objects.py

import json
from pathlib import Path
import os

# --- Configuration ---
BASE_DATA_DIR = Path(__file__).resolve().parent / '../data'
RAW_DATA_DIR = BASE_DATA_DIR / 'raw'
R_OBJECTS_DIR = RAW_DATA_DIR / 'r_objects'
OUTPUT_FILE = BASE_DATA_DIR / 'player_ids.json'

def main():
    print("🚀 Starting parser for raw R objects...")

    if not R_OBJECTS_DIR.exists():
        print(f"❌ Error: Directory not found: {R_OBJECTS_DIR}")
        return

    all_player_ids = set()
    
    if OUTPUT_FILE.exists():
        try:
            with open(OUTPUT_FILE, 'r') as f:
                existing_ids = json.load(f)
                all_player_ids.update(existing_ids)
            print(f"📜 Found {len(all_player_ids)} existing IDs in output file.")
        except json.JSONDecodeError:
            print(f"⚠️ Could not read existing output file. Starting fresh.")


    json_files = sorted(R_OBJECTS_DIR.glob('page_*.json'))
    print(f"ℹ️ Found {len(json_files)} raw data files to process.")

    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_json_object = json.load(f)

            # --- FIX: Extract the array from the 'tsr' key ---
            if not isinstance(raw_json_object, dict) or 'tsr' not in raw_json_object:
                print(f"⚠️ Skipping {file_path.name}, expected a dictionary with a 'tsr' key.")
                continue
            
            r_array = raw_json_object['tsr'] # Get the list from the object
            
            if not isinstance(r_array, list):
                print(f"⚠️ Skipping {file_path.name}, the 'tsr' key did not contain a list.")
                continue
            # --- END FIX ---

            # Find the dictionary inside the list that contains the player data
            player_data_block = next((item for item in r_array if item and 'l' in item and 'playerItems' in item['l']), None)

            if player_data_block:
                players = player_data_block.get('l', {}).get('playerItems', {}).get('data', [])
                if players:
                    ids_found = {player['eaId'] for player in players if 'eaId' in player}
                    all_player_ids.update(ids_found)
                    print(f"✅ Parsed {file_path.name}, found {len(ids_found)} IDs. Total unique IDs: {len(all_player_ids)}")
                else:
                    print(f"⚠️ No players in data block for {file_path.name}")
            else:
                print(f"⚠️ No player data block found in {file_path.name}")

        except Exception as e:
            print(f"❌ Error processing {file_path.name}: {e}")

    # Save the final unique list of IDs
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(sorted(list(all_player_ids)), f, indent=2)

    print(f"\n🎉 Finished! Saved {len(all_player_ids)} unique player IDs to {OUTPUT_FILE.name}.")

if __name__ == "__main__":
    main()