# build_gg_training_dataset.py

import json
from pathlib import Path
import pandas as pd
from collections import defaultdict

# --- Configuration ---
BASE_DATA_DIR = Path(__file__).resolve().parent / '../data'
RAW_DATA_DIR = BASE_DATA_DIR / 'raw'
GG_DATA_DIR = RAW_DATA_DIR / 'ggData'
GG_META_DIR = RAW_DATA_DIR / 'ggMeta'
MAPS_FILE = BASE_DATA_DIR / 'maps.json'
MANUAL_DATA_FILE = BASE_DATA_DIR / 'manual_gg_data.csv'
OVERSAMPLE_FACTOR = 50 # Duplicate manual entries to give them more weight
OUTPUT_GG_FILE = BASE_DATA_DIR / 'training_dataset_gg.csv'

# --- Helper Functions (unchanged) ---
def load_json_file(file_path, default_val=None):
    try:
        with open(file_path, 'r', encoding='utf-8') as f: return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError): return default_val

def calculate_acceleration_type(accel, agility, strength, height):
    if not all(isinstance(x, (int, float)) and x is not None for x in [accel, agility, strength, height]): return "CONTROLLED"
    accel, agility, strength, height = int(accel), int(agility), int(strength), int(height)
    if (agility - strength) >= 20 and accel >= 80 and height <= 175 and agility >= 80: return "EXPLOSIVE"
    if (agility - strength) >= 12 and accel >= 80 and height <= 182 and agility >= 70: return "MOSTLY_EXPLOSIVE"
    if (agility - strength) >= 4 and accel >= 70 and height <= 182 and agility >= 65: return "CONTROLLED_EXPLOSIVE"
    if (strength - agility) >= 20 and strength >= 80 and height >= 188 and accel >= 55: return "LENGTHY"
    if (strength - agility) >= 12 and strength >= 75 and height >= 183 and accel >= 55: return "MOSTLY_LENGTHY"
    if (strength - agility) >= 4 and strength >= 65 and height >= 181 and accel >= 40: return "CONTROLLED_LENGTHY"
    return "CONTROLLED"

def get_attribute_with_boost(base_attributes, attr_name, boost_modifiers, default_val=0):
    base_val = base_attributes.get(attr_name, default_val)
    boost_val = boost_modifiers.get(attr_name, 0)
    if not isinstance(base_val, (int, float)): base_val = default_val if default_val is not None else 0
    if not isinstance(boost_val, (int, float)): boost_val = 0
    if base_val is None: base_val = 0
    return min(int(base_val) + int(boost_val), 99)

def main():
    print("🚀 Starting script to build the ggMeta training dataset...")
    maps = load_json_file(MAPS_FILE)
    if not maps:
        print(f"❌ Critical error: Could not load {MAPS_FILE}. Exiting.")
        return
    
    chem_style_boosts = {item['name'].lower(): item['threeChemistryModifiers'] for item in maps.get("ChemistryStylesBoosts", []) if 'name' in item}
    all_playstyles = list(maps.get("playstyles", {}).values())

    all_rows = []
    player_ids = [f.stem.split('_')[0] for f in GG_DATA_DIR.glob("*_ggData.json")]
    print(f"ℹ️ Found {len(player_ids)} players in raw data directory.")

    for i, ea_id_str in enumerate(player_ids):
        if (i + 1) % 100 == 0: print(f"⏳ Processing player {i + 1}/{len(player_ids)}...")
        gg_data = load_json_file(GG_DATA_DIR / f"{ea_id_str}_ggData.json")
        gg_meta = load_json_file(GG_META_DIR / f"{ea_id_str}_ggMeta.json")
        if not gg_data or "data" not in gg_data or not gg_meta or "data" not in gg_meta or "scores" not in gg_meta["data"]: continue
        
        player_def = gg_data["data"]
        base_attributes = {k: v for k, v in player_def.items() if k.startswith("attribute")}
        base_features = {
            "height": player_def.get("height"), "weight": player_def.get("weight"),
            "skillMoves": player_def.get("skillMoves"), "weakFoot": player_def.get("weakFoot"),
            "bodytype": maps.get("bodytypeCode", {}).get(str(player_def.get("bodytypeCode"))),
            "foot": maps.get("foot", {}).get(str(player_def.get("foot")))
        }
        
        player_ps = {maps.get("playstyles", {}).get(str(p)) for p in player_def.get("playstyles", [])}
        player_ps_plus = {maps.get("playstyles", {}).get(str(p)) for p in player_def.get("playstylesPlus", [])}
        for ps in all_playstyles:
            base_features[ps] = 2 if ps in player_ps_plus else 1 if ps in player_ps else 0

        for score_entry in gg_meta["data"]["scores"]:
            role_id = str(score_entry.get("role"))
            chem_id = str(score_entry.get("chemistryStyle"))
            role_name = maps.get("role", {}).get(role_id)
            chem_style_name = maps.get("ggChemistryStyleNames", {}).get(chem_id, "basic").lower()
            if not role_name: continue
            
            row = base_features.copy()
            row["role"] = role_name
            boosts = chem_style_boosts.get(chem_style_name, {})
            for attr in base_attributes:
                row[attr] = get_attribute_with_boost(base_attributes, attr, boosts)
            row["accelerateType"] = calculate_acceleration_type(row.get("attributeAcceleration"), row.get("attributeAgility"), row.get("attributeStrength"), row.get("height"))
            row["target_ggMeta"] = score_entry.get("score")
            all_rows.append(row)

    print("✅ Finished processing raw player data. Creating DataFrame...")
    df_large = pd.DataFrame(all_rows)

    if MANUAL_DATA_FILE.exists():
        print(f"ℹ️ Found manual data file at {MANUAL_DATA_FILE.name}. Oversampling and merging...")
        try:
            df_manual = pd.read_csv(MANUAL_DATA_FILE)
            df_manual_oversampled = pd.concat([df_manual] * OVERSAMPLE_FACTOR, ignore_index=True)
            df_final = pd.concat([df_large, df_manual_oversampled], ignore_index=True)
            print(f"✅ Merged {len(df_large)} auto-generated rows with {len(df_manual_oversampled)} oversampled manual rows.")
        except Exception as e:
            print(f"⚠️ Could not process manual data file, using auto-generated data only. Error: {e}")
            df_final = df_large
    else:
        print("ℹ️ No manual data file found. Using auto-generated data only.")
        df_final = df_large

    df_final.dropna(subset=['target_ggMeta'], inplace=True)
    df_final.fillna(0, inplace=True)
    df_final = pd.get_dummies(df_final, columns=["role", "bodytype", "accelerateType", "foot"], dtype=int)
    
    df_final.to_csv(OUTPUT_GG_FILE, index=False)
    print(f"💾 Saved combined ggMeta training data with {len(df_final)} rows to {OUTPUT_GG_FILE.name}")

if __name__ == "__main__":
    main()