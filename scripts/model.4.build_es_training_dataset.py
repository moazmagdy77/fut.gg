import json
from pathlib import Path
import pandas as pd
from collections import defaultdict
import os

# --- Configuration ---
BASE_DATA_DIR = Path(__file__).resolve().parent / '../data'
RAW_DATA_DIR = BASE_DATA_DIR / 'raw'
GG_DATA_DIR = RAW_DATA_DIR / 'ggData'
ES_META_DIR = RAW_DATA_DIR / 'esMeta'
MAPS_FILE = BASE_DATA_DIR / 'maps.json'
OUTPUT_0_CHEM_FILE = BASE_DATA_DIR / 'training_dataset_0_chem.csv'
OUTPUT_3_CHEM_FILE = BASE_DATA_DIR / 'training_dataset_3_chem.csv'

# --- Helper Functions (from main processing script) ---
def load_json_file(file_path, default_val=None):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default_val

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

# --- Main Logic ---
def main():
    print("🚀 Starting script to build training datasets...")

    maps = load_json_file(MAPS_FILE)
    if not maps:
        print(f"❌ Critical error: Could not load {MAPS_FILE}. Exiting.")
        return
    
    chem_style_boosts = {item['name'].lower(): item['threeChemistryModifiers'] 
                         for item in maps.get("ChemistryStylesBoosts", []) if 'name' in item}
    all_playstyles = list(maps.get("playstyles", {}).values())

    rows_0_chem = []
    rows_3_chem = []

    player_ids = [f.stem.split('_')[0] for f in GG_DATA_DIR.glob("*_ggData.json")]
    print(f"ℹ️ Found {len(player_ids)} players in raw data directory.")

    for i, ea_id_str in enumerate(player_ids):
        if (i + 1) % 100 == 0:
            print(f"⏳ Processing player {i + 1}/{len(player_ids)}...")

        gg_data = load_json_file(GG_DATA_DIR / f"{ea_id_str}_ggData.json")
        es_meta = load_json_file(ES_META_DIR / f"{ea_id_str}_esMeta.json", [])

        if not gg_data or "data" not in gg_data or not es_meta:
            continue

        player_def = gg_data["data"]
        base_attributes = {k: v for k, v in player_def.items() if k.startswith("attribute")}
        
        # Create a base feature set for the player
        base_features = {
            "height": player_def.get("height"),
            "weight": player_def.get("weight"),
            "skillMoves": player_def.get("skillMoves"),
            "weakFoot": player_def.get("weakFoot"),
            "bodytype": maps.get("bodytypeCode", {}).get(str(player_def.get("bodytypeCode"))),
        }
        
        # Encode Playstyles (0=No, 1=PS, 2=PS+)
        player_ps = {maps.get("playstyles", {}).get(str(p)) for p in player_def.get("playstyles", [])}
        player_ps_plus = {maps.get("playstyles", {}).get(str(p)) for p in player_def.get("playstylesPlus", [])}
        for ps in all_playstyles:
            base_features[ps] = 2 if ps in player_ps_plus else 1 if ps in player_ps else 0

        # Determine all roles this player has from EasySBC data
        player_es_roles = {str(item['roleId']): item for item in es_meta}

        for es_role_id, role_data in player_es_roles.items():
            role_name = maps.get("esRoleId", {}).get(es_role_id)
            if not role_name or not role_data.get("data", {}).get("metaRatings"):
                continue

            es_ratings = role_data["data"]["metaRatings"]

            # --- Process 0-Chem Data ---
            rating_0_chem = next((r for r in es_ratings if r.get("chemistry") == 0 and str(r.get("playerRoleId")) == es_role_id), None)
            if rating_0_chem and rating_0_chem.get("metaRating") is not None:
                row_0 = base_features.copy()
                row_0["role"] = role_name
                row_0.update(base_attributes)
                row_0["accelerateType"] = calculate_acceleration_type(
                    row_0.get("attributeAcceleration"), row_0.get("attributeAgility"),
                    row_0.get("attributeStrength"), row_0.get("height")
                )
                row_0["target_esMetaSub"] = rating_0_chem["metaRating"]
                rows_0_chem.append(row_0)

            # --- Process 3-Chem Data ---
            ratings_3_chem = [r for r in es_ratings if r.get("chemistry") == 3 and str(r.get("playerRoleId")) == es_role_id]
            for rating in ratings_3_chem:
                chem_style_name = maps.get("esChemistryStyleNames", {}).get(str(rating.get("chemstyleId"))).lower()
                boosts = chem_style_boosts.get(chem_style_name, {})

                row_3 = base_features.copy()
                row_3["role"] = role_name
                # Add boosted attributes as features
                for attr in base_attributes:
                    row_3[attr] = get_attribute_with_boost(base_attributes, attr, boosts)
                
                row_3["accelerateType"] = calculate_acceleration_type(
                    row_3.get("attributeAcceleration"), row_3.get("attributeAgility"),
                    row_3.get("attributeStrength"), row_3.get("height")
                )
                row_3["target_esMeta"] = rating["metaRating"]
                rows_3_chem.append(row_3)

    print("✅ Finished processing all players. Creating DataFrames...")
    
    # Create and save 0-chem dataset
    df_0_chem = pd.DataFrame(rows_0_chem)
    df_0_chem = pd.get_dummies(df_0_chem, columns=["role", "bodytype", "accelerateType"], dtype=int)
    df_0_chem.dropna(subset=['target_esMetaSub'], inplace=True)
    df_0_chem.to_csv(OUTPUT_0_CHEM_FILE, index=False)
    print(f"💾 Saved 0-chem training data with {len(df_0_chem)} rows to {OUTPUT_0_CHEM_FILE.name}")

    # Create and save 3-chem dataset
    df_3_chem = pd.DataFrame(rows_3_chem)
    df_3_chem = pd.get_dummies(df_3_chem, columns=["role", "bodytype", "accelerateType"], dtype=int)
    df_3_chem.dropna(subset=['target_esMeta'], inplace=True)
    df_3_chem.to_csv(OUTPUT_3_CHEM_FILE, index=False)
    print(f"💾 Saved 3-chem training data with {len(df_3_chem)} rows to {OUTPUT_3_CHEM_FILE.name}")

if __name__ == "__main__":
    main()