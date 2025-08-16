# build_gg_delta_dataset.py

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
OUTPUT_FILE = BASE_DATA_DIR / 'training_dataset_gg_delta.csv'

# --- Helper Functions ---
def load_json_file(file_path, default_val=None):
    try:
        with open(file_path, 'r', encoding='utf-8') as f: return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError): return default_val

def main():
    print("🚀 Starting script to build the ggMeta DELTA training dataset...")
    maps = load_json_file(MAPS_FILE)
    if not maps:
        print(f"❌ Critical error: Could not load {MAPS_FILE}. Exiting.")
        return
    
    all_playstyles = list(maps.get("playstyles", {}).values())
    all_rows = []

    player_ids = [f.stem.split('_')[0] for f in GG_DATA_DIR.glob("*_ggData.json")]
    print(f"ℹ️ Found {len(player_ids)} players to process.")

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

        scores_by_role = defaultdict(list)
        for score_entry in gg_meta["data"]["scores"]:
            scores_by_role[str(score_entry.get("role"))].append(score_entry)

        for role_id_str, scores in scores_by_role.items():
            role_name = maps.get("role", {}).get(role_id_str)
            if not role_name: continue

            basic_score_entry = next((s for s in scores if maps["ggChemistryStyleNames"].get(str(s.get("chemistryStyle")), "").lower() == 'basic'), None)
            if not basic_score_entry or basic_score_entry.get("score") is None: continue
            basic_score = basic_score_entry["score"]
            
            for score_entry in scores:
                chem_id = str(score_entry.get("chemistryStyle"))
                chem_style_name = maps.get("ggChemistryStyleNames", {}).get(chem_id, "basic").lower()
                
                row = base_features.copy()
                row.update(base_attributes)
                row["role"] = role_name
                row["chem_style"] = chem_style_name
                
                # The target is the DIFFERENCE from the basic score
                row["target_ggMetaDelta"] = score_entry.get("score", basic_score) - basic_score
                all_rows.append(row)

    print("✅ Finished processing. Creating final DataFrame...")
    df = pd.DataFrame(all_rows)
    df.dropna(subset=['target_ggMetaDelta'], inplace=True)
    df.fillna(0, inplace=True)
    df = pd.get_dummies(df, columns=["role", "bodytype", "foot", "chem_style"], dtype=int)
    
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"💾 Saved ggMeta delta training data with {len(df)} rows to {OUTPUT_FILE.name}")

if __name__ == "__main__":
    main()