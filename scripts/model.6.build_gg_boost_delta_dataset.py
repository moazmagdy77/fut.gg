# model.6.build_gg_boost_delta_dataset.py

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
OUTPUT_DELTA_FILE = BASE_DATA_DIR / 'training_dataset_gg_boost_delta.csv'
OUTPUT_SUB_FILE   = BASE_DATA_DIR / 'training_dataset_gg_sub.csv'

# --- Helpers ---
def load_json_file(file_path, default_val=None):
    try:
        with open(file_path, 'r', encoding='utf-8') as f: return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError): return default_val

def calculate_acceleration_type(accel, agility, strength, height):
    try:
        if accel is None or agility is None or strength is None or height is None:
            return "CONTROLLED"
        accel, agility, strength, height = int(accel), int(agility), int(strength), int(height)
    except Exception:
        return "CONTROLLED"
    if (agility - strength) >= 20 and accel >= 80 and height <= 175 and agility >= 80: return "EXPLOSIVE"
    if (agility - strength) >= 12 and accel >= 80 and height <= 182 and agility >= 70: return "MOSTLY_EXPLOSIVE"
    if (agility - strength) >= 4 and accel >= 70 and height <= 182 and agility >= 65: return "CONTROLLED_EXPLOSIVE"
    if (strength - agility) >= 20 and strength >= 80 and height >= 188 and accel >= 55: return "LENGTHY"
    if (strength - agility) >= 12 and strength >= 75 and height >= 183 and accel >= 55: return "MOSTLY_LENGTHY"
    if (strength - agility) >= 4 and strength >= 65 and height >= 181 and accel >= 40: return "CONTROLLED_LENGTHY"
    return "CONTROLLED"

def familiarity_for_role(role_name, player_def, maps):
    roles_plus_map = maps.get("rolesPlus", {})            # id -> name
    roles_pp_map   = maps.get("rolesPlusPlus", {})        # id -> name
    plus_names = {roles_plus_map.get(str(x)) for x in (player_def.get("rolesPlus") or [])}
    pp_names   = {roles_pp_map.get(str(x))   for x in (player_def.get("rolesPlusPlus") or [])}
    plus_names = {n for n in plus_names if n}
    pp_names   = {n for n in pp_names if n}
    if role_name in pp_names: return 2
    if role_name in plus_names: return 1
    return 0

def main():
    print("🚀 Building gg datasets (boost-delta + sub)...")
    maps = load_json_file(MAPS_FILE)
    if not maps:
        print(f"❌ Critical error: Could not load {MAPS_FILE}. Exiting.")
        return

    role_id_to_name = maps.get("role", {})
    foot_map = maps.get("foot", {})
    all_playstyles = list(maps.get("playstyles", {}).values())

    rows_delta = []
    rows_sub   = []

    player_ids = [f.stem.split('_')[0] for f in GG_DATA_DIR.glob("*_ggData.json")]
    print(f"ℹ️ Found {len(player_ids)} players to process.")

    for i, ea_id_str in enumerate(player_ids):
        if (i + 1) % 100 == 0: print(f"⏳ Processing player {i + 1}/{len(player_ids)}...")

        gg_data  = load_json_file(GG_DATA_DIR / f"{ea_id_str}_ggData.json")
        gg_meta  = load_json_file(GG_META_DIR / f"{ea_id_str}_ggMeta.json")
        if not gg_data or "data" not in gg_data or not gg_meta or "data" not in gg_meta or "scores" not in gg_meta["data"]:
            continue

        player_def = gg_data["data"]
        base_attributes = {k: v for k, v in player_def.items() if k.startswith("attribute")}

        base_features = {
            "height": player_def.get("height"),
            "weight": player_def.get("weight"),
            "skillMoves": player_def.get("skillMoves"),
            "weakFoot": player_def.get("weakFoot"),
            "bodytype": maps.get("bodytypeCode", {}).get(str(player_def.get("bodytypeCode"))),
            "foot": foot_map.get(str(player_def.get("foot")))
        }

        # Playstyles (0=no, 1=PS, 2=PS+)
        ps_map = maps.get("playstyles", {}) or {}
        player_ps      = {ps_map.get(str(p)) for p in (player_def.get("playstyles") or [])}
        player_ps_plus = {ps_map.get(str(p)) for p in (player_def.get("playstylesPlus") or [])}
        for ps in all_playstyles:
            base_features[ps] = 2 if ps in player_ps_plus else 1 if ps in player_ps else 0

        # Role → list of score entries
        scores_by_role = defaultdict(list)
        for s in gg_meta["data"]["scores"]:
            scores_by_role[str(s.get("role"))].append(s)

        for role_id_str, scores in scores_by_role.items():
            role_name = role_id_to_name.get(role_id_str)
            if not role_name: 
                continue

            # Familiarity + accelerateType (unboosted, for sub)
            fam = familiarity_for_role(role_name, player_def, maps)
            accel = calculate_acceleration_type(
                base_attributes.get("attributeAcceleration"),
                base_attributes.get("attributeAgility"),
                base_attributes.get("attributeStrength"),
                base_features.get("height")
            )

            # ---- SUB dataset row (Basic at chem=0 conceptually) ----
            basic_entry = next(
                (s for s in scores if (maps.get("ggChemistryStyleNames", {}) or {}).get(str(s.get("chemistryStyle")), "").lower() == "basic"),
                None
            )
            if basic_entry and basic_entry.get("score") is not None:
                row_sub = base_features.copy()
                row_sub.update(base_attributes)
                row_sub["role"] = role_name
                row_sub["accelerateType"] = accel
                row_sub["familiarity"] = fam
                row_sub["target_ggMetaSub"] = float(basic_entry["score"])
                rows_sub.append(row_sub)

            # ---- BOOST-DELTA dataset rows (difference from Basic) ----
            if not scores or not basic_entry or basic_entry.get("score") is None:
                continue
            basic_score = float(basic_entry["score"])
            row_base = base_features.copy()
            row_base.update(base_attributes)
            row_base["role"] = role_name
            for s in scores:
                chem_id = str(s.get("chemistryStyle"))
                chem_style_name = (maps.get("ggChemistryStyleNames", {}) or {}).get(chem_id, "basic").lower()
                final_row = row_base.copy()
                final_row["chem_style"] = chem_style_name
                final_row["target_ggMetaBoostDelta"] = float(s.get("score", basic_score)) - basic_score
                rows_delta.append(final_row)

    # -- Save SUB dataset
    if rows_sub:
        df_sub = pd.DataFrame(rows_sub).dropna(subset=["target_ggMetaSub"]).fillna(0)
        df_sub = pd.get_dummies(df_sub, columns=["role", "bodytype", "foot", "accelerateType"], dtype=int)
        df_sub.to_csv(OUTPUT_SUB_FILE, index=False)
        print(f"💾 Saved ggMeta SUB training data with {len(df_sub)} rows to {OUTPUT_SUB_FILE.name}")
    else:
        print("⚠️ No ggMeta SUB rows produced (check inputs).")

    # -- Save DELTA dataset
    if rows_delta:
        df_delta = pd.DataFrame(rows_delta).dropna(subset=["target_ggMetaBoostDelta"]).fillna(0)
        df_delta = pd.get_dummies(df_delta, columns=["role", "bodytype", "foot", "chem_style"], dtype=int)
        df_delta.to_csv(OUTPUT_DELTA_FILE, index=False)
        print(f"💾 Saved ggMeta BOOST DELTA training data with {len(df_delta)} rows to {OUTPUT_DELTA_FILE.name}")
    else:
        print("⚠️ No ggMeta BOOST DELTA rows produced (check inputs).")

if __name__ == "__main__":
    main()
