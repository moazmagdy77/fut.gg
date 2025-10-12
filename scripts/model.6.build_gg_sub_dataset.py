# model.6b.build_gg_sub_dataset.py
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
OUTPUT_FILE = BASE_DATA_DIR / 'training_dataset_gg_sub.csv'

# --- Helpers ---
def load_json_file(file_path, default_val=None):
    try:
        with open(file_path, 'r', encoding='utf-8') as f: return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError): return default_val

def _normalize_gender(gender_val, maps):
    try:
        g = (maps.get("gender") or {}).get(str(gender_val))
        if isinstance(g, str) and g: return g
    except Exception:
        pass
    return "Male"

def calculate_acceleration_type(accel, agility, strength, height, gender: str = "Male"):
    # Gender-aware rules from your latest game update
    try:
        if accel is None or agility is None or strength is None or height is None:
            return "CONTROLLED"
        accel = int(accel); agility = int(agility); strength = int(strength); height = int(height)
    except Exception:
        return "CONTROLLED"
    is_female = str(gender or "Male").lower().startswith("f")
    exp_height_ok = (height <= 162) if is_female else (height <= 182)
    len_height_ok = (height >= 164) if is_female else (height >= 183)
    if (agility >= 65 and (agility - strength) >= 10 and accel >= 80 and exp_height_ok):
        return "EXPLOSIVE"
    if (strength >= 65 and (strength - agility) >= 4 and accel >= 40 and len_height_ok):
        return "LENGTHY"
    return "CONTROLLED"

def get_attribute_with_boost(base_attributes, attr_name, boost_modifiers, default_val=0):
    # For ggMetaSub we use NO BOOSTS (chem = Basic), but leave this utility here for parity
    base_val = base_attributes.get(attr_name, default_val)
    try:
        base_val = int(base_val) if base_val is not None else 0
    except Exception:
        base_val = 0
    return min(base_val, 99)

def familiarity_for_role(role_name, player_def, maps):
    rp = maps.get("rolesPlus", {}); rpp = maps.get("rolesPlusPlus", {})
    plus_names = {rp.get(str(x)) for x in (player_def.get("rolesPlus") or [])}
    pp_names   = {rpp.get(str(x)) for x in (player_def.get("rolesPlusPlus") or [])}
    plus_names = {n for n in plus_names if n}; pp_names = {n for n in pp_names if n}
    if role_name in pp_names: return 2
    if role_name in plus_names: return 1
    return 0

def main():
    print("🚀 Building ggMetaSub (Basic) training dataset...")
    maps = load_json_file(MAPS_FILE)
    if not maps:
        print(f"❌ Critical error: Could not load {MAPS_FILE}. Exiting.")
        return

    all_playstyles = list((maps.get("playstyles") or {}).values())
    foot_map = maps.get("foot", {})

    rows = []
    player_ids = [f.stem.split('_')[0] for f in GG_DATA_DIR.glob("*_ggData.json")]
    print(f"ℹ️ Found {len(player_ids)} players to process.")

    for i, ea_id_str in enumerate(player_ids):
        if (i + 1) % 100 == 0: print(f"⏳ {i+1}/{len(player_ids)}")

        gg_data = load_json_file(GG_DATA_DIR / f"{ea_id_str}_ggData.json")
        gg_meta = load_json_file(GG_META_DIR / f"{ea_id_str}_ggMeta.json")
        if not gg_data or "data" not in gg_data or not gg_meta or "data" not in gg_meta or "scores" not in gg_meta["data"]:
            continue

        player_def = gg_data["data"]
        base_attributes = {k: v for k, v in player_def.items() if k.startswith("attribute")}
        gender_text = _normalize_gender(player_def.get("gender"), maps)

        base_features = {
            "height": player_def.get("height"),
            "weight": player_def.get("weight"),
            "skillMoves": player_def.get("skillMoves"),
            "weakFoot": player_def.get("weakFoot"),
            "bodytype": maps.get("bodytypeCode", {}).get(str(player_def.get("bodytypeCode"))),
            "foot": foot_map.get(str(player_def.get("foot")))
            # NOTE: do NOT include a 'gender' feature column
        }

        # playstyles 0/1/2
        ps_map = maps.get("playstyles", {}) or {}
        ps = {ps_map.get(str(p)) for p in (player_def.get("playstyles") or [])}
        ps_plus = {ps_map.get(str(p)) for p in (player_def.get("playstylesPlus") or [])}
        for name in all_playstyles:
            base_features[name] = 2 if name in ps_plus else (1 if name in ps else 0)

        # group GG scores by role and fetch 'Basic' only
        by_role = defaultdict(list)
        for s in gg_meta["data"]["scores"]:
            by_role[str(s.get("role"))].append(s)

        for role_id_str, scores in by_role.items():
            role_name = (maps.get("role") or {}).get(role_id_str)
            if not role_name: continue
            basic_row = next(
                (s for s in scores if ((maps.get("ggChemistryStyleNames") or {}).get(str(s.get("chemistryStyle")), "") or "").lower() == "basic"),
                None
            )
            if not basic_row or basic_row.get("score") is None: continue
            target_basic = float(basic_row["score"])

            # assemble training row (NO boosts)
            row = base_features.copy()
            row["role"] = role_name
            row["familiarity"] = familiarity_for_role(role_name, player_def, maps)
            # add raw attributes
            for attr, v in base_attributes.items():
                row[attr] = get_attribute_with_boost(base_attributes, attr, None)
            # gender-aware accelerateType at 0 chem
            row["accelerateType"] = calculate_acceleration_type(
                row.get("attributeAcceleration"), row.get("attributeAgility"),
                row.get("attributeStrength"), row.get("height"), gender_text
            )
            row["target_ggMetaSub"] = target_basic
            rows.append(row)

    print("✅ Finished building rows. Creating DataFrame...")
    df = pd.DataFrame(rows)
    if df.empty:
        print("⚠️ No rows created. Check inputs.")
        return

    # ensure no stray non-numeric text columns other than those we one-hot
    if "gender" in df.columns:
        df.drop(columns=["gender"], inplace=True)

    df = pd.get_dummies(df, columns=["role", "bodytype", "accelerateType", "foot"], dtype=int)
    df.dropna(subset=["target_ggMetaSub"], inplace=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"💾 Saved ggMetaSub training data with {len(df)} rows to {OUTPUT_FILE.name}")

if __name__ == "__main__":
    main()
