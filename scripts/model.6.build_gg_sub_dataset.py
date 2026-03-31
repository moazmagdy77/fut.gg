# model.6.build_gg_boost_delta_dataset.py
# Builds an ABSOLUTE ggMeta training dataset (not deltas).
# We learn ggMeta = f(boosted attributes, familiarity, playstyles, ...).
# At inference we pass "no-chem" (unboosted) features to predict ggMetaSub.

import json
from pathlib import Path
import pandas as pd
from collections import defaultdict
from shared_utils import load_json_file, _normalize_gender, calculate_acceleration_type, get_attribute_with_boost, familiarity_for_role

# --- Configuration ---
BASE_DATA_DIR = Path(__file__).resolve().parent / '../data'
RAW_DATA_DIR = BASE_DATA_DIR / 'raw'
GG_DATA_DIR = RAW_DATA_DIR / 'ggData'
GG_META_DIR = RAW_DATA_DIR / 'ggMeta'
MAPS_FILE = BASE_DATA_DIR / 'maps.json'
# New absolute dataset filename (we keep the script name for your runner)
OUTPUT_FILE = BASE_DATA_DIR / 'training_dataset_gg_sub_abs.csv'



def main():
    print("🚀 Building ABSOLUTE ggMeta training dataset for ggMetaSub...")

    maps = load_json_file(MAPS_FILE)
    if not maps:
        print(f"❌ Critical error: Could not load {MAPS_FILE}. Exiting.")
        return

    gg_style_names = maps.get("ggChemistryStyleNames", {}) or {}
    chem_style_boosts = {
        item['name'].lower(): item['threeChemistryModifiers']
        for item in maps.get("ChemistryStylesBoosts", []) if 'name' in item and 'threeChemistryModifiers' in item
    }
    all_playstyles = list(maps.get("playstyles", {}).values())

    player_ids = [f.stem.split('_')[0] for f in GG_DATA_DIR.glob("*_ggData.json")]
    print(f"ℹ️ Found {len(player_ids)} players to process.")

    rows = []

    for i, ea_id_str in enumerate(player_ids):
        if (i + 1) % 200 == 0: print(f"⏳ {i+1}/{len(player_ids)} players...")

        gg_data = load_json_file(GG_DATA_DIR / f"{ea_id_str}_ggData.json")
        gg_meta = load_json_file(GG_META_DIR / f"{ea_id_str}_ggMeta.json")

        if not gg_data or "data" not in gg_data or not gg_meta or "data" not in gg_meta or "scores" not in gg_meta["data"]:
            continue

        pdata = gg_data["data"]
        gender_txt = _normalize_gender(pdata.get("gender"), maps)
        base_attributes = {k: v for k, v in pdata.items() if isinstance(k, str) and k.startswith("attribute")}

        # shared base fields
        base_fields = {
            "height": pdata.get("height"),
            "weight": pdata.get("weight"),
            "skillMoves": pdata.get("skillMoves"),
            "weakFoot": pdata.get("weakFoot"),
            "bodytype": (maps.get("bodytypeCode", {}) or {}).get(str(pdata.get("bodytypeCode"))),
            "foot": (maps.get("foot", {}) or {}).get(str(pdata.get("foot"))),
        }

        # PS encodings (0/1/2)
        ps_map = maps.get("playstyles", {}) or {}
        player_ps = {ps_map.get(str(p)) for p in (pdata.get("playstyles") or [])}
        player_ps_plus = {ps_map.get(str(p)) for p in (pdata.get("playstylesPlus") or [])}
        for ps in all_playstyles:
            base_fields[ps] = 2 if ps in player_ps_plus else 1 if ps in player_ps else 0

        # scores by role
        scores_by_role = defaultdict(list)
        for score_entry in gg_meta["data"]["scores"]:
            scores_by_role[str(score_entry.get("role"))].append(score_entry)

        for role_id_str, scores in scores_by_role.items():
            role_name = (maps.get("role", {}) or {}).get(role_id_str)
            if not role_name: 
                continue
            fam = familiarity_for_role(role_name, pdata, maps)

            # For *every* chem style's absolute score, create a row with boosted attributes (that produced that score)
            for s in scores:
                chem_id = str(s.get("chemistryStyle"))
                chem_name = (gg_style_names.get(chem_id) or "").lower()
                boosts = chem_style_boosts.get(chem_name, {})

                row = base_fields.copy()
                row["role"] = role_name
                # boosted attributes
                for attr in base_attributes:
                    row[attr] = get_attribute_with_boost(base_attributes, attr, boosts)
                # accel type from boosted attributes using gender-aware rules
                row["accelerateType"] = calculate_acceleration_type(
                    row.get("attributeAcceleration"), row.get("attributeAgility"),
                    row.get("attributeStrength"), row.get("height"), gender_txt
                )
                row["familiarity"] = fam
                # target = absolute ggMeta for that (chem, role)
                if s.get("score") is not None:
                    row["target_ggMetaAbs"] = float(s["score"])
                    rows.append(row)

    print("✅ Finished building rows. Creating DataFrame...")

    df = pd.DataFrame(rows)
    if df.empty:
        print("⚠️ No rows produced. Check inputs.")
        return

    # One-hot only the categorical columns; gender already used in accel logic; not included as feature to avoid strings.
    df = pd.get_dummies(df, columns=["role", "bodytype", "accelerateType", "foot"], dtype=int)
    df.dropna(subset=["target_ggMetaAbs"], inplace=True)
    df.fillna(0, inplace=True)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"💾 Saved ABSOLUTE ggMeta training data with {len(df)} rows to {OUTPUT_FILE.name}")

if __name__ == "__main__":
    main()
