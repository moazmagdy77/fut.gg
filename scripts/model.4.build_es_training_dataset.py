# model.4.build_es_training_dataset.py

import json
from pathlib import Path
import pandas as pd
from collections import defaultdict
import os
from shared_utils import load_json_file, _normalize_gender, calculate_acceleration_type, get_attribute_with_boost, familiarity_for_role

# --- Configuration ---
BASE_DATA_DIR = Path(__file__).resolve().parent / '../data'
RAW_DATA_DIR = BASE_DATA_DIR / 'raw'
GG_DATA_DIR = RAW_DATA_DIR / 'ggData'
ES_META_DIR = RAW_DATA_DIR / 'esMeta'
MAPS_FILE = BASE_DATA_DIR / 'maps.json'
OUTPUT_COMBINED_FILE = BASE_DATA_DIR / 'training_dataset_es_meta_combined.csv'



def main():
    print("🚀 Starting script to build training datasets...")

    maps = load_json_file(MAPS_FILE)
    if not maps:
        print(f"❌ Critical error: Could not load {MAPS_FILE}. Exiting.")
        return

    # ES id->name robust mapping
    es_id_to_name = maps.get("esRoleId", {}) or {str(v): k for k, v in (maps.get("roleNameToEsRoleId", {}) or {}).items()}
    foot_map = maps.get("foot", {})

    chem_style_boosts = {item['name'].lower(): item['threeChemistryModifiers']
                         for item in maps.get("ChemistryStylesBoosts", []) if 'name' in item and 'threeChemistryModifiers' in item}
    all_playstyles = list(maps.get("playstyles", {}).values())

    rows_combined = []
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
        gender_text = _normalize_gender(player_def.get("gender"), maps)

        base_features = {
            "height": player_def.get("height"),
            "weight": player_def.get("weight"),
            "skillMoves": player_def.get("skillMoves"),
            "weakFoot": player_def.get("weakFoot"),
            "bodytype": maps.get("bodytypeCode", {}).get(str(player_def.get("bodytypeCode"))),
            "foot": foot_map.get(str(player_def.get("foot")))
            # NOTE: do NOT store 'gender' as a feature
        }

        # playstyles 0/1/2
        ps_map = maps.get("playstyles", {}) or {}
        player_ps = {ps_map.get(str(p)) for p in (player_def.get("playstyles") or [])}
        player_ps_plus = {ps_map.get(str(p)) for p in (player_def.get("playstylesPlus") or [])}
        for ps in all_playstyles:
            base_features[ps] = 2 if ps in player_ps_plus else (1 if ps in player_ps else 0)

        # ES roles present
        player_es_roles = {str(item.get('roleId')): item for item in es_meta if isinstance(item, dict)}
        for es_role_id, role_block in player_es_roles.items():
            role_name = es_id_to_name.get(es_role_id)
            ratings = role_block.get("data", {}).get("metaRatings")
            if not role_name or not ratings:
                continue

            fam = familiarity_for_role(role_name, player_def, maps)

            # 0-chem row
            rating_0 = next((r for r in ratings if r.get("chemistry") == 0 and str(r.get("playerRoleId")) == es_role_id), None)
            if rating_0 and rating_0.get("metaRating") is not None:
                row_0 = base_features.copy()
                row_0["role"] = role_name
                row_0.update(base_attributes)
                row_0["accelerateType"] = calculate_acceleration_type(
                    row_0.get("attributeAcceleration"), row_0.get("attributeAgility"),
                    row_0.get("attributeStrength"), row_0.get("height"), gender_text
                )
                row_0["familiarity"] = fam
                row_0["target_esMeta"] = float(rating_0["metaRating"])
                rows_combined.append(row_0)

            # 3-chem rows
            ratings_3 = [r for r in ratings if r.get("chemistry") == 3 and str(r.get("playerRoleId")) == es_role_id]
            for r3 in ratings_3:
                chem_id = str(r3.get("chemstyleId"))
                chem_name = (maps.get("esChemistryStyleNames", {}) or {}).get(chem_id)
                boosts = chem_style_boosts.get(chem_name.lower() if isinstance(chem_name, str) else None, {})

                row_3 = base_features.copy(); row_3["role"] = role_name
                for attr in base_attributes:
                    row_3[attr] = get_attribute_with_boost(base_attributes, attr, boosts)
                row_3["accelerateType"] = calculate_acceleration_type(
                    row_3.get("attributeAcceleration"), row_3.get("attributeAgility"),
                    row_3.get("attributeStrength"), row_3.get("height"), gender_text
                )
                row_3["familiarity"] = fam
                if r3.get("metaRating") is not None:
                    row_3["target_esMeta"] = float(r3["metaRating"])
                    rows_combined.append(row_3)

    print("✅ Finished processing all players. Creating DataFrames...")

    df_combined = pd.DataFrame(rows_combined)
    if not df_combined.empty:
        # ensure no stray 'gender' col
        if "gender" in df_combined.columns: df_combined.drop(columns=["gender"], inplace=True)
        df_combined = pd.get_dummies(df_combined, columns=["role", "bodytype", "accelerateType", "foot"], dtype=int)
        df_combined.dropna(subset=['target_esMeta'], inplace=True)
        df_combined.to_csv(OUTPUT_COMBINED_FILE, index=False)
        print(f"💾 Saved combined training data with {len(df_combined)} rows to {OUTPUT_COMBINED_FILE.name}")
    else:
        print("⚠️ No rows produced (check inputs).")

if __name__ == "__main__":
    main()
