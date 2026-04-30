# build_all_players_summary.py
# Builds a static summary JSON of ALL players in the raw data directory.
# Uses API-fetched meta ratings (esMeta, esMetaSub, ggMeta) — no model predictions.
# Run via retrain.py or standalone.

import json
import sys
import time
from pathlib import Path
from collections import defaultdict
from shared_utils import load_json_file, _normalize_gender, calculate_acceleration_type, get_attribute_with_boost

# --- Configuration ---
BASE_DATA_DIR = Path(__file__).resolve().parent / '../data'
RAW_DATA_DIR = BASE_DATA_DIR / 'raw'
GG_DATA_DIR = RAW_DATA_DIR / 'ggData'
ES_META_DIR = RAW_DATA_DIR / 'esMeta'
GG_META_DIR = RAW_DATA_DIR / 'ggMeta'
MAPS_FILE = BASE_DATA_DIR / 'maps.json'
OUTPUT_FILE = BASE_DATA_DIR / 'all_players_summary.json'

MIN_HEIGHT_CM = 195


def main():
    start = time.time()
    print("🚀 Building all-players summary...")

    maps = load_json_file(MAPS_FILE)
    if not maps:
        print("❌ Could not load maps.json. Exiting.")
        return

    foot_map = maps.get("foot", {})
    bodytype_map = maps.get("bodytypeCode", {})
    position_map = maps.get("position", {})
    playstyles_map = maps.get("playstyles", {})
    roles_plus_map = maps.get("rolesPlus", {})
    roles_pp_map = maps.get("rolesPlusPlus", {})
    role_map = maps.get("role", {})
    es_role_id_map = maps.get("esRoleId", {}) or {str(v): k for k, v in (maps.get("roleNameToEsRoleId", {}) or {}).items()}
    gg_chem_names = maps.get("ggChemistryStyleNames", {})
    es_chem_names = maps.get("esChemistryStyleNames", {})
    chem_style_boosts_map = {
        item['name'].lower(): item['threeChemistryModifiers']
        for item in maps.get("ChemistryStylesBoosts", [])
        if 'name' in item and 'threeChemistryModifiers' in item
    }

    gg_files = sorted(GG_DATA_DIR.glob("*_ggData.json"))
    print(f"ℹ️  Found {len(gg_files)} player files.")

    results = []

    for i, gg_file in enumerate(gg_files):
        if (i + 1) % 2000 == 0:
            print(f"  ⏳ {i + 1}/{len(gg_files)}...")

        try:
            gg_raw = json.loads(gg_file.read_text(encoding='utf-8'))
        except Exception:
            continue

        p = gg_raw.get("data")
        if not p:
            continue

        ea_id = str(p.get("eaId", ""))
        base_attributes = {k: v for k, v in p.items() if k.startswith("attribute")}

        # --- Basic info ---
        gender_text = _normalize_gender(p.get("gender"), maps)

        player = {
            "eaId": p.get("eaId"),
            "commonName": p.get("commonName") or p.get("nickname") or f"{p.get('firstName', '')} {p.get('lastName', '')}".strip(),
            "overall": p.get("overall"),
            "height": p.get("height"),
            "weight": p.get("weight"),
            "skillMoves": p.get("skillMoves"),
            "weakFoot": p.get("weakFoot"),
            "gender": gender_text,
        }
        player.update(base_attributes)

        # Positions
        numeric_positions = [str(pos) for pos in [p.get("position")] + (p.get("alternativePositionIds") or []) if pos is not None]
        player["positions"] = list(set(position_map.get(pos) for pos in numeric_positions if pos in position_map))

        # Foot, body type
        player["foot"] = foot_map.get(str(p.get("foot")))
        player["bodyType"] = bodytype_map.get(str(p.get("bodytypeCode")))

        # Playstyles
        player["PS"] = [playstyles_map.get(str(ps)) for ps in (p.get("playstyles") or []) if str(ps) in playstyles_map]
        player["PS+"] = [playstyles_map.get(str(ps)) for ps in (p.get("playstylesPlus") or []) if str(ps) in playstyles_map]

        # Roles
        player["roles+"] = [roles_plus_map.get(str(r)) for r in (p.get("rolesPlus") or []) if str(r) in roles_plus_map]
        player["roles++"] = [roles_pp_map.get(str(r)) for r in (p.get("rolesPlusPlus") or []) if str(r) in roles_pp_map]

        # Rarity
        rarity_data = p.get("rarity")
        player["rarity"] = rarity_data.get("name") if isinstance(rarity_data, dict) else None

        # Tall flag
        try:
            player_height = int(player.get("height") or 0)
        except (ValueError, TypeError):
            player_height = 0
        player["isTall"] = player_height >= MIN_HEIGHT_CM

        # Sub accel type (0 chem)
        sub_accel_type = calculate_acceleration_type(
            base_attributes.get("attributeAcceleration"),
            base_attributes.get("attributeAgility"),
            base_attributes.get("attributeStrength"),
            player.get("height"),
            gender_text
        )

        # --- Meta Ratings (API-fetched) ---
        # Parse ggMeta
        gg_scores_by_role = defaultdict(list)
        gg_meta_file = GG_META_DIR / f"{ea_id}_ggMeta.json"
        gg_meta_raw = load_json_file(gg_meta_file)
        if gg_meta_raw and "data" in gg_meta_raw and "scores" in gg_meta_raw["data"]:
            for score in gg_meta_raw["data"]["scores"]:
                gg_scores_by_role[str(score.get("role"))].append(score)

        # Parse esMeta
        es_meta_raw = load_json_file(ES_META_DIR / f"{ea_id}_esMeta.json", [])

        player["metaRatings"] = []

        for role_id_str, scores in gg_scores_by_role.items():
            role_name = role_map.get(role_id_str)
            if not role_name:
                continue

            meta_entry = {"role": role_name, "subAccelType": sub_accel_type}

            # GG Best (highest score across all chem styles)
            best_gg = max(scores, key=lambda x: x.get("score", 0), default=None)
            if best_gg:
                meta_entry["ggMeta"] = round(best_gg.get("score", 0.0), 2)
                chem_id = str(best_gg.get("chemistryStyle"))
                meta_entry["ggChemStyle"] = gg_chem_names.get(chem_id)

                # Compute gg accel type with that chem style's boosts
                boosts = chem_style_boosts_map.get((meta_entry.get("ggChemStyle") or "").lower(), {})
                meta_entry["ggAccelType"] = calculate_acceleration_type(
                    get_attribute_with_boost(base_attributes, "attributeAcceleration", boosts),
                    get_attribute_with_boost(base_attributes, "attributeAgility", boosts),
                    get_attribute_with_boost(base_attributes, "attributeStrength", boosts),
                    player.get("height"), gender_text
                )

            # ES Meta for this role
            es_role_id = maps.get("roleNameToEsRoleId", {}).get(role_name)
            if es_role_id and es_meta_raw:
                filtered = []
                for block in es_meta_raw:
                    rs = (block.get("data", {}) or {}).get("metaRatings", []) or []
                    filtered.extend(r for r in rs if str(r.get("playerRoleId")) == str(es_role_id))

                if filtered:
                    # esMetaSub = 0-chem rating
                    r0 = next((r for r in filtered if r.get("chemistry") == 0 and r.get("metaRating")), None)
                    if r0:
                        meta_entry["esMetaSub"] = round(float(r0["metaRating"]), 2)

                    # esMeta = best 3-chem rating
                    best3 = max(
                        [r for r in filtered if r.get("chemistry") == 3],
                        key=lambda x: x.get("metaRating", -1),
                        default=None
                    )
                    if best3 and best3.get("metaRating"):
                        meta_entry["esMeta"] = round(float(best3["metaRating"]), 2)
                        meta_entry["esChemStyle"] = es_chem_names.get(str(best3.get("chemstyleId")))

                        # ES accel type with that chem style's boosts
                        es_boosts = chem_style_boosts_map.get((meta_entry.get("esChemStyle") or "").lower(), {})
                        meta_entry["esAccelType"] = calculate_acceleration_type(
                            get_attribute_with_boost(base_attributes, "attributeAcceleration", es_boosts),
                            get_attribute_with_boost(base_attributes, "attributeAgility", es_boosts),
                            get_attribute_with_boost(base_attributes, "attributeStrength", es_boosts),
                            player.get("height"), gender_text
                        )

            # Averages
            gg_meta_val = meta_entry.get("ggMeta")
            es_meta_val = meta_entry.get("esMeta")
            es_meta_sub_val = meta_entry.get("esMetaSub")

            if gg_meta_val and es_meta_val:
                meta_entry["avgMeta"] = round((gg_meta_val + es_meta_val) / 2, 2)
            else:
                meta_entry["avgMeta"] = gg_meta_val or es_meta_val

            # avgMetaSub: we only have esMetaSub (no ggMetaSub without model)
            meta_entry["avgMetaSub"] = es_meta_sub_val

            player["metaRatings"].append(meta_entry)

        results.append(player)

    # Write output
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False)

    elapsed = round(time.time() - start, 1)
    print(f"✅ Saved {len(results)} players to {OUTPUT_FILE.name} ({elapsed}s)")


if __name__ == "__main__":
    main()
