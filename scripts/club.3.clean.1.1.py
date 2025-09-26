# club.3.clean.py  (TEMPORARY: no evo players, no model predictions)

import json
from pathlib import Path
from collections import defaultdict
import warnings
from typing import Optional, Dict, Any, List

# Suppress harmless warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- Configuration ---
BASE_DATA_DIR = Path(__file__).resolve().parent / '../data'
RAW_DATA_DIR = BASE_DATA_DIR / 'raw'
GG_DATA_DIR = RAW_DATA_DIR / 'ggData'
ES_META_DIR = RAW_DATA_DIR / 'esMeta'
GG_META_DIR = RAW_DATA_DIR / 'ggMeta'
MAPS_FILE = BASE_DATA_DIR / 'maps.json'
OUTPUT_FILE = BASE_DATA_DIR / 'club_final.json'
CLUB_IDS_FILE = BASE_DATA_DIR / 'club_ids.json'

# --- Helpers ---
def load_json_file(file_path: Path, default_val=None):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default_val

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

def get_attribute_with_boost(base_attributes, attr_name, boost_modifiers, default_val=0):
    base_val = base_attributes.get(attr_name, default_val)
    boost_val = (boost_modifiers or {}).get(attr_name, 0)
    try:
        base_val = int(base_val) if base_val is not None else 0
        boost_val = int(boost_val) if boost_val is not None else 0
    except Exception:
        base_val = 0; boost_val = 0
    return min(base_val + boost_val, 99)

# --- Core logic (standard players only) ---

def _roles_from_gg(gg_meta_raw: Optional[Dict[str, Any]], maps) -> List[str]:
    """Collect role names present in ggMeta (if any)."""
    roles = set()
    if gg_meta_raw and "data" in gg_meta_raw and "scores" in gg_meta_raw["data"]:
        for s in gg_meta_raw["data"]["scores"]:
            rn = maps["role_id_to_name_map"].get(str(s.get("role")))
            if rn:
                roles.add(rn)
    return sorted(roles)

def _roles_from_es(es_meta_raw: Optional[list], maps) -> List[str]:
    """Collect role names present in ES meta by reading top-level roleId or playerRoleId."""
    roles = set()
    if not es_meta_raw:
        return []
    # Prefer top-level roleId -> name
    for b in es_meta_raw:
        rid = b.get("roleId")
        if rid is not None:
            name = maps["es_id_to_name_map"].get(str(rid))
            if name:
                roles.add(name)
        # Also look inside metaRatings for playerRoleId
        ratings = (b.get("data", {}) or {}).get("metaRatings", []) or []
        for r in ratings:
            pr = r.get("playerRoleId")
            if pr is not None:
                name = maps["es_id_to_name_map"].get(str(pr))
                if name:
                    roles.add(name)
    return sorted(roles)

def _es_for_role_standard(es_meta_raw: Optional[list], role_name: str, maps, base_attributes, height) -> Dict[str, Any]:
    """
    Return esMetaSub/esMeta/esChemStyle/esAccelType for a STANDARD player and a given role.
    Strictly filter by playerRoleId == ES-role-id of that role.
    """
    out = {}
    if not es_meta_raw:
        return out

    es_role_id = maps["roleNameToEsRoleId"].get(role_name)
    if not es_role_id:
        return out

    # Flatten & filter by playerRoleId (role-specific)
    filtered = []
    for b in es_meta_raw:
        rs = (b.get("data", {}) or {}).get("metaRatings", []) or []
        filtered.extend([r for r in rs if str(r.get("playerRoleId")) == str(es_role_id)])

    if not filtered:
        return out

    # 0-chem (sub)
    r0 = next((r for r in filtered if r.get("chemistry") == 0 and r.get("metaRating") is not None), None)
    if r0:
        out["esMetaSub"] = round(float(r0["metaRating"]), 2)

    # best at chem=3
    best_3 = max(
        [r for r in filtered if r.get("chemistry") == 3 and r.get("isBestChemstyleAtChem") and r.get("metaRating") is not None],
        key=lambda x: x.get("metaRating", -1), default=None
    )
    if best_3:
        out["esMeta"] = round(float(best_3["metaRating"]), 2)
        chem_id_3 = str(best_3.get("chemstyleId"))
        out["esChemStyle"] = maps["es_chem_style_names_map"].get(chem_id_3)

        # esAccelType using the chosen ES chemstyle boosts
        boosts_3 = maps["chem_style_boosts_map"].get((out.get("esChemStyle") or "").lower(), {})
        out["esAccelType"] = calculate_acceleration_type(
            get_attribute_with_boost(base_attributes, "attributeAcceleration", boosts_3),
            get_attribute_with_boost(base_attributes, "attributeAgility", boosts_3),
            get_attribute_with_boost(base_attributes, "attributeStrength", boosts_3),
            height
        )
    return out

def _gg_for_role_standard(gg_meta_raw: Optional[Dict[str, Any]], role_name: str, maps, base_attributes, height) -> Dict[str, Any]:
    """
    Return ggMeta/ggChemStyle/ggAccelType and ggMetaSub for a STANDARD player and a given role, if available.
    """
    out = {}
    if not (gg_meta_raw and "data" in gg_meta_raw and "scores" in gg_meta_raw["data"]):
        return out

    # Scores for this role_name
    role_scores = [s for s in gg_meta_raw["data"]["scores"]
                   if maps["role_id_to_name_map"].get(str(s.get("role"))) == role_name]

    if role_scores:
        best_gg = max(role_scores, key=lambda x: x.get("score", 0))
        if best_gg and best_gg.get("score") is not None:
            out["ggMeta"] = round(float(best_gg["score"]), 2)
            chem_id = str(best_gg.get("chemistryStyle"))
            out["ggChemStyle"] = maps["gg_chem_style_names_map"].get(chem_id)
            boosts = maps["chem_style_boosts_map"].get((out.get("ggChemStyle") or "").lower(), {})
            out["ggAccelType"] = calculate_acceleration_type(
                get_attribute_with_boost(base_attributes, "attributeAcceleration", boosts),
                get_attribute_with_boost(base_attributes, "attributeAgility", boosts),
                get_attribute_with_boost(base_attributes, "attributeStrength", boosts),
                height
            )

        basic_gg = next((s for s in role_scores
                         if (maps["gg_chem_style_names_map"].get(str(s.get("chemistryStyle")), "") or "").lower() == "basic"), None)
        if basic_gg and basic_gg.get("score") is not None:
            out["ggMetaSub"] = round(float(basic_gg["score"]), 2)
        elif "GK" in role_name and out.get("ggMeta") is not None:
            # mild fallback when basic is absent for GKs
            out["ggMetaSub"] = round(out["ggMeta"] * 0.95, 2)

    return out

def process_player_standard(player_def: Dict[str, Any], maps) -> Dict[str, Any]:
    """Process a STANDARD player only, no evos, no ML predictions."""
    player_output = {
        "eaId": player_def.get("eaId"),
        "evolution": False
    }
    base_attributes = {k: v for k, v in player_def.items() if k.startswith("attribute")}

    # Common simple fields
    for key in ["commonName", "overall", "height", "weight", "skillMoves", "weakFoot"]:
        player_output[key] = player_def.get(key)
    player_output.update(base_attributes)

    # Positions and categorical labels
    numeric_positions = [str(p) for p in [player_def.get("position")] + (player_def.get("alternativePositionIds") or []) if p is not None]
    player_output["positions"] = list(set([maps["position_map"].get(p) for p in numeric_positions if p in maps["position_map"]]))
    player_output["foot"] = maps["foot_map"].get(str(player_def.get("foot")))
    player_output["PS"] = [maps["playstyles_map"].get(str(p)) for p in (player_def.get("playstyles") or []) if str(p) in maps["playstyles_map"]]
    player_output["PS+"] = [maps["playstyles_map"].get(str(p)) for p in (player_def.get("playstylesPlus") or []) if str(p) in maps["playstyles_map"]]
    player_output["roles+"]  = [maps["roles_plus_map"].get(str(r)) for r in (player_def.get("rolesPlus") or []) if str(r) in maps["roles_plus_map"]]
    player_output["roles++"] = [maps["roles_plus_plus_map"].get(str(r)) for r in (player_def.get("rolesPlusPlus") or []) if str(r) in maps["roles_plus_plus_map"]]
    player_output["bodyType"] = maps["bodytype_code_map"].get(str(player_def.get("bodytypeCode")))

    sub_accel_type = calculate_acceleration_type(
        base_attributes.get("attributeAcceleration"),
        base_attributes.get("attributeAgility"),
        base_attributes.get("attributeStrength"),
        player_output.get("height")
    )

    # Raw blobs
    es_meta_raw = load_json_file(ES_META_DIR / f"{player_output['eaId']}_esMeta.json", [])
    gg_meta_raw = load_json_file(GG_META_DIR / f"{player_output['eaId']}_ggMeta.json")

    # Roles to process = union(roles from gg, roles from es)
    roles = set(_roles_from_gg(gg_meta_raw, maps)) | set(_roles_from_es(es_meta_raw, maps))
    roles = sorted(list(roles))

    player_output["metaRatings"] = []
    for role_name in roles:
        meta_entry = {"role": role_name, "subAccelType": sub_accel_type}

        # gg (if available)
        gg_part = _gg_for_role_standard(gg_meta_raw, role_name, maps, base_attributes, player_output.get("height"))
        meta_entry.update(gg_part)

        # es (always try; strictly role-filtered)
        es_part = _es_for_role_standard(es_meta_raw, role_name, maps, base_attributes, player_output.get("height"))
        meta_entry.update(es_part)

        # Guarantee ggMetaSub ≤ ggMeta if both exist
        if meta_entry.get("ggMetaSub") is not None and meta_entry.get("ggMeta") is not None:
            if meta_entry["ggMetaSub"] > meta_entry["ggMeta"]:
                meta_entry["ggMetaSub"] = round(float(meta_entry["ggMeta"]), 2)

        # Averages
        gg_meta, es_meta = meta_entry.get("ggMeta"), meta_entry.get("esMeta")
        gg_meta_sub, es_meta_sub = meta_entry.get("ggMetaSub"), meta_entry.get("esMetaSub")
        meta_entry["avgMeta"] = round((gg_meta + es_meta) / 2, 2) if (gg_meta is not None and es_meta is not None) else (gg_meta if gg_meta is not None else es_meta)
        meta_entry["avgMetaSub"] = round((gg_meta_sub + es_meta_sub) / 2, 2) if (gg_meta_sub is not None and es_meta_sub is not None) else (gg_meta_sub if gg_meta_sub is not None else es_meta_sub)

        player_output["metaRatings"].append(meta_entry)

    return player_output

def main():
    print("🚀 Starting Player Data Processing Script (STANDARD ONLY, no models/evo)...")
    maps_data = load_json_file(MAPS_FILE)
    if not maps_data:
        print("❌ Critical error: Could not load maps.json. Exiting.")
        return

    # Robust ES role maps
    es_id_to_name = maps_data.get("esRoleId", {}) or {}
    roleNameToEsRoleId = maps_data.get("roleNameToEsRoleId", {}) or {}
    if not es_id_to_name and roleNameToEsRoleId:
        es_id_to_name = {str(v): k for k, v in roleNameToEsRoleId.items()}
    if not roleNameToEsRoleId and es_id_to_name:
        roleNameToEsRoleId = {v: k for k, v in es_id_to_name.items()}

    maps = {
        "position_map": maps_data.get("position", {}),
        "foot_map": maps_data.get("foot", {}),
        "playstyles_map": maps_data.get("playstyles", {}),
        "roles_plus_map": maps_data.get("rolesPlus", {}),
        "roles_plus_plus_map": maps_data.get("rolesPlusPlus", {}),
        "bodytype_code_map": maps_data.get("bodytypeCode", {}),
        "gg_chem_style_names_map": maps_data.get("ggChemistryStyleNames", {}),
        "es_chem_style_names_map": maps_data.get("esChemistryStyleNames", {}),
        "chem_style_boosts_map": {item['name'].lower(): item['threeChemistryModifiers'] for item in maps_data.get("ChemistryStylesBoosts", []) if 'name' in item},
        "role_id_to_name_map": maps_data.get("role", {}),             # gg role id -> name
        "roleNameToEsRoleId": roleNameToEsRoleId,                      # es role name -> id
        "es_id_to_name_map": es_id_to_name                             # es role id -> name
    }

    processed_players = []

    # --- Processing Club Players (standard only) ---
    club_ids_data = load_json_file(CLUB_IDS_FILE)
    club_player_ids = [str(id) for id in club_ids_data] if isinstance(club_ids_data, list) else []
    print(f"ℹ️ Found {len(club_player_ids)} club players to process from {CLUB_IDS_FILE.name}.")

    for i, ea_id in enumerate(club_player_ids):
        if (i + 1) % 50 == 0:
            print(f"  - Player {i+1}/{len(club_player_ids)}")

        gg_data_raw = load_json_file(GG_DATA_DIR / f"{ea_id}_ggData.json")
        if not (gg_data_raw and "data" in gg_data_raw):
            continue
        player = process_player_standard(gg_data_raw["data"], maps)
        if player:
            processed_players.append(player)

    # (No evo processing in this temporary build)

    print("\n--- Deduplicating Players ---")
    final_players = {}
    for player in processed_players:
        # (No evos, so first wins)
        if player['eaId'] not in final_players:
            final_players[player['eaId']] = player
    final_list = list(final_players.values())
    print(f"ℹ️ Total unique players: {len(final_list)}")

    print("\n--- Saving Final Output ---")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_list, f, indent=2, ensure_ascii=False)
    print(f"\n🎉 Success! Processed {len(final_list)} players to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
