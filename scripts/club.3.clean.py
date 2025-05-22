import json
from pathlib import Path
import os
from collections import defaultdict

# --- Configuration ---
BASE_DATA_DIR = Path(__file__).resolve().parent / '../data'
RAW_DATA_DIR = BASE_DATA_DIR / 'raw'
GG_DATA_DIR = RAW_DATA_DIR / 'ggData'
ES_META_DIR = RAW_DATA_DIR / 'esMeta'
GG_META_DIR = RAW_DATA_DIR / 'ggMeta'
MAPS_FILE = BASE_DATA_DIR / 'maps.json'
OUTPUT_FILE = BASE_DATA_DIR / 'club_final.json'

# Fields to remove from the final player object (after necessary data is extracted)
# This list can be adjusted based on what's truly not needed for Streamlit.
# Start with fields from your old club.4.clean.gg.py and adjust.
FUTGG_FIELDS_TO_REMOVE = [
    "playerType", "id", "evolutionId", "cosmeticEvolutionId", 
    "partialEvolutionId", "basePlayerEaId", "nickname",
    # "foot", # Will be mapped
    # "position", # Will be combined and mapped
    # "alternativePositionIds", # Will be combined
    # "playstyles", # Will be mapped
    # "playstylesPlus", # Will be mapped
    # "rolesPlus", # Will be mapped
    # "rolesPlusPlus", # Will be mapped
    "url", 
    # "bodytypeCode", # Will be mapped
    "isRealFace", 
    "facePace", "faceShooting", "facePassing", "faceDribbling", "faceDefending",
    "facePhysicality", "gkFaceDiving", "gkFaceHandling", "gkFaceKicking", 
    "gkFaceReflexes", "gkFaceSpeed", "gkFacePositioning", 
    "isUserEvolutions", "isEvoLabPlayerItem", "shirtNumber", 
    "totalFaceStats", "totalIgs", "game", "slug", "basePlayerSlug", 
    "gender", "searchableName", "dateOfBirth", "attackingWorkrate",
    "defensiveWorkrate", "nationEaId", "leagueEaId", "clubEaId", "uniqueClubEaId",
    "uniqueClubSlug", "rarityEaId", "raritySquadId", "guid", 
    "accelerateTypes", # This is the fut.gg direct field, we calculate our own
    "hasDynamic", "renderOnlyAsHtml", "createdAt", "isHidden", 
    "previousVersionsIds", "imagePath", "simpleCardImagePath", 
    "futggCardImagePath", "cardImagePath", "shareImagePath", "socialImagePath",
    "targetFacePace", "targetFaceShooting", "targetFacePassing",
    "targetFaceDribbling", "targetFaceDefending", "targetFacePhysicality",
    "isOnMarket", "isUntradeable", "sbcSetEaId", "sbcChallengeEaId",
    "objectiveGroupEaId", "objectiveGroupObjectiveEaId", 
    "objectiveCampaignLevelId", "campaignProps", "contentTypeId",
    "numberOfEvolutions", "blurbText", "smallBlurbText", "upgrades",
    "hasPrice", "trackerId", "liveHubTrackerId", "playerScore",
    "coinCost", "pointCost", "onLoanFromClubEaId", "isSbcItem", "isObjectiveItem"
]


# --- Helper Functions ---

def load_json_file(file_path, default_val=None):
    """Loads a JSON file. Returns default_val if file not found or error."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        # print(f"🔍 File not found: {file_path}. Returning default.")
        return default_val
    except json.JSONDecodeError:
        print(f"⚠️ Error decoding JSON from: {file_path}. Returning default.")
        return default_val

def calculate_acceleration_type(accel, agility, strength, height):
    """Calculates player acceleration type based on attributes."""
    if not all(isinstance(x, (int, float)) for x in [accel, agility, strength, height]):
        return "CONTROLLED" # Default if any attribute is invalid

    accel = int(accel)
    agility = int(agility)
    strength = int(strength)
    height = int(height)

    if (agility - strength) >= 20 and accel >= 80 and height <= 175 and agility >=80:
        return "EXPLOSIVE"
    elif (agility - strength) >= 12 and accel >= 80 and height <= 182 and agility >= 70:
        return "MOSTLY_EXPLOSIVE"
    elif (agility - strength) >= 4 and accel >= 70 and height <= 182 and agility >= 65:
        return "CONTROLLED_EXPLOSIVE"
    elif (strength - agility) >= 20 and strength >= 80 and height >= 188 and accel >= 55:
        return "LENGTHY"
    elif (strength - agility) >= 12 and strength >= 75 and height >= 183 and accel >= 55:
        return "MOSTLY_LENGTHY"
    elif (strength - agility) >= 4 and strength >= 65 and height >= 181 and accel >= 40:
        return "CONTROLLED_LENGTHY"
    else:
        return "CONTROLLED"

def get_attribute_with_boost(base_attributes, attr_name, boost_modifiers, default_val=0):
    """Gets a base attribute and applies a boost, capping at 99."""
    base_val = base_attributes.get(attr_name, default_val)
    boost_val = boost_modifiers.get(attr_name, 0)
    # Ensure base_val and boost_val are numbers
    if not isinstance(base_val, (int, float)): base_val = default_val
    if not isinstance(boost_val, (int, float)): boost_val = 0
    return min(base_val + boost_val, 99)

# --- Main Processing ---
def main():
    print("🚀 Starting Python processing script...")

    maps_data = load_json_file(MAPS_FILE)
    if not maps_data:
        print(f"❌ Critical error: Could not load maps.json from {MAPS_FILE}. Exiting.")
        return

    # Extract specific maps for convenience
    position_map = maps_data.get("position", {})
    foot_map = maps_data.get("foot", {})
    playstyles_map = maps_data.get("playstyles", {}) # Used for both PS and PS+
    roles_plus_map = maps_data.get("rolesPlus", {})
    roles_plus_plus_map = maps_data.get("rolesPlusPlus", {})
    bodytype_code_map = maps_data.get("bodytypeCode", {})
    
    gg_chem_style_names_map = maps_data.get("ggChemistryStyleNames", {})
    es_chem_style_names_map = maps_data.get("esChemistryStyleNames", {})
    chem_style_boosts_map = {item['name'].lower(): item['threeChemistryModifiers'] 
                             for item in maps_data.get("ChemistryStylesBoosts", []) if 'name' in item}
    
    role_id_to_name_map = maps_data.get("role", {})
    role_name_to_archetype_map = maps_data.get("roleToArchetype", {})


    processed_players_list = []
    player_ea_ids = []

    if not GG_DATA_DIR.exists():
        print(f"❌ Error: Fut.gg data directory not found: {GG_DATA_DIR}")
        return
        
    for filename in os.listdir(GG_DATA_DIR):
        if filename.endswith("_ggData.json"):
            player_ea_ids.append(filename.split('_')[0])
    
    print(f"ℹ️ Found {len(player_ea_ids)} players to process based on ggData files.")

    for ea_id_str in player_ea_ids:
        ea_id = int(ea_id_str) # Assuming eaId is numeric in your data
        print(f"\n--- Processing player EA ID: {ea_id} ---")

        player_output = {"eaId": ea_id}

        # 1. Load and Process Fut.gg Player Details (_ggData.json)
        gg_details_raw = load_json_file(GG_DATA_DIR / f"{ea_id}_ggData.json")
        
        base_attributes = {} # To store original attributes for accel type calcs

        if gg_details_raw and "data" in gg_details_raw:
            gg_player_data = gg_details_raw["data"]
            
            # Basic info
            player_output["commonName"] = gg_player_data.get("commonName")
            player_output["overall"] = gg_player_data.get("overall")
            player_output["height"] = gg_player_data.get("height")
            player_output["weight"] = gg_player_data.get("weight")
            player_output["skillMoves"] = gg_player_data.get("skillMoves")
            player_output["weakFoot"] = gg_player_data.get("weakFoot")
            
            # Store all attributes
            for key, value in gg_player_data.items():
                if key.startswith("attribute"):
                    player_output[key] = value
                    base_attributes[key] = value # Store for later boost calcs
            
            # Positions
            numeric_positions = []
            if gg_player_data.get("position") is not None:
                numeric_positions.append(str(gg_player_data.get("position")))
            if isinstance(gg_player_data.get("alternativePositionIds"), list):
                numeric_positions.extend([str(pos_id) for pos_id in gg_player_data.get("alternativePositionIds")])
            player_output["positions"] = list(set([position_map.get(pos_id, pos_id) for pos_id in numeric_positions]))

            # Mappings
            player_output["foot"] = foot_map.get(str(gg_player_data.get("foot")))
            player_output["PS"] = [playstyles_map.get(str(ps_id), ps_id) for ps_id in gg_player_data.get("playstyles", [])]
            player_output["PS+"] = [playstyles_map.get(str(ps_id), ps_id) for ps_id in gg_player_data.get("playstylesPlus", [])]
            player_output["roles+"] = [roles_plus_map.get(str(rp_id), rp_id) for rp_id in gg_player_data.get("rolesPlus", [])]
            player_output["roles++"] = [roles_plus_plus_map.get(str(rp_id), rp_id) for rp_id in gg_player_data.get("rolesPlusPlus", [])]
            player_output["bodyType"] = bodytype_code_map.get(str(gg_player_data.get("bodytypeCode")))
            player_output["accelerateType"] = gg_player_data.get("accelerateType") # Base accelerate type from fut.gg

            # Calculate subAccelType (base attributes, no chem)
            sub_accel_type = calculate_acceleration_type(
                base_attributes.get("attributeAcceleration", 0),
                base_attributes.get("attributeAgility", 0),
                base_attributes.get("attributeStrength", 0),
                player_output.get("height", 0)
            )

            # Remove unwanted fut.gg fields after extracting necessary info
            for field in FUTGG_FIELDS_TO_REMOVE:
                if field in gg_player_data: # Check if field exists before popping
                    gg_player_data.pop(field, None)
            # If you want to keep only specific fields, uncomment and adjust this:
            # player_output = {k: v for k, v in player_output.items() if k not in FUTGG_FIELDS_TO_REMOVE and k not in gg_player_data_specific_removals}

        else:
            print(f"⚠️ No fut.gg details data found for player {ea_id}.")
            # Decide if you want to continue processing this player or skip
            # For now, we'll continue and other parts will handle missing data

        # 2. Load Meta Rating Data
        gg_meta_raw = load_json_file(GG_META_DIR / f"{ea_id}_ggMeta.json")
        es_meta_raw_list = load_json_file(ES_META_DIR / f"{ea_id}_esMeta.json", []) # Default to empty list

        player_output["metaRatings"] = []

        # Determine unique roles from fut.gg metarank data
        # gg_meta_raw is expected to be like: {"data": {"eaId": ..., "scores": [{"role": id, "chemistryStyle": id, "score": ...}, ...]}}
        unique_gg_role_ids = set()
        if gg_meta_raw and "data" in gg_meta_raw and "scores" in gg_meta_raw["data"]:
            for score_entry in gg_meta_raw["data"]["scores"]:
                if "role" in score_entry:
                    unique_gg_role_ids.add(str(score_entry["role"]))
        
        if not unique_gg_role_ids:
             print(f"ℹ️ No fut.gg roles found in metarank data for player {ea_id}. Will try to derive from esMeta if available.")
             # Fallback: if no ggMeta roles, try to derive from esMeta archetypes if needed,
             # though the primary loop is driven by ggMeta roles as per request.
             # For now, if no gg_roles, metaRatings will be empty unless esMeta processing is changed.


        for role_id_str in unique_gg_role_ids:
            role_name = role_id_to_name_map.get(role_id_str, f"UnknownRoleID_{role_id_str}")
            
            meta_entry = {
                "role": role_name,
                "ggMeta": None, "ggChemStyle": None, "ggAccelType": None,
                "esMetaSub": None, "esMeta": None, "esChemStyle": None, "esAccelType": None,
                "subAccelType": sub_accel_type # Base accel type is same for all roles
            }

            # Process fut.gg Meta (ggMeta) for this role
            if gg_meta_raw and "data" in gg_meta_raw and "scores" in gg_meta_raw["data"]:
                best_gg_score_for_role = None
                for score_entry in gg_meta_raw["data"]["scores"]:
                    if str(score_entry.get("role")) == role_id_str:
                        if best_gg_score_for_role is None or score_entry.get("score", 0) > best_gg_score_for_role.get("score", 0):
                            best_gg_score_for_role = score_entry
                
                if best_gg_score_for_role:
                    meta_entry["ggMeta"] = round(best_gg_score_for_role.get("score"),2) if best_gg_score_for_role.get("score") is not None else None
                    gg_chem_id = str(best_gg_score_for_role.get("chemistryStyle"))
                    meta_entry["ggChemStyle"] = gg_chem_style_names_map.get(gg_chem_id)
                    
                    if meta_entry["ggChemStyle"]:
                        boosts = chem_style_boosts_map.get(meta_entry["ggChemStyle"].lower(), {})
                        meta_entry["ggAccelType"] = calculate_acceleration_type(
                            get_attribute_with_boost(base_attributes, "attributeAcceleration", boosts),
                            get_attribute_with_boost(base_attributes, "attributeAgility", boosts),
                            get_attribute_with_boost(base_attributes, "attributeStrength", boosts),
                            player_output.get("height", 0)
                        )

            # Process EasySBC Meta (esMeta) for this role
            es_archetype_for_role = role_name_to_archetype_map.get(role_name)
            if es_archetype_for_role and es_meta_raw_list:
                # es_meta_raw_list is an array of {archetype: "name", ratings: [...]}
                es_archetype_data = next((item for item in es_meta_raw_list if item.get("archetype") == es_archetype_for_role), None)
                
                if es_archetype_data and "ratings" in es_archetype_data:
                    es_ratings = es_archetype_data["ratings"]
                    
                    # 0 Chem
                    rating_0_chem = next((r for r in es_ratings if r.get("chemistry") == 0), None)
                    if rating_0_chem:
                        meta_entry["esMetaSub"] = round(rating_0_chem.get("metaRating"),2) if rating_0_chem.get("metaRating") is not None else None
                    
                    # 3 Chem (best)
                    ratings_3_chem = [r for r in es_ratings if r.get("chemistry") == 3 and r.get("isBestChemstyleAtChem") is True]
                    if ratings_3_chem: # Should ideally be one, but take first if multiple marked as best
                        best_3_chem_rating = ratings_3_chem[0]
                        meta_entry["esMeta"] = round(best_3_chem_rating.get("metaRating"),2) if best_3_chem_rating.get("metaRating") is not None else None
                        es_chem_id = str(best_3_chem_rating.get("chemstyleId"))
                        meta_entry["esChemStyle"] = es_chem_style_names_map.get(es_chem_id)

                        if meta_entry["esChemStyle"]:
                            boosts = chem_style_boosts_map.get(meta_entry["esChemStyle"].lower(), {})
                            meta_entry["esAccelType"] = calculate_acceleration_type(
                                get_attribute_with_boost(base_attributes, "attributeAcceleration", boosts),
                                get_attribute_with_boost(base_attributes, "attributeAgility", boosts),
                                get_attribute_with_boost(base_attributes, "attributeStrength", boosts),
                                player_output.get("height", 0)
                            )
            player_output["metaRatings"].append(meta_entry)
        
        if not player_output["metaRatings"] and not unique_gg_role_ids:
             print(f"ℹ️ No meta ratings could be processed for player {ea_id} as no roles were derived.")


        processed_players_list.append(player_output)
        print(f"✅ Finished processing player EA ID: {ea_id}")

    # 3. Save Output
    try:
        if not BASE_DATA_DIR.exists():
            BASE_DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(processed_players_list, f, indent=2, ensure_ascii=False)
        print(f"\n🎉 Successfully processed {len(processed_players_list)} players.")
        print(f"💾 Final data saved to: {OUTPUT_FILE}")
    except Exception as e:
        print(f"❌ Error saving final output file: {e}")

if __name__ == "__main__":
    main()
