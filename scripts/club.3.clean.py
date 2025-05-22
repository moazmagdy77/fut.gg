import json
from pathlib import Path
import os
from collections import defaultdict
import copy # For deepcopying player objects
import pandas as pd # For esMeta prediction data handling
import joblib # For loading scikit-learn models

# --- Configuration ---
BASE_DATA_DIR = Path(__file__).resolve().parent / '../data'
MODELS_DIR = Path(__file__).resolve().parent / '../models' # For esMeta Lasso models
RAW_DATA_DIR = BASE_DATA_DIR / 'raw'
GG_DATA_DIR = RAW_DATA_DIR / 'ggData'
ES_META_DIR = RAW_DATA_DIR / 'esMeta'
GG_META_DIR = RAW_DATA_DIR / 'ggMeta'
EVOLAB_FILE = BASE_DATA_DIR / 'evolab.json' # Path to evolab.json
MAPS_FILE = BASE_DATA_DIR / 'maps.json'
OUTPUT_FILE = BASE_DATA_DIR / 'club_final.json' # Changed name to reflect combined output

# Fields to remove from the CLUB player object (after necessary data is extracted)
CLUB_PLAYER_FUTGG_FIELDS_TO_REMOVE = [
    "id", "pathHash", "evolutionPath", "numberOfEvolutions", "evolutionId", "playerType", "game", "id", 
    "evolutionId", "cosmeticEvolutionId", "partialEvolutionId", "basePlayerEaId", "basePlayerSlug",
    "gender", "slug", "firstName", "lastName", "nickname", "searchableName", "dateOfBirth", "attackingWorkrate",
    "defensiveWorkrate", "nationEaId", "leagueEaId", "clubEaId", "uniqueClubEaId", "uniqueClubSlug",
    "rarityEaId", "raritySquadId", "guid", "accelerateType", "accelerateTypes", "hasDynamic", "url",
    "renderOnlyAsHtml", "isRealFace", "isHidden", "previousVersionsIds", "imagePath", "simpleCardImagePath",
    "futggCardImagePath", "cardImagePath", "shareImagePath", "socialImagePath", 
    "facePace", "faceShooting", "facePassing", "faceDribbling", "faceDefending",
    "facePhysicality", "targetFacePace", "targetFaceShooting", "targetFacePassing",
    "targetFaceDribbling", "targetFaceDefending", "targetFacePhysicality", "gkFaceDiving",
    "gkFaceHandling", "gkFaceKicking", "gkFaceReflexes", "gkFaceSpeed", "gkFacePositioning",
    "isOnMarket", "isUntradeable", "sbcSetEaId", "sbcChallengeEaId", "objectiveGroupEaId",
    "objectiveGroupObjectiveEaId", "objectiveCampaignLevelId", "campaignProps", "contentTypeId",
    "numberOfEvolutions", "blurbText", "smallBlurbText", "upgrades", "hasPrice", "trackerId",
    "liveHubTrackerId", "isUserEvolutions", "isEvoLabPlayerItem", "totalIgsBoost",
    "evolutionIgsBoost", "playerScore", "coinCost", "pointCost", "shirtNumber", "onLoanFromClubEaId",
    "premiumSeasonPassLevel", "standardSeasonPassLevel", "metarankStr", "ggRating", "ggRatingPos",
    "isSbcItem", "isObjectiveItem", "totalFaceStats", "totalIgs", "wasUpgraded", "totalTrainingTime",
    "maxTimeToStart", "totalIgsBoost", "evolutionIgsBoost"
]

# Fields to remove from EVO player's playerItemDefinition
EVO_PLAYER_DEFINITION_FIELDS_TO_REMOVE = [
    "playerType", "game", "id", "evolutionId", "cosmeticEvolutionId", "partialEvolutionId",
    # "basePlayerEaId", # Will be used as 'eaId' (actually, 'eaId' is directly available in playerItemDefinition)
    "basePlayerSlug", "gender", "slug", "firstName", "lastName", "nickname", # commonName is preferred
    "searchableName", "dateOfBirth",
    "attackingWorkrate", "defensiveWorkrate", "nationEaId", "leagueEaId", "clubEaId", "uniqueClubEaId",
    "uniqueClubSlug", "rarityEaId", "raritySquadId", "guid",
    "accelerateTypes", # Original fut.gg field, we calculate our own
    "hasDynamic", "url",
    "renderOnlyAsHtml", "isRealFace", "createdAt", "isHidden", "previousVersionsIds", "imagePath",
    "simpleCardImagePath", "futggCardImagePath", "cardImagePath", "shareImagePath", "socialImagePath",
    "attributeGkDiving", "attributeGkHandling", "attributeGkKicking", "attributeGkReflexes", "attributeGkPositioning", # If not GK
    "facePace", "faceShooting", "facePassing", "faceDribbling", "faceDefending", "facePhysicality",
    "targetFacePace", "targetFaceShooting", "targetFacePassing", "targetFaceDribbling", "targetFaceDefending",
    "targetFacePhysicality", "gkFaceDiving", "gkFaceHandling", "gkFaceKicking", "gkFaceReflexes", "gkFaceSpeed",
    "gkFacePositioning", "isOnMarket", "isUntradeable", "sbcSetEaId", "sbcChallengeEaId", "objectiveGroupEaId",
    "objectiveGroupObjectiveEaId", "objectiveCampaignLevelId", "campaignProps", "contentTypeId", "numberOfEvolutions",
    "blurbText", "smallBlurbText", "upgrades", "hasPrice", "trackerId", "liveHubTrackerId", "isUserEvolutions",
    "isEvoLabPlayerItem", # Flag used to identify, then remove
    "totalIgsBoost", "evolutionIgsBoost", "playerScore", "coinCost", "pointCost", "shirtNumber", "onLoanFromClubEaId",
    "premiumSeasonPassLevel", "standardSeasonPassLevel",
    "metarankStr", "ggRating", "ggRatingPos", "ggRatingStr", # Used for ggMeta, then remove
    "isSbcItem", "isObjectiveItem", "totalFaceStats", "totalIgs"
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
    if not all(isinstance(x, (int, float)) and x is not None for x in [accel, agility, strength, height]):
        return "CONTROLLED"
    accel = int(accel)
    agility = int(agility)
    strength = int(strength)
    height = int(height)

    if (agility - strength) >= 20 and accel >= 80 and height <= 175 and agility >= 80:
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
    if not isinstance(base_val, (int, float)): base_val = default_val if default_val is not None else 0
    if not isinstance(boost_val, (int, float)): boost_val = 0
    
    # Ensure base_val is not None before attempting int conversion
    if base_val is None: base_val = 0

    return min(int(base_val) + int(boost_val), 99)

def parse_gg_rating_str(gg_rating_str_raw):
    """
    Parses the ggRatingStr like "chemstyleid:roleid:ggMeta||..."
    Returns a dict: {role_id_str: [{"chem_id_str": X, "score": Y}, ...]}
    """
    parsed_ratings_by_role = defaultdict(list)
    if not gg_rating_str_raw:
        return parsed_ratings_by_role

    parts = gg_rating_str_raw.split('||')
    for part in parts:
        try:
            chem_id_str, role_id_str, score_str = part.split(':')
            score = float(score_str)
            parsed_ratings_by_role[role_id_str].append({
                "chem_id_str": chem_id_str,
                "score": score
            })
        except ValueError:
            # print(f"⚠️ Skipping malformed ggRatingStr part: {part}")
            continue
    return parsed_ratings_by_role


# --- Club Player Processing ---
def process_single_club_player(ea_id_str, gg_data_dir, gg_meta_dir, es_meta_dir, maps, fields_to_remove):
    """Processes a single club player's data."""
    ea_id = int(ea_id_str)
    player_output = {"eaId": ea_id, "evolution": False} 
    base_attributes = {}

    gg_details_raw = load_json_file(gg_data_dir / f"{ea_id}_ggData.json")
    if not (gg_details_raw and "data" in gg_details_raw):
        print(f"⚠️ No fut.gg details data found for club player {ea_id}.")
        return None 
    
    gg_player_data = gg_details_raw["data"]

    player_output["commonName"] = gg_player_data.get("commonName")
    player_output["overall"] = gg_player_data.get("overall")
    player_output["height"] = gg_player_data.get("height")
    player_output["weight"] = gg_player_data.get("weight")
    player_output["skillMoves"] = gg_player_data.get("skillMoves")
    player_output["weakFoot"] = gg_player_data.get("weakFoot")

    for key, value in gg_player_data.items():
        if key.startswith("attribute"):
            player_output[key] = value
            base_attributes[key] = value
    
    numeric_positions = []
    if gg_player_data.get("position") is not None:
        numeric_positions.append(str(gg_player_data.get("position")))
    if isinstance(gg_player_data.get("alternativePositionIds"), list):
        numeric_positions.extend([str(pos_id) for pos_id in gg_player_data.get("alternativePositionIds")])
    player_output["positions"] = list(set([maps["position_map"].get(pos_id, pos_id) for pos_id in numeric_positions]))

    player_output["foot"] = maps["foot_map"].get(str(gg_player_data.get("foot")))
    player_output["PS"] = [maps["playstyles_map"].get(str(ps_id), ps_id) for ps_id in gg_player_data.get("playstyles", [])]
    player_output["PS+"] = [maps["playstyles_map"].get(str(ps_id), ps_id) for ps_id in gg_player_data.get("playstylesPlus", [])]
    player_output["roles+"] = [maps["roles_plus_map"].get(str(rp_id), rp_id) for rp_id in gg_player_data.get("rolesPlus", [])]
    player_output["roles++"] = [maps["roles_plus_plus_map"].get(str(rp_id), rp_id) for rp_id in gg_player_data.get("rolesPlusPlus", [])]
    player_output["bodyType"] = maps["bodytype_code_map"].get(str(gg_player_data.get("bodytypeCode")))
    player_output["accelerateType"] = gg_player_data.get("accelerateType") 

    sub_accel_type = calculate_acceleration_type(
        base_attributes.get("attributeAcceleration"),
        base_attributes.get("attributeAgility"),
        base_attributes.get("attributeStrength"),
        player_output.get("height")
    )
    player_output["subAccelType"] = sub_accel_type

    for field in fields_to_remove:
        gg_player_data.pop(field, None)

    gg_meta_raw = load_json_file(gg_meta_dir / f"{ea_id}_ggMeta.json")
    es_meta_raw_list = load_json_file(es_meta_dir / f"{ea_id}_esMeta.json", [])
    player_output["metaRatings"] = []

    unique_gg_role_ids = set()
    if gg_meta_raw and "data" in gg_meta_raw and "scores" in gg_meta_raw["data"]:
        for score_entry in gg_meta_raw["data"]["scores"]:
            if "role" in score_entry:
                unique_gg_role_ids.add(str(score_entry["role"]))

    for role_id_str in unique_gg_role_ids:
        role_name = maps["role_id_to_name_map"].get(role_id_str, f"UnknownRoleID_{role_id_str}")
        meta_entry = {
            "role": role_name,
            "ggMeta": None, "ggChemStyle": None, "ggAccelType": None,
            "esMetaSub": None, "esMeta": None, "esChemStyle": None, "esAccelType": None,
            "subAccelType": sub_accel_type
        }

        if gg_meta_raw and "data" in gg_meta_raw and "scores" in gg_meta_raw["data"]:
            best_gg_score_for_role = None
            for score_entry in gg_meta_raw["data"]["scores"]:
                if str(score_entry.get("role")) == role_id_str:
                    current_score = score_entry.get("score", 0)
                    if best_gg_score_for_role is None or current_score > best_gg_score_for_role.get("score", 0):
                        best_gg_score_for_role = score_entry
            if best_gg_score_for_role:
                meta_entry["ggMeta"] = round(best_gg_score_for_role.get("score"), 2) if best_gg_score_for_role.get("score") is not None else None
                gg_chem_id = str(best_gg_score_for_role.get("chemistryStyle"))
                meta_entry["ggChemStyle"] = maps["gg_chem_style_names_map"].get(gg_chem_id)
                if meta_entry["ggChemStyle"]:
                    boosts = maps["chem_style_boosts_map"].get(meta_entry["ggChemStyle"].lower(), {})
                    meta_entry["ggAccelType"] = calculate_acceleration_type(
                        get_attribute_with_boost(base_attributes, "attributeAcceleration", boosts),
                        get_attribute_with_boost(base_attributes, "attributeAgility", boosts),
                        get_attribute_with_boost(base_attributes, "attributeStrength", boosts),
                        player_output.get("height")
                    )
        
        es_archetype_for_role = maps["role_name_to_archetype_map"].get(role_name)
        if es_archetype_for_role and es_meta_raw_list:
            es_archetype_data = next((item for item in es_meta_raw_list if item.get("archetype") == es_archetype_for_role), None)
            if es_archetype_data and "ratings" in es_archetype_data:
                es_ratings = es_archetype_data["ratings"]
                rating_0_chem = next((r for r in es_ratings if r.get("chemistry") == 0), None)
                if rating_0_chem:
                    meta_entry["esMetaSub"] = round(rating_0_chem.get("metaRating"),2) if rating_0_chem.get("metaRating") is not None else None
                
                ratings_3_chem = [r for r in es_ratings if r.get("chemistry") == 3 and r.get("isBestChemstyleAtChem") is True]
                if ratings_3_chem:
                    best_3_chem_rating = ratings_3_chem[0]
                    meta_entry["esMeta"] = round(best_3_chem_rating.get("metaRating"),2) if best_3_chem_rating.get("metaRating") is not None else None
                    es_chem_id = str(best_3_chem_rating.get("chemstyleId"))
                    meta_entry["esChemStyle"] = maps["es_chem_style_names_map"].get(es_chem_id)
                    if meta_entry["esChemStyle"]:
                        boosts = maps["chem_style_boosts_map"].get(meta_entry["esChemStyle"].lower(), {})
                        meta_entry["esAccelType"] = calculate_acceleration_type(
                            get_attribute_with_boost(base_attributes, "attributeAcceleration", boosts),
                            get_attribute_with_boost(base_attributes, "attributeAgility", boosts),
                            get_attribute_with_boost(base_attributes, "attributeStrength", boosts),
                            player_output.get("height")
                        )
        player_output["metaRatings"].append(meta_entry)
    return player_output

# --- Evo Player Processing ---
def process_single_evo_player_basic_and_ggmeta(evo_player_def_raw, maps, fields_to_remove):
    """Processes a single evo player's basic data and ggMeta ratings."""
    player_output = {"evolution": True} 
    base_attributes = {} 
    evo_player_def = copy.deepcopy(evo_player_def_raw)

    player_output["eaId"] = evo_player_def.get("eaId") 
    if player_output["eaId"] is None:
        return None

    player_output["commonName"] = evo_player_def.get("commonName") or f"{evo_player_def.get('firstName', '')} {evo_player_def.get('lastName', '')}".strip()
    player_output["overall"] = evo_player_def.get("overall")
    player_output["height"] = evo_player_def.get("height")
    player_output["weight"] = evo_player_def.get("weight")
    player_output["skillMoves"] = evo_player_def.get("skillMoves")
    player_output["weakFoot"] = evo_player_def.get("weakFoot")

    for key, value in evo_player_def.items():
        if key.startswith("attribute") and not key.startswith("attributeGk"):
            player_output[key] = value
            base_attributes[key] = value
    
    primary_pos_id_str = str(evo_player_def.get("position"))
    if maps["position_map"].get(primary_pos_id_str) == "GK":
        for key in ["attributeGkDiving", "attributeGkHandling", "attributeGkKicking", "attributeGkReflexes", "attributeGkPositioning"]:
            if key in evo_player_def and evo_player_def[key] is not None: # Check for None
                player_output[key] = evo_player_def[key]
                base_attributes[key] = evo_player_def[key]

    numeric_positions = []
    if evo_player_def.get("position") is not None:
        numeric_positions.append(str(evo_player_def.get("position")))
    if isinstance(evo_player_def.get("alternativePositionIds"), list):
        numeric_positions.extend([str(pos_id) for pos_id in evo_player_def.get("alternativePositionIds")])
    
    player_output["numeric_position_ids_evo"] = numeric_positions # Store original IDs for archetype lookup later
    player_output["positions"] = list(set([maps["position_map"].get(pos_id, pos_id) for pos_id in numeric_positions]))


    player_output["foot"] = maps["foot_map"].get(str(evo_player_def.get("foot")))
    player_output["PS"] = [maps["playstyles_map"].get(str(ps_id), ps_id) for ps_id in evo_player_def.get("playstyles", []) if maps["playstyles_map"].get(str(ps_id))]
    player_output["PS+"] = [maps["playstyles_map"].get(str(ps_id), ps_id) for ps_id in evo_player_def.get("playstylesPlus", []) if maps["playstyles_map"].get(str(ps_id))]
    player_output["roles+"] = [maps["roles_plus_map"].get(str(rp_id), rp_id) for rp_id in evo_player_def.get("rolesPlus", []) if maps["roles_plus_map"].get(str(rp_id))]
    player_output["roles++"] = [maps["roles_plus_plus_map"].get(str(rp_id), rp_id) for rp_id in evo_player_def.get("rolesPlusPlus", []) if maps["roles_plus_plus_map"].get(str(rp_id))]
    player_output["bodyType"] = maps["bodytype_code_map"].get(str(evo_player_def.get("bodytypeCode")))
    
    player_output["accelerateType"] = calculate_acceleration_type(
        base_attributes.get("attributeAcceleration"),
        base_attributes.get("attributeAgility"),
        base_attributes.get("attributeStrength"),
        player_output.get("height")
    )
    player_output["subAccelType"] = player_output["accelerateType"]

    player_output["metaRatings"] = []
    gg_rating_str_raw = evo_player_def.get("ggRatingStr")
    parsed_gg_ratings_by_role = parse_gg_rating_str(gg_rating_str_raw)
    
    unique_evo_role_ids = set(parsed_gg_ratings_by_role.keys())

    for role_id_str in unique_evo_role_ids:
        role_name = maps["role_id_to_name_map"].get(role_id_str, f"UnknownRoleID_{role_id_str}")
        meta_entry = {
            "role": role_name,
            "ggMeta": None, "ggChemStyle": None, "ggAccelType": None,
            "esMetaSub": None, "esMeta": None, "esChemStyle": None, "esAccelType": None, 
            "subAccelType": player_output["subAccelType"]
        }

        best_gg_score_for_role_val = None
        best_chem_id_for_role = None

        for rating_info in parsed_gg_ratings_by_role.get(role_id_str, []):
            if best_gg_score_for_role_val is None or rating_info["score"] > best_gg_score_for_role_val:
                best_gg_score_for_role_val = rating_info["score"]
                best_chem_id_for_role = rating_info["chem_id_str"]
        
        if best_gg_score_for_role_val is not None:
            meta_entry["ggMeta"] = round(best_gg_score_for_role_val, 2)
            meta_entry["ggChemStyle"] = maps["gg_chem_style_names_map"].get(best_chem_id_for_role)
            if meta_entry["ggChemStyle"]:
                boosts = maps["chem_style_boosts_map"].get(meta_entry["ggChemStyle"].lower(), {})
                meta_entry["ggAccelType"] = calculate_acceleration_type(
                    get_attribute_with_boost(base_attributes, "attributeAcceleration", boosts),
                    get_attribute_with_boost(base_attributes, "attributeAgility", boosts),
                    get_attribute_with_boost(base_attributes, "attributeStrength", boosts),
                    player_output.get("height")
                )
        player_output["metaRatings"].append(meta_entry)

    for field in fields_to_remove:
        evo_player_def.pop(field, None)
    
    for key, value in evo_player_def.items():
        if key not in player_output: # Add any remaining fields not explicitly handled
            player_output[key] = value
            
    return player_output


# --- esMeta Prediction for Evo Players (Adapted from user's 4 scripts) ---
def prepare_evo_for_es_prediction(intermediate_evo_players, maps):
    """Prepares evo player data for esMeta prediction."""
    processed_for_df = []
    
    accelerate_types_list = [
        "EXPLOSIVE", "MOSTLY_EXPLOSIVE", "CONTROLLED_EXPLOSIVE", "CONTROLLED",
        "CONTROLLED_LENGTHY", "MOSTLY_LENGTHY", "LENGTHY"
    ]
    bodytypes_list = list(maps["bodytype_code_map"].values()) # Get from maps to ensure all are covered
    playstyles_list_all = list(maps["playstyles_map"].values()) 

    for player_data_const in intermediate_evo_players:
        player_initial = copy.deepcopy(player_data_const) 

        # Base one-hot encoding for accelerateType and bodytype (without chem boosts)
        current_accel_type = player_initial.get("accelerateType")
        for acc_type in accelerate_types_list:
            player_initial[f"accelerateType_{acc_type}"] = 1 if current_accel_type == acc_type else 0
        
        current_body_type = player_initial.get("bodyType")
        for body_type_val_map in bodytypes_list:
            player_initial[f"bodytype_{body_type_val_map}"] = 1 if current_body_type == body_type_val_map else 0 # Use original name for column

        # Encode playstyles (PS) and playstylesPlus (PS+)
        for ps_name_map_val in playstyles_list_all:
            player_initial[ps_name_map_val] = 0 
        
        current_ps = player_initial.get("PS", [])
        current_ps_plus = player_initial.get("PS+", [])

        for ps_name in current_ps:
            if ps_name in playstyles_list_all:
                 player_initial[ps_name] = 1
        for ps_name in current_ps_plus: 
            if ps_name in playstyles_list_all:
                 player_initial[ps_name] = 2
        
        # Expand for each position, archetype, and chemstyle
        # Use mapped position names for iteration, then get archetypes using the new map
        mapped_positions = player_initial.get("positions", [])
        if not mapped_positions: continue

        for pos_name in mapped_positions: # pos_name is "CB", "ST", etc.
            archetypes_for_pos = maps["position_name_to_archetype_map"].get(pos_name, [])
            if not archetypes_for_pos:
                # print(f"Debug: No archetypes found for position name '{pos_name}' for player {player_initial.get('eaId')}")
                continue

            for archetype_name in archetypes_for_pos:
                # For "None" chemstyle (0 chem)
                chem0_player_row = copy.deepcopy(player_initial) # Start from player_initial with base encodings
                chem0_player_row["position_pred"] = pos_name 
                chem0_player_row["archetype_pred"] = archetype_name
                chem0_player_row["chemstyle_pred"] = "None"
                # Accelerate type is already the base one (subAccelType)
                # One-hot encoding for accel and body type already done on player_initial
                processed_for_df.append(chem0_player_row)

                # For each actual chemstyle (3 chem)
                for chem_style_name_lower, boosts in maps["chem_style_boosts_map"].items():
                    if chem_style_name_lower == "none": continue 

                    boosted_player_row = copy.deepcopy(player_initial) # Start from player_initial
                    boosted_player_row["position_pred"] = pos_name
                    boosted_player_row["archetype_pred"] = archetype_name
                    boosted_player_row["chemstyle_pred"] = chem_style_name_lower.title()
                    
                    temp_base_attributes = {k: v for k,v in player_initial.items() if k.startswith("attribute")}

                    for attr, boost_val in boosts.items():
                        if attr in boosted_player_row : 
                             boosted_player_row[attr] = get_attribute_with_boost(temp_base_attributes, attr, {attr: boost_val}, temp_base_attributes.get(attr))
                    
                    boosted_accel_type = calculate_acceleration_type(
                        boosted_player_row.get("attributeAcceleration"),
                        boosted_player_row.get("attributeAgility"),
                        boosted_player_row.get("attributeStrength"),
                        boosted_player_row.get("height")
                    )
                    boosted_player_row["accelerateType"] = boosted_accel_type # Store the boosted accel type
                    for acc_type_l in accelerate_types_list: # Re-one-hot this
                         boosted_player_row[f"accelerateType_{acc_type_l}"] = 1 if boosted_accel_type == acc_type_l else 0
                    
                    processed_for_df.append(boosted_player_row)
    
    return pd.DataFrame(processed_for_df)


def predict_evo_es_meta(df_evo_pred_ready, models_base_dir):
    """Predicts esMeta ratings using Lasso models."""
    predictions_list = []
    cols_to_drop_before_pred = [
        'eaId', 'commonName', 'positions', 'PS', 'PS+', 'roles+', 'roles++', 'metaRatings',
        'position_pred', 'archetype_pred', 'chemstyle_pred', 'subAccelType', 'evolution',
        'accelerateType', 'bodyType', 'numeric_position_ids_evo', # Added this
        # Potentially other non-feature columns that might have been added
    ]
    # Add specific one-hot encoded columns if they are not features but were created temporarily
    # For example, if 'accelerateType_CONTROLLED' itself is a feature, it should NOT be in cols_to_drop_before_pred.
    # The key is that `input_row_df = input_row_df[expected_features]` will select the correct ones.

    for idx, row_series in df_evo_pred_ready.iterrows():
        archetype = row_series['archetype_pred']
        model_path = models_base_dir / f"{archetype}_lasso.pkl"
        
        pred_data = {
            "eaId": row_series["eaId"],
            "archetype": archetype,
            "chemstyle": row_series["chemstyle_pred"],
            "predicted_meta_rating": None,
            "accelerateType_chem": row_series.get("accelerateType") # Accel type for this specific row (could be base or boosted)
        }

        if not model_path.exists():
            # print(f"[WARN] Model for archetype '{archetype}' not found at {model_path}. Skipping for {row_series['eaId']}.")
            predictions_list.append(pred_data)
            continue

        try:
            loaded_model_data = joblib.load(model_path)
            model = loaded_model_data["model"]
            scaler = loaded_model_data["scaler"]
            expected_features = loaded_model_data["features"] 
            target_scaler = loaded_model_data["target_scaler"]

            # Prepare the input row for the model
            # Start with a copy of the series, convert to DataFrame row
            input_row_for_model = row_series.copy()
            
            # Drop columns not needed for prediction explicitly
            # This ensures that only potential features remain
            for col_to_drop in cols_to_drop_before_pred:
                if col_to_drop in input_row_for_model:
                    input_row_for_model = input_row_for_model.drop(col_to_drop)
            
            input_row_df = input_row_for_model.to_frame().T
            
            # Align with expected features: add missing as 0, and reorder/select
            for feature in expected_features:
                if feature not in input_row_df.columns:
                    input_row_df[feature] = 0 # Add missing features with 0
            input_row_df = input_row_df[expected_features] # Ensure correct order and selection

            input_row_scaled = pd.DataFrame(scaler.transform(input_row_df), columns=expected_features)
            
            pred_scaled = model.predict(input_row_scaled)[0]
            pred_original_scale = target_scaler.inverse_transform([[pred_scaled]])[0][0]
            pred_data["predicted_meta_rating"] = round(pred_original_scale, 2)

        except Exception as e:
            # print(f"❌ Error predicting for eaId {row_series.get('eaId')}, archetype {archetype}, chem {row_series.get('chemstyle_pred')}: {e}")
            pass 
        
        predictions_list.append(pred_data)

    return pd.DataFrame(predictions_list)


def inject_evo_es_meta(intermediate_evo_players, df_predictions, maps):
    """Structures and injects predicted esMeta into evo player objects."""
    ratings_summary = {} 

    for _, row in df_predictions.iterrows():
        ea_id = row["eaId"]
        archetype = row["archetype"]
        chemstyle = row["chemstyle"]
        rating = row["predicted_meta_rating"]
        accel_type_with_chem = row["accelerateType_chem"]

        if pd.isna(rating):
            continue

        if ea_id not in ratings_summary:
            ratings_summary[ea_id] = {}
        if archetype not in ratings_summary[ea_id]:
            ratings_summary[ea_id][archetype] = {
                "esMetaSub": None, "esAccelType_0": None, 
                "esMeta": None, "esChemStyle": None, "esAccelType_3": None 
            }
        
        entry = ratings_summary[ea_id][archetype]

        if chemstyle.lower() == "none":
            entry["esMetaSub"] = rating
            entry["esAccelType_0"] = accel_type_with_chem 
        else:
            if entry["esMeta"] is None or rating > entry["esMeta"]:
                entry["esMeta"] = rating
                entry["esChemStyle"] = chemstyle 
                entry["esAccelType_3"] = accel_type_with_chem
    
    for evo_player in intermediate_evo_players:
        ea_id = evo_player["eaId"]
        if ea_id in ratings_summary:
            for meta_rating_entry in evo_player.get("metaRatings", []):
                role_name = meta_rating_entry["role"]
                archetype_for_role = maps["role_name_to_archetype_map"].get(role_name)
                if archetype_for_role and archetype_for_role in ratings_summary[ea_id]:
                    es_summary = ratings_summary[ea_id][archetype_for_role]
                    meta_rating_entry["esMetaSub"] = es_summary["esMetaSub"]
                    meta_rating_entry["esMeta"] = es_summary["esMeta"]
                    meta_rating_entry["esChemStyle"] = es_summary["esChemStyle"]
                    meta_rating_entry["esAccelType"] = es_summary["esAccelType_3"] 
                    # subAccelType (base 0-chem accel) is already in meta_rating_entry from initial processing
                    # If esMetaSub is present, its esAccelType_0 should ideally match subAccelType.
                    # The current subAccelType in meta_entry is player's base, which is correct for 0-chem.
                    # esAccelType_0 from prediction also represents 0-chem accel type.

# --- Main Processing Logic ---
def main():
    print("🚀 Starting Enhanced Player Data Processing Script...")
    
    maps_data_full = load_json_file(MAPS_FILE)
    if not maps_data_full:
        print(f"❌ Critical error: Could not load maps.json from {MAPS_FILE}. Exiting.")
        return

    # Create position_name_to_archetype_map
    position_id_to_name_map_local = maps_data_full.get("position", {})
    id_to_archetypes_raw = maps_data_full.get("positionIdToArchetype", {})
    name_to_archetypes_map = {}
    for pos_id, archetypes in id_to_archetypes_raw.items():
        pos_name = position_id_to_name_map_local.get(pos_id)
        if pos_name:
            name_to_archetypes_map[pos_name] = archetypes
    
    maps_for_processing = {
        "position_map": maps_data_full.get("position", {}),
        "foot_map": maps_data_full.get("foot", {}),
        "playstyles_map": maps_data_full.get("playstyles", {}),
        "roles_plus_map": maps_data_full.get("rolesPlus", {}),
        "roles_plus_plus_map": maps_data_full.get("rolesPlusPlus", {}),
        "bodytype_code_map": maps_data_full.get("bodytypeCode", {}),
        "gg_chem_style_names_map": maps_data_full.get("ggChemistryStyleNames", {}),
        "es_chem_style_names_map": maps_data_full.get("esChemistryStyleNames", {}),
        "chem_style_boosts_map": {item['name'].lower(): item['threeChemistryModifiers']
                                  for item in maps_data_full.get("ChemistryStylesBoosts", []) if 'name' in item},
        "role_id_to_name_map": maps_data_full.get("role", {}),
        "role_name_to_archetype_map": maps_data_full.get("roleToArchetype", {}),
        "position_name_to_archetype_map": name_to_archetypes_map # Added this new map
    }

    processed_players_list = []

    # 1. Process Club Players
    print("\n--- Processing Club Players ---")
    if not GG_DATA_DIR.exists():
        print(f"❌ Error: Fut.gg data directory for club players not found: {GG_DATA_DIR}")
    else:
        club_player_ea_ids = [filename.split('_')[0] for filename in os.listdir(GG_DATA_DIR) if filename.endswith("_ggData.json")]
        print(f"ℹ️ Found {len(club_player_ea_ids)} club players to process.")
        for i, ea_id_str in enumerate(club_player_ea_ids):
            player_data = process_single_club_player(ea_id_str, GG_DATA_DIR, GG_META_DIR, ES_META_DIR, maps_for_processing, CLUB_PLAYER_FUTGG_FIELDS_TO_REMOVE)
            if player_data:
                processed_players_list.append(player_data)
        print(f"✅ Finished processing {len(processed_players_list)} club players (out of {len(club_player_ea_ids)} found).")


    # 2. Process Evo Players
    print("\n--- Processing Evo Players ---")
    evolab_data_full = load_json_file(EVOLAB_FILE)
    if evolab_data_full and "data" in evolab_data_full:
        raw_evo_player_definitions = [item["playerItemDefinition"] for item in evolab_data_full["data"] if "playerItemDefinition" in item]
        print(f"ℹ️ Found {len(raw_evo_player_definitions)} evo player definitions to process.")

        intermediate_evo_players = []
        for i, evo_raw_def in enumerate(raw_evo_player_definitions):
            evo_player = process_single_evo_player_basic_and_ggmeta(evo_raw_def, maps_for_processing, EVO_PLAYER_DEFINITION_FIELDS_TO_REMOVE)
            if evo_player:
                intermediate_evo_players.append(evo_player)
        print(f"✅ Finished basic processing and ggMeta for {len(intermediate_evo_players)} evo players.")
        
        if intermediate_evo_players:
            print("\n--- Preparing Evo Players for esMeta Prediction ---")
            df_evo_for_prediction = prepare_evo_for_es_prediction(intermediate_evo_players, maps_for_processing)
            print(f"ℹ️ Prepared {len(df_evo_for_prediction)} rows for esMeta prediction.")

            if not df_evo_for_prediction.empty:
                print("\n--- Predicting esMeta for Evo Players ---")
                df_evo_with_es_predictions = predict_evo_es_meta(df_evo_for_prediction, MODELS_DIR) 
                print(f"ℹ️ Received {len(df_evo_with_es_predictions)} prediction results.")

                print("\n--- Injecting Predicted esMeta into Evo Players ---")
                inject_evo_es_meta(intermediate_evo_players, df_evo_with_es_predictions, maps_for_processing)
                print("✅ Finished injecting esMeta for evo players.")
            else:
                print("ℹ️ No evo player data prepared for esMeta prediction. Skipping prediction step.")
            
            processed_players_list.extend(intermediate_evo_players)
        else:
            print("ℹ️ No valid evo players after basic processing. Skipping esMeta prediction.")
    else:
        print(f"⚠️ Evolab file not found or has no data: {EVOLAB_FILE}")
    
    # 3. Deduplicate players: Keep evo version if both club and evo exist for the same eaId
    print("\n--- Deduplicating Players ---")
    evo_player_ea_ids = {player['eaId'] for player in processed_players_list if player.get('evolution') is True}
    
    final_deduplicated_list = []
    club_duplicates_removed_count = 0
    for player in processed_players_list:
        if player.get('evolution') is True:
            final_deduplicated_list.append(player)
        else: # It's a club player
            if player['eaId'] not in evo_player_ea_ids:
                final_deduplicated_list.append(player)
            else:
                club_duplicates_removed_count +=1
                # print(f"ℹ️ Club player {player['eaId']} removed due to existing evo version.")
    
    print(f"ℹ️ Removed {club_duplicates_removed_count} club player duplicates.")
    processed_players_list = final_deduplicated_list # Replace with the deduplicated list

    # 4. Save Output
    print("\n--- Saving Final Output ---")
    try:
        if not BASE_DATA_DIR.exists():
            BASE_DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(processed_players_list, f, indent=2, ensure_ascii=False)
        print(f"\n🎉 Successfully processed and deduplicated a total of {len(processed_players_list)} players.")
        print(f"💾 Final combined data saved to: {OUTPUT_FILE}")
    except Exception as e:
        print(f"❌ Error saving final output file: {e}")

if __name__ == "__main__":
    main()
