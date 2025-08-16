import json
from pathlib import Path
import os
from collections import defaultdict
import copy
import pandas as pd
import joblib
import warnings

# Suppress the harmless FutureWarning from pandas
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Configuration ---
BASE_DATA_DIR = Path(__file__).resolve().parent / '../data'
MODELS_DIR = Path(__file__).resolve().parent / '../models'
RAW_DATA_DIR = BASE_DATA_DIR / 'raw'
GG_DATA_DIR = RAW_DATA_DIR / 'ggData'
ES_META_DIR = RAW_DATA_DIR / 'esMeta'
GG_META_DIR = RAW_DATA_DIR / 'ggMeta'
EVOLAB_FILE = BASE_DATA_DIR / 'evolab.json'
MAPS_FILE = BASE_DATA_DIR / 'maps.json'
OUTPUT_FILE = BASE_DATA_DIR / 'club_final.json'
CLUB_IDS_FILE = BASE_DATA_DIR / 'club_ids.json'

# --- Fields to remove ---
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
EVO_PLAYER_DEFINITION_FIELDS_TO_REMOVE = [
    "playerType", "game", "id", "evolutionId", "cosmeticEvolutionId", "partialEvolutionId",
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
    try:
        with open(file_path, 'r', encoding='utf-8') as f: return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError): return default_val

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

def parse_gg_rating_str(gg_rating_str_raw):
    parsed_ratings_by_role = defaultdict(list)
    if not gg_rating_str_raw: return parsed_ratings_by_role
    for part in gg_rating_str_raw.split('||'):
        try:
            chem_id_str, role_id_str, score_str = part.split(':')
            parsed_ratings_by_role[role_id_str].append({"chem_id_str": chem_id_str, "score": float(score_str)})
        except (ValueError, IndexError): continue
    return parsed_ratings_by_role

# --- Model Prediction Engine ---
class ModelManager:
    def __init__(self, models_dir):
        self.models_dir = models_dir
        self.models = self._load_models()

    def _load_models(self):
        models = {}
        if not self.models_dir.exists():
            print(f"⚠️ Models directory not found: {self.models_dir}")
            return models
        
        for pkl_file in self.models_dir.glob("*.pkl"):
            try:
                models[pkl_file.stem] = joblib.load(pkl_file)
            except Exception as e:
                print(f"⚠️ Error loading model {pkl_file.name}: {e}")
        print(f"✅ Loaded {len(models)} models from disk.")
        return models

    def get_model(self, role_name, model_type):
        safe_role_name = role_name.replace(" ", "_").replace("-", "_")
        model_key = f"{safe_role_name}_{model_type}_model"
        return self.models.get(model_key)

def prepare_features(player_data, maps, boosts={}):
    features = {}
    base_attributes = {k: v for k, v in player_data.items() if k.startswith("attribute")}
    
    for attr in base_attributes:
        features[attr] = get_attribute_with_boost(base_attributes, attr, boosts)
    
    for key in ["height", "weight", "skillMoves", "weakFoot"]:
        features[key] = player_data.get(key, 0)

    all_playstyles = list(maps.get("playstyles", {}).values())
    player_ps = set(player_data.get("PS", []))
    player_ps_plus = set(player_data.get("PS+", []))
    for ps in all_playstyles:
        features[ps] = 2 if ps in player_ps_plus else 1 if ps in player_ps else 0
    
    features["bodytype"] = player_data.get("bodyType")
    features["accelerateType"] = calculate_acceleration_type(
        features.get("attributeAcceleration"), features.get("attributeAgility"),
        features.get("attributeStrength"), features.get("height")
    )
    return features

def predict_rating(model_bundle, feature_dict):
    if not model_bundle: return None
    try:
        input_df = pd.DataFrame([feature_dict])
        input_df = pd.get_dummies(input_df, columns=["bodytype", "accelerateType", "foot"], dtype=int)
        model_features = model_bundle["features"]
        for feature in model_features:
            if feature not in input_df.columns: input_df[feature] = 0
        input_df = input_df[model_features]
        
        input_df = input_df.fillna(0).infer_objects(copy=False)

        input_scaled = model_bundle["feature_scaler"].transform(input_df)
        prediction = model_bundle["model"].predict(input_scaled)
        
        if "target_scaler" in model_bundle:
            prediction = model_bundle["target_scaler"].inverse_transform(prediction.reshape(-1, 1))[0][0]
        else:
            prediction = prediction[0]

        return round(float(prediction), 2)
    except Exception: return None

def predict_and_inject_ratings(player_output, model_manager, maps):
    for meta_entry in player_output.get("metaRatings", []):
        role_name = meta_entry["role"]
        
        gg_sub_model = model_manager.get_model(role_name, 'ggMetaSub')
        if gg_sub_model:
            base_features = prepare_features(player_output, maps, boosts={})
            gg_sub_pred = predict_rating(gg_sub_model, base_features)
            if gg_sub_pred: meta_entry["ggMetaSub"] = gg_sub_pred
        
        elif "GK" in role_name and meta_entry.get("ggMeta") is not None:
            meta_entry["ggMetaSub"] = round(meta_entry["ggMeta"] * 0.9, 2)

        if player_output.get("evolution"):
            es_sub_model = model_manager.get_model(role_name, 'esMetaSub')
            if es_sub_model:
                base_features = prepare_features(player_output, maps, boosts={})
                es_sub_pred = predict_rating(es_sub_model, base_features)
                if es_sub_pred: meta_entry["esMetaSub"] = es_sub_pred

            es_model = model_manager.get_model(role_name, 'esMeta')
            if es_model:
                best_esMeta, best_esChem, best_esAccel = 0, "basic", base_features.get("accelerateType")
                for chem_name, boosts in maps["chem_style_boosts_map"].items():
                    boosted_features = prepare_features(player_output, maps, boosts=boosts)
                    prediction = predict_rating(es_model, boosted_features)
                    if prediction and prediction > best_esMeta:
                        best_esMeta, best_esChem, best_esAccel = prediction, chem_name.title(), boosted_features.get("accelerateType")
                
                meta_entry["esMeta"], meta_entry["esChemStyle"], meta_entry["esAccelType"] = (best_esMeta if best_esMeta > 0 else None), best_esChem, best_esAccel
        
        gg_meta, es_meta = meta_entry.get("ggMeta"), meta_entry.get("esMeta")
        gg_meta_sub, es_meta_sub = meta_entry.get("ggMetaSub"), meta_entry.get("esMetaSub")

        if gg_meta is not None and es_meta is not None:
            meta_entry["avgMeta"] = round((gg_meta + es_meta) / 2, 2)
        else:
            meta_entry["avgMeta"] = gg_meta or es_meta

        if gg_meta_sub is not None and es_meta_sub is not None:
            meta_entry["avgMetaSub"] = round((gg_meta_sub + es_meta_sub) / 2, 2)
        else:
            meta_entry["avgMetaSub"] = gg_meta_sub or es_meta_sub
            
    return player_output

# --- Player Processing Functions ---
def process_single_club_player(ea_id_str, model_manager, maps):
    ea_id = int(ea_id_str)
    gg_details_raw = load_json_file(GG_DATA_DIR / f"{ea_id}_ggData.json")
    if not (gg_details_raw and "data" in gg_details_raw): return None
    
    gg_player_data = gg_details_raw["data"]
    player_output = {"eaId": ea_id, "evolution": False} 
    base_attributes = {k: v for k, v in gg_player_data.items() if k.startswith("attribute")}
    for key in ["commonName", "overall", "height", "weight", "skillMoves", "weakFoot"]:
        player_output[key] = gg_player_data.get(key)
    player_output.update(base_attributes)
    
    player_output["positions"] = list(set([maps["position_map"].get(str(p)) for p in [gg_player_data.get("position")] + gg_player_data.get("alternativePositionIds", []) if p is not None]))
    player_output["foot"] = maps["foot_map"].get(str(gg_player_data.get("foot")))
    player_output["PS"] = [maps["playstyles_map"].get(str(p)) for p in gg_player_data.get("playstyles", []) if str(p) in maps["playstyles_map"]]
    player_output["PS+"] = [maps["playstyles_map"].get(str(p)) for p in gg_player_data.get("playstylesPlus", []) if str(p) in maps["playstyles_map"]]
    player_output["roles+"] = [maps["roles_plus_map"].get(str(r)) for r in gg_player_data.get("rolesPlus", []) if str(r) in maps["roles_plus_map"]]
    player_output["roles++"] = [maps["roles_plus_plus_map"].get(str(r)) for r in gg_player_data.get("rolesPlusPlus", []) if str(r) in maps["roles_plus_plus_map"]]
    player_output["bodyType"] = maps["bodytype_code_map"].get(str(gg_player_data.get("bodytypeCode")))
    player_output["accelerateType"] = gg_player_data.get("accelerateType")

    sub_accel_type = calculate_acceleration_type(
        base_attributes.get("attributeAcceleration"), base_attributes.get("attributeAgility"),
        base_attributes.get("attributeStrength"), player_output.get("height")
    )

    gg_meta_raw = load_json_file(GG_META_DIR / f"{ea_id}_ggMeta.json")
    es_meta_raw_list = load_json_file(ES_META_DIR / f"{ea_id}_esMeta.json", [])
    player_output["metaRatings"] = []

    unique_gg_role_ids = set()
    if gg_meta_raw and gg_meta_raw.get("data", {}).get("scores"):
        for score_entry in gg_meta_raw["data"]["scores"]:
            unique_gg_role_ids.add(str(score_entry.get("role")))

    for gg_role_id_str in unique_gg_role_ids:
        role_name = maps["role_id_to_name_map"].get(gg_role_id_str, f"UnknownRoleID_{gg_role_id_str}")
        meta_entry = {
            "role": role_name, "ggMeta": None, "ggMetaSub": None, "ggChemStyle": None, "ggAccelType": None,
            "esMetaSub": None, "esMeta": None, "esChemStyle": None, "esAccelType": None,
            "subAccelType": sub_accel_type, "avgMeta": None, "avgMetaSub": None
        }

        if gg_meta_raw and gg_meta_raw.get("data", {}).get("scores"):
            best_gg_score = max((s for s in gg_meta_raw["data"]["scores"] if str(s.get("role")) == gg_role_id_str), 
                                key=lambda x: x.get("score", 0), default=None)
            if best_gg_score:
                meta_entry["ggMeta"] = round(best_gg_score["score"], 2)
                gg_chem_id = str(best_gg_score.get("chemistryStyle"))
                meta_entry["ggChemStyle"] = maps["gg_chem_style_names_map"].get(gg_chem_id)
                if meta_entry["ggChemStyle"]:
                    boosts = maps["chem_style_boosts_map"].get(meta_entry["ggChemStyle"].lower(), {})
                    meta_entry["ggAccelType"] = calculate_acceleration_type(
                        get_attribute_with_boost(base_attributes, "attributeAcceleration", boosts),
                        get_attribute_with_boost(base_attributes, "attributeAgility", boosts),
                        get_attribute_with_boost(base_attributes, "attributeStrength", boosts),
                        player_output.get("height")
                    )
        
        es_player_role_id_to_find = maps["roleNameToEsRoleId"].get(role_name)
        if es_player_role_id_to_find and es_meta_raw_list:
            es_data_for_role = next((b for b in es_meta_raw_list if any(str(r.get("playerRoleId")) == es_player_role_id_to_find for r in b.get("data", {}).get("metaRatings", []))), None)
            if es_data_for_role:
                all_es_ratings = es_data_for_role["data"]["metaRatings"]
                specific_es_ratings = [r for r in all_es_ratings if str(r.get("playerRoleId")) == str(es_player_role_id_to_find)]
                if specific_es_ratings:
                    rating_0_chem = next((r for r in specific_es_ratings if r.get("chemistry") == 0), None)
                    if rating_0_chem: meta_entry["esMetaSub"] = round(rating_0_chem["metaRating"], 2)
                    
                    best_3_chem = max([r for r in specific_es_ratings if r.get("chemistry") == 3 and r.get("isBestChemstyleAtChem")], key=lambda x: x.get("metaRating", 0), default=None)
                    if best_3_chem:
                        meta_entry["esMeta"] = round(best_3_chem["metaRating"], 2)
                        es_chem_id = str(best_3_chem.get("chemstyleId"))
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
    
    return predict_and_inject_ratings(player_output, model_manager, maps)

def process_single_evo_player(evo_player_def_raw, model_manager, maps):
    if not evo_player_def_raw or evo_player_def_raw.get("eaId") is None: return None
    
    player_output = {"evolution": True, "eaId": evo_player_def_raw.get("eaId")}
    base_attributes = {k:v for k,v in evo_player_def_raw.items() if k.startswith("attribute")}
    for key in ["commonName", "overall", "height", "weight", "skillMoves", "weakFoot"]:
        player_output[key] = evo_player_def_raw.get(key)
    player_output.update(base_attributes)

    numeric_positions = [str(p) for p in [evo_player_def_raw.get("position")] + evo_player_def_raw.get("alternativePositionIds", []) if p is not None]
    player_output["positions"] = list(set([maps["position_map"].get(p) for p in numeric_positions if p in maps["position_map"]]))
    player_output["foot"] = maps["foot_map"].get(str(evo_player_def_raw.get("foot")))
    player_output["PS"] = [maps["playstyles_map"].get(str(p)) for p in evo_player_def_raw.get("playstyles", []) if str(p) in maps["playstyles_map"]]
    player_output["PS+"] = [maps["playstyles_map"].get(str(p)) for p in evo_player_def_raw.get("playstylesPlus", []) if str(p) in maps["playstyles_map"]]
    player_output["roles+"] = [maps["roles_plus_map"].get(str(r)) for r in evo_player_def_raw.get("rolesPlus", []) if str(r) in maps["roles_plus_map"]]
    player_output["roles++"] = [maps["roles_plus_plus_map"].get(str(r)) for r in evo_player_def_raw.get("rolesPlusPlus", []) if str(r) in maps["roles_plus_plus_map"]]
    player_output["bodyType"] = maps["bodytype_code_map"].get(str(evo_player_def_raw.get("bodytypeCode")))
    
    sub_accel_type = calculate_acceleration_type(
        base_attributes.get("attributeAcceleration"), base_attributes.get("attributeAgility"),
        base_attributes.get("attributeStrength"), player_output.get("height")
    )
    
    player_output["metaRatings"] = []
    parsed_gg_ratings = parse_gg_rating_str(evo_player_def_raw.get("ggRatingStr"))
    
    for role_id_str, ratings in parsed_gg_ratings.items():
        role_name = maps["role_id_to_name_map"].get(role_id_str)
        if not role_name: continue

        meta_entry = { "role": role_name, "ggMeta": None, "ggChemStyle": None, "ggAccelType": None, "avgMeta": None, "avgMetaSub": None, "subAccelType": sub_accel_type }
        best_gg_rating = max(ratings, key=lambda x: x["score"], default=None)
        if best_gg_rating:
            meta_entry["ggMeta"] = round(best_gg_rating["score"], 2)
            meta_entry["ggChemStyle"] = maps["gg_chem_style_names_map"].get(best_gg_rating["chem_id_str"])
            if meta_entry["ggChemStyle"]:
                boosts = maps["chem_style_boosts_map"].get(meta_entry["ggChemStyle"].lower(), {})
                meta_entry["ggAccelType"] = calculate_acceleration_type(
                    get_attribute_with_boost(base_attributes, "attributeAcceleration", boosts),
                    get_attribute_with_boost(base_attributes, "attributeAgility", boosts),
                    get_attribute_with_boost(base_attributes, "attributeStrength", boosts),
                    player_output.get("height")
                )
        player_output["metaRatings"].append(meta_entry)

    return predict_and_inject_ratings(player_output, model_manager, maps)

def main():
    print("🚀 Starting Player Data Processing Script...")
    maps_data = load_json_file(MAPS_FILE)
    if not maps_data:
        print(f"❌ Critical error: Could not load {MAPS_FILE}. Exiting.")
        return
    
    maps = {
        "position_map": maps_data.get("position", {}), "foot_map": maps_data.get("foot", {}),
        "playstyles_map": maps_data.get("playstyles", {}), "roles_plus_map": maps_data.get("rolesPlus", {}),
        "roles_plus_plus_map": maps_data.get("rolesPlusPlus", {}), "bodytype_code_map": maps_data.get("bodytypeCode", {}),
        "gg_chem_style_names_map": maps_data.get("ggChemistryStyleNames", {}), "es_chem_style_names_map": maps_data.get("esChemistryStyleNames", {}),
        "chem_style_boosts_map": {item['name'].lower(): item['threeChemistryModifiers'] for item in maps_data.get("ChemistryStylesBoosts", []) if 'name' in item},
        "role_id_to_name_map": maps_data.get("role", {}), "esRoleId": maps_data.get("esRoleId", {}),
        "roleNameToEsRoleId": maps_data.get("roleNameToEsRoleId", {})
    }

    model_manager = ModelManager(MODELS_DIR)

    processed_players = []
    print("\n--- Processing Club Players ---")
    club_ids_data = load_json_file(CLUB_IDS_FILE)
    club_player_ids = [str(id) for id in club_ids_data] if isinstance(club_ids_data, list) else []
    print(f"ℹ️ Found {len(club_player_ids)} club players to process from {CLUB_IDS_FILE.name}.")
    for i, ea_id in enumerate(club_player_ids):
        if (i + 1) % 100 == 0: print(f"  - Player {i+1}/{len(club_player_ids)}")
        player = process_single_club_player(ea_id, model_manager, maps)
        if player: processed_players.append(player)
    
    print("\n--- Processing Evo Players ---")
    evolab_data = load_json_file(EVOLAB_FILE)
    if evolab_data and "data" in evolab_data:
        evo_defs = [item["playerItemDefinition"] for item in evolab_data["data"] if "playerItemDefinition" in item]
        print(f"ℹ️ Found {len(evo_defs)} evo player definitions.")
        for i, evo_def in enumerate(evo_defs):
            if (i + 1) % 100 == 0: print(f"  - Evo {i+1}/{len(evo_defs)}")
            player = process_single_evo_player(evo_def, model_manager, maps)
            if player: processed_players.append(player)
    
    print("\n--- Scaling Predicted ggMetaSub Ratings ---")
    if processed_players:
        # Flatten the data into a DataFrame for easy manipulation
        flat_data = []
        all_meta_keys = set()
        for player in processed_players:
            for rating_entry in player.get('metaRatings', []):
                all_meta_keys.update(rating_entry.keys())
                row = {**{k: v for k, v in player.items() if k != 'metaRatings'}, **rating_entry}
                flat_data.append(row)
        
        df = pd.DataFrame(flat_data)

        TARGET_MAX_SUB_RATING = 95.0
        scaled_dfs = []
        for role, group in df.groupby('role'):
            if 'ggMetaSub' in group.columns and group['ggMetaSub'].max() > 0:
                current_max = group['ggMetaSub'].max()
                scaling_factor = TARGET_MAX_SUB_RATING / current_max if current_max > 0 else 1
                print(f"  - Scaling role '{role}': Max predicted was {current_max:.2f}. Factor is {scaling_factor:.2f}.")
                group['ggMetaSub'] = (group['ggMetaSub'] * scaling_factor).round(2)
            
            gg_sub = group.get('ggMetaSub', pd.Series(0, index=group.index)).fillna(0)
            es_sub = group.get('esMetaSub', pd.Series(0, index=group.index)).fillna(0)
            avg_sub = pd.concat([gg_sub, es_sub], axis=1)
            group['avgMetaSub'] = avg_sub.apply(lambda x: (x[0] + x[1]) / 2 if x[0] > 0 and x[1] > 0 else x[0] or x[1], axis=1).round(2)
            scaled_dfs.append(group)

        if scaled_dfs:
            df_scaled = pd.concat(scaled_dfs)
            final_players_dict = {}
            
            for _, row in df_scaled.iterrows():
                player_id = row['eaId']
                if player_id not in final_players_dict:
                    player_info = {k: v for k, v in row.items() if k not in all_meta_keys}
                    player_info['metaRatings'] = []
                    final_players_dict[player_id] = player_info
                
                meta_rating_info = {k: v for k, v in row.items() if k in all_meta_keys}
                final_players_dict[player_id]['metaRatings'].append(meta_rating_info)
            processed_players = list(final_players_dict.values())

    print("\n--- Deduplicating Players ---")
    final_players_dedup = {}
    for player in processed_players:
        if player['eaId'] not in final_players_dedup or player.get('evolution'):
            final_players_dedup[player['eaId']] = player
    final_list = list(final_players_dedup.values())
    print(f"ℹ️ Total unique players: {len(final_list)}")

    print("\n--- Saving Final Output ---")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_list, f, indent=2, ensure_ascii=False)
    print(f"\n🎉 Success! Processed {len(final_list)} players to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
