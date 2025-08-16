import json
from pathlib import Path
import os
from collections import defaultdict
import copy
import pandas as pd
import joblib
import warnings

# Suppress harmless warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

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

# --- Model Prediction Engine (for esMeta on Evos ONLY) ---
class ModelManager:
    def __init__(self, models_dir):
        self.models = self._load_models(models_dir)

    def _load_models(self, models_dir):
        models = {}
        if not models_dir.exists():
            print(f"⚠️ Models directory not found: {models_dir}")
            return models
        for pkl_file in models_dir.glob("*_esMeta*_model.pkl"):
            try:
                models[pkl_file.stem] = joblib.load(pkl_file)
            except Exception as e:
                print(f"⚠️ Error loading model {pkl_file.name}: {e}")
        print(f"✅ Loaded {len(models)} esMeta models from disk.")
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
    for key in ["height", "weight", "skillMoves", "weakFoot", "foot"]:
        features[key] = player_data.get(key)
    all_playstyles = list(maps.get("playstyles", {}).values())
    player_ps = set(player_data.get("PS", [])); player_ps_plus = set(player_data.get("PS+", []))
    for ps in all_playstyles:
        features[ps] = 2 if ps in player_ps_plus else 1 if ps in player_ps else 0
    features["bodytype"] = player_data.get("bodyType")
    features["accelerateType"] = calculate_acceleration_type(features.get("attributeAcceleration"), features.get("attributeAgility"), features.get("attributeStrength"), features.get("height"))
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
        pred_scaled = model_bundle["model"].predict(input_scaled)
        
        if "target_scaler" in model_bundle:
            prediction = model_bundle["target_scaler"].inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
        else:
            prediction = pred_scaled[0]

        return round(float(prediction), 2)
    except Exception as e:
        print(f"⚠️ Prediction failed for {feature_dict.get('commonName')}: {e}")
        return None

# --- Main Player Processing Function ---
def process_player(player_def, is_evo, model_manager, maps):
    player_output = {"eaId": player_def.get("eaId"), "evolution": is_evo}
    base_attributes = {k: v for k, v in player_def.items() if k.startswith("attribute")}
    
    for key in ["commonName", "overall", "height", "weight", "skillMoves", "weakFoot"]:
        player_output[key] = player_def.get(key)
    player_output.update(base_attributes)

    numeric_positions = [str(p) for p in [player_def.get("position")] + player_def.get("alternativePositionIds", []) if p is not None]
    player_output["positions"] = list(set([maps["position_map"].get(p) for p in numeric_positions if p in maps["position_map"]]))
    player_output["foot"] = maps["foot_map"].get(str(player_def.get("foot")))
    player_output["PS"] = [maps["playstyles_map"].get(str(p)) for p in player_def.get("playstyles", []) if str(p) in maps["playstyles_map"]]
    player_output["PS+"] = [maps["playstyles_map"].get(str(p)) for p in player_def.get("playstylesPlus", []) if str(p) in maps["playstyles_map"]]
    player_output["roles+"] = [maps["roles_plus_map"].get(str(r)) for r in player_def.get("rolesPlus", []) if str(r) in maps["roles_plus_map"]]
    player_output["roles++"] = [maps["roles_plus_plus_map"].get(str(r)) for r in player_def.get("rolesPlusPlus", []) if str(r) in maps["roles_plus_plus_map"]]
    player_output["bodyType"] = maps["bodytype_code_map"].get(str(player_def.get("bodytypeCode")))
    
    sub_accel_type = calculate_acceleration_type(base_attributes.get("attributeAcceleration"), base_attributes.get("attributeAgility"), base_attributes.get("attributeStrength"), player_output.get("height"))

    es_meta_raw = load_json_file(ES_META_DIR / f"{player_output['eaId']}_esMeta.json", [])
    gg_meta_raw = load_json_file(GG_META_DIR / f"{player_output['eaId']}_ggMeta.json")
    
    gg_scores_by_role = defaultdict(list)
    if gg_meta_raw and "data" in gg_meta_raw and "scores" in gg_meta_raw["data"]:
        for score in gg_meta_raw["data"]["scores"]:
            gg_scores_by_role[str(score.get("role"))].append(score)

    player_output["metaRatings"] = []
    for role_id_str, scores in gg_scores_by_role.items():
        role_name = maps["role_id_to_name_map"].get(role_id_str)
        if not role_name: continue

        meta_entry = {"role": role_name, "subAccelType": sub_accel_type}

        best_gg_score = max(scores, key=lambda x: x.get("score", 0), default=None)
        if best_gg_score:
            meta_entry["ggMeta"] = round(best_gg_score["score"], 2)
            chem_id = str(best_gg_score.get("chemistryStyle"))
            meta_entry["ggChemStyle"] = maps["gg_chem_style_names_map"].get(chem_id)
            boosts = maps["chem_style_boosts_map"].get(meta_entry.get("ggChemStyle", "").lower(), {})
            meta_entry["ggAccelType"] = calculate_acceleration_type(get_attribute_with_boost(base_attributes, "attributeAcceleration", boosts), get_attribute_with_boost(base_attributes, "attributeAgility", boosts), get_attribute_with_boost(base_attributes, "attributeStrength", boosts), player_output.get("height"))

        basic_gg_score = next((s for s in scores if maps["gg_chem_style_names_map"].get(str(s.get("chemistryStyle")), "").lower() == 'basic'), None)
        if basic_gg_score:
            meta_entry["ggMetaSub"] = round(basic_gg_score.get("score"), 2)

        if is_evo:
            es_sub_model = model_manager.get_model(role_name, 'esMetaSub')
            if es_sub_model:
                base_features = prepare_features(player_output, maps)
                meta_entry["esMetaSub"] = predict_rating(es_sub_model, base_features)
            
            es_model = model_manager.get_model(role_name, 'esMeta')
            if es_model:
                best_esMeta, best_esChem, best_esAccel = 0, "basic", sub_accel_type
                for chem_name, boosts in maps["chem_style_boosts_map"].items():
                    boosted_features = prepare_features(player_output, maps, boosts=boosts)
                    prediction = predict_rating(es_model, boosted_features)
                    if prediction and prediction > best_esMeta:
                        best_esMeta, best_esChem, best_esAccel = prediction, chem_name.title(), boosted_features.get("accelerateType")
                meta_entry["esMeta"], meta_entry["esChemStyle"], meta_entry["esAccelType"] = (best_esMeta if best_esMeta > 0 else None), best_esChem, best_esAccel
        else:
            es_role_id = maps["roleNameToEsRoleId"].get(role_name)
            if es_role_id and es_meta_raw:
                es_role_block = next((b for b in es_meta_raw if any(str(r.get("playerRoleId")) == es_role_id for r in b.get("data", {}).get("metaRatings", []))), None)
                if es_role_block:
                    ratings = [r for r in es_role_block["data"]["metaRatings"] if str(r.get("playerRoleId")) == str(es_role_id)]
                    rating_0_chem = next((r for r in ratings if r.get("chemistry") == 0), None)
                    if rating_0_chem: meta_entry["esMetaSub"] = round(rating_0_chem["metaRating"], 2)
                    best_3_chem = max([r for r in ratings if r.get("chemistry") == 3 and r.get("isBestChemstyleAtChem")], key=lambda x: x.get("metaRating", 0), default=None)
                    if best_3_chem:
                        meta_entry["esMeta"] = round(best_3_chem["metaRating"], 2)
                        chem_id_3 = str(best_3_chem.get("chemstyleId"))
                        meta_entry["esChemStyle"] = maps["es_chem_style_names_map"].get(chem_id_3)
                        boosts_3 = maps["chem_style_boosts_map"].get(meta_entry.get("esChemStyle", "").lower(), {})
                        meta_entry["esAccelType"] = calculate_acceleration_type(get_attribute_with_boost(base_attributes, "attributeAcceleration", boosts_3), get_attribute_with_boost(base_attributes, "attributeAgility", boosts_3), get_attribute_with_boost(base_attributes, "attributeStrength", boosts_3), player_output.get("height"))

        gg_meta, es_meta = meta_entry.get("ggMeta"), meta_entry.get("esMeta")
        gg_meta_sub, es_meta_sub = meta_entry.get("ggMetaSub"), meta_entry.get("esMetaSub")
        meta_entry["avgMeta"] = round((gg_meta + es_meta) / 2, 2) if gg_meta is not None and es_meta is not None else gg_meta or es_meta
        meta_entry["avgMetaSub"] = round((gg_meta_sub + es_meta_sub) / 2, 2) if gg_meta_sub is not None and es_meta_sub is not None else gg_meta_sub or es_meta_sub
        
        player_output["metaRatings"].append(meta_entry)
        
    return player_output

def main():
    print("🚀 Starting Player Data Processing Script...")
    maps_data = load_json_file(MAPS_FILE)
    if not maps_data:
        print("❌ Critical error: Could not load maps.json. Exiting.")
        return
    
    maps = {
        "position_map": maps_data.get("position", {}), "foot_map": maps_data.get("foot", {}),
        "playstyles_map": maps_data.get("playstyles", {}), "roles_plus_map": maps_data.get("rolesPlus", {}),
        "roles_plus_plus_map": maps_data.get("rolesPlusPlus", {}), "bodytype_code_map": maps_data.get("bodytypeCode", {}),
        "gg_chem_style_names_map": maps_data.get("ggChemistryStyleNames", {}), "es_chem_style_names_map": maps_data.get("esChemistryStyleNames", {}),
        "chem_style_boosts_map": {item['name'].lower(): item['threeChemistryModifiers'] for item in maps_data.get("ChemistryStylesBoosts", []) if 'name' in item},
        "role_id_to_name_map": maps_data.get("role", {}),
        "roleNameToEsRoleId": maps_data.get("roleNameToEsRoleId", {})
    }

    model_manager = ModelManager(MODELS_DIR)
    processed_players = []

    print("\n--- Processing Club Players ---")
    club_ids_data = load_json_file(CLUB_IDS_FILE)
    club_player_ids = [str(id) for id in club_ids_data] if isinstance(club_ids_data, list) else []
    print(f"ℹ️ Found {len(club_player_ids)} club players to process from {CLUB_IDS_FILE.name}.")
    for i, ea_id in enumerate(club_player_ids):
        if (i + 1) % 50 == 0: print(f"  - Player {i+1}/{len(club_player_ids)}")
        player_def_raw = load_json_file(GG_DATA_DIR / f"{ea_id}_ggData.json")
        if player_def_raw and "data" in player_def_raw:
            player = process_player(player_def_raw["data"], False, model_manager, maps)
            if player: processed_players.append(player)
    
    print("\n--- Processing Evo Players ---")
    evolab_data = load_json_file(EVOLAB_FILE)
    if evolab_data and "data" in evolab_data:
        evo_defs = [item["playerItemDefinition"] for item in evolab_data["data"] if "playerItemDefinition" in item]
        print(f"ℹ️ Found {len(evo_defs)} evo player definitions.")
        for i, evo_def in enumerate(evo_defs):
            if (i + 1) % 50 == 0: print(f"  - Evo {i+1}/{len(evo_defs)}")
            player = process_player(evo_def, True, model_manager, maps)
            if player: processed_players.append(player)
    
    print("\n--- Deduplicating Players ---")
    final_players = {}
    for player in processed_players:
        if player['eaId'] not in final_players or player.get('evolution'):
            final_players[player['eaId']] = player
    final_list = list(final_players.values())
    print(f"ℹ️ Total unique players: {len(final_list)}")

    print("\n--- Saving Final Output ---")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_list, f, indent=2, ensure_ascii=False)
    print(f"\n🎉 Success! Processed {len(final_list)} players to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
