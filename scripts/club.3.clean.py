import json
from pathlib import Path
import os
from collections import defaultdict
import copy
import numpy as np
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


def resolve_anchor_source_eaid(evo_def):
    """
    Try common keys to map an evolved item back to its base EA ID.
    Fallback: use the evolved id if no mapping is present.
    """
    for k in ("baseEaId", "originalEaId", "rootEaId", "rootDefinitionEaId", "baseItemEaId", "baseCardEaId"):
        v = evo_def.get(k)
        if v is not None:
            try:
                return str(int(v))
            except Exception:
                continue
    # fallback: use the item's own id
    eid = evo_def.get("eaId")
    return str(int(eid)) if eid is not None else None

def to_player_like_from_ggdata(gg_data_obj, maps):
    """
    Convert fut.gg 'data' object into the dict shape expected by prepare_features():
    attributes + height/weight/skills/foot + PS/PS+ + bodyType (string).
    """
    if not gg_data_obj:
        return None
    d = {}
    # copy over attributes (same keys as your pipeline uses)
    for k, v in gg_data_obj.items():
        if isinstance(k, str) and k.startswith("attribute"):
            d[k] = v
    d["height"] = gg_data_obj.get("height")
    d["weight"] = gg_data_obj.get("weight")
    d["skillMoves"] = gg_data_obj.get("skillMoves")
    d["weakFoot"] = gg_data_obj.get("weakFoot")
    # string foot/bodytype/playstyles
    foot_code = gg_data_obj.get("foot")
    d["foot"] = maps["foot_map"].get(str(foot_code))
    d["bodyType"] = maps["bodytype_code_map"].get(str(gg_data_obj.get("bodytypeCode")))
    d["PS"]  = [maps["playstyles_map"].get(str(p)) for p in gg_data_obj.get("playstyles", []) if str(p) in maps["playstyles_map"]]
    d["PS+"] = [maps["playstyles_map"].get(str(p)) for p in gg_data_obj.get("playstylesPlus", []) if str(p) in maps["playstyles_map"]]
    return d

def get_es_anchors_for_role(es_meta_raw, role_name, maps):
    """
    From EasySBC API blob (list of {roleId, data.metaRatings}), extract:
      - sub_anchor (chem=0) for the role
      - a dict of chem=3 anchors per chemstyle name (lowercased)
    """
    es_role_id = maps["roleNameToEsRoleId"].get(role_name)
    if not (es_meta_raw and es_role_id):
        return None, {}
    # find the block for this role id
    role_block = next((b for b in es_meta_raw if any(str(r.get("playerRoleId")) == str(es_role_id)
                                                     for r in b.get("data", {}).get("metaRatings", []))), None)
    if not role_block:
        return None, {}
    ratings = [r for r in role_block["data"]["metaRatings"] if str(r.get("playerRoleId")) == str(es_role_id)]
    # chem=0 anchor
    r0 = next((r for r in ratings if r.get("chemistry") == 0), None)
    sub_anchor = float(r0["metaRating"]) if r0 else None
    # chem=3 per-style anchors
    es_style_map = maps["es_chem_style_names_map"]
    anchor_3_by_style = {}
    for r in ratings:
        if r.get("chemistry") == 3:
            chem_id = str(r.get("chemstyleId"))
            name = es_style_map.get(chem_id)
            if name:
                anchor_3_by_style[name.lower()] = float(r.get("metaRating", 0.0))
    return sub_anchor, anchor_3_by_style


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
                bundle = joblib.load(pkl_file)  # <-- define it first
                # Warn (don’t fail) if this is an older bundle without coef_unscaled
                if isinstance(bundle, dict) and "coef_unscaled" not in bundle:
                    print(f"⚠️ {pkl_file.name} missing 'coef_unscaled' (retrain with updated trainer).")
                models[pkl_file.stem] = bundle
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

def predict_rating(model_bundle, feature_dict, player_name, model_name):
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
        prediction = model_bundle["target_scaler"].inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
        return round(float(prediction), 2)
    except Exception as e:
        # print(f"⚠️ Prediction failed for {player_name} with model '{model_name}'. Error: {e}")
        return None

# --- Anchored Prediction Utilities (paste below predict_rating) ---

def _build_model_input(model_bundle, feature_dict):
    """One-hot align a single feature dict to the model's feature vector."""
    df = pd.DataFrame([feature_dict])
    df = pd.get_dummies(df, columns=["bodytype", "accelerateType", "foot"], dtype=int)
    # ensure all expected columns exist, in order
    feats = model_bundle["features"]
    for f in feats:
        if f not in df.columns:
            df[f] = 0
    df = df[feats].fillna(0).infer_objects(copy=False)
    return df.values.astype(float).reshape(1, -1)

def _predict_absolute(model_bundle, feature_dict):
    """Use your existing scaler+model+inverse scaler pipeline to get an absolute prediction."""
    try:
        X = _build_model_input(model_bundle, feature_dict)
        pred_scaled = model_bundle["model"].predict(model_bundle["feature_scaler"].transform(pd.DataFrame(X, columns=model_bundle["features"])))
        y = model_bundle["target_scaler"].inverse_transform(np.array(pred_scaled).reshape(-1,1))[0,0]
        return float(y)
    except Exception:
        # graceful fallback
        return None

def predict_es_with_anchor(
    model_bundle,
    evo_features: dict,
    *,
    base_features: dict | None,
    api_anchor: float | None,
    hard_min: float | None = None,
    hard_max: float = 99.99
) -> float | None:
    """
    Anchored inference for esMeta/esMetaSub.
    - Adds non-negative feature delta contribution to the base anchor.
    - Enforces floor=max(anchor, hard_min) and ceiling=hard_max.
    """
    if not model_bundle:
        return None

    # 1) Absolute model predictions for evo and (optionally) base
    pred_abs_evo = _predict_absolute(model_bundle, evo_features)
    pred_abs_base = _predict_absolute(model_bundle, base_features) if base_features else None

    # 2) Floor to satisfy "evolved >= base"
    floors = []
    if api_anchor is not None:
        floors.append(float(api_anchor))
    if pred_abs_base is not None:
        floors.append(float(pred_abs_base))
    if hard_min is not None:
        floors.append(float(hard_min))
    floor_val = max(floors) if floors else None

    # 3) Delta contribution in *unscaled* units when we have base features and weights
    anchored_via_delta = None
    try:
        if base_features is not None and "coef_unscaled" in model_bundle:
            Xe = _build_model_input(model_bundle, evo_features)
            Xb = _build_model_input(model_bundle, base_features)
            dX = np.maximum(0.0, Xe - Xb)  # evolution never decreases => non-negative deltas only
            w = np.array(model_bundle["coef_unscaled"], dtype=float).reshape(1, -1)
            if w.shape[1] == dX.shape[1]:
                delta_rating = float((dX * w).sum())
                start = floor_val if floor_val is not None else 0.0
                anchored_via_delta = start + max(0.0, delta_rating)
    except Exception:
        anchored_via_delta = None  # fallback if anything goes wrong

    # 4) Choose the best available estimate, then clamp
    candidates = []
    if pred_abs_evo is not None:
        candidates.append(pred_abs_evo)
    if anchored_via_delta is not None:
        candidates.append(anchored_via_delta)
    if not candidates:
        return None
    pred = max(candidates)  # never punish evo relative to absolute model

    # apply floor and cap
    if floor_val is not None:
        pred = max(pred, floor_val)
    pred = min(pred, hard_max)
    return round(float(pred), 2)


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
    
    # Determine the source of ggMeta data
    if is_evo:
        gg_scores_by_role = parse_gg_rating_str(player_def.get("ggRatingStr"))
    else:
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
            chem_id = best_gg_score.get("chem_id_str") if is_evo else str(best_gg_score.get("chemistryStyle"))
            meta_entry["ggChemStyle"] = maps["gg_chem_style_names_map"].get(chem_id)
            boosts = maps["chem_style_boosts_map"].get(meta_entry.get("ggChemStyle", "").lower(), {})
            meta_entry["ggAccelType"] = calculate_acceleration_type(get_attribute_with_boost(base_attributes, "attributeAcceleration", boosts), get_attribute_with_boost(base_attributes, "attributeAgility", boosts), get_attribute_with_boost(base_attributes, "attributeStrength", boosts), player_output.get("height"))

        basic_gg_score = next((s for s in scores if maps["gg_chem_style_names_map"].get(s.get("chem_id_str") if is_evo else str(s.get("chemistryStyle")), "").lower() == 'basic'), None)
        if basic_gg_score:
            meta_entry["ggMetaSub"] = round(basic_gg_score.get("score"), 2)
        elif "GK" in role_name and meta_entry.get("ggMeta") is not None:
             meta_entry["ggMetaSub"] = round(meta_entry["ggMeta"] * 0.95, 2)

        if is_evo:
            # --- Load base references for anchoring ---
            # 1) which EA id to anchor against?
            base_anchor_eaid = resolve_anchor_source_eaid(player_def)  # may be the same as evo id if mapping missing
            # 2) EasySBC API anchors (for the base card)
            base_es_meta_raw = load_json_file(ES_META_DIR / f"{base_anchor_eaid}_esMeta.json", [])
            sub_anchor, anchor3_by_style = get_es_anchors_for_role(base_es_meta_raw, role_name, maps)
            # 3) Base features from fut.gg (for delta on model feature space)
            base_gg_raw = load_json_file(GG_DATA_DIR / f"{base_anchor_eaid}_ggData.json")
            base_player_like = to_player_like_from_ggdata(base_gg_raw.get("data") if base_gg_raw and "data" in base_gg_raw else None, maps)

            # --- esMetaSub (chem = 0) ---
            es_sub_model = model_manager.get_model(role_name, 'esMetaSub')
            if es_sub_model:
                evo_features_sub = prepare_features(player_output, maps, boosts={})
                base_features_sub = prepare_features(base_player_like, maps, boosts={}) if base_player_like else None
                meta_entry["esMetaSub"] = predict_es_with_anchor(
                    es_sub_model,
                    evo_features_sub,
                    base_features=base_features_sub,
                    api_anchor=sub_anchor,          # lower floor from API if available
                    hard_min=sub_anchor,            # enforce never below the base (API) value
                    hard_max=99.99
                )

            # --- esMeta (chem = 3) best chemstyle ---
            es_model = model_manager.get_model(role_name, 'esMeta')
            if es_model:
                best_val, best_chem, best_accel = None, None, sub_accel_type
                for chem_name, boosts in maps["chem_style_boosts_map"].items():
                    evo_features_chem = prepare_features(player_output, maps, boosts=boosts)
                    base_features_chem = prepare_features(base_player_like, maps, boosts=boosts) if base_player_like else None
                    chem_anchor = anchor3_by_style.get(chem_name.lower())  # per-style anchor from API
                    val = predict_es_with_anchor(
                        es_model,
                        evo_features_chem,
                        base_features=base_features_chem,
                        api_anchor=chem_anchor,
                        hard_min=chem_anchor,
                        hard_max=99.99
                    )
                    if val is not None and (best_val is None or val > best_val):
                        best_val = val
                        best_chem = chem_name.title()
                        best_accel = evo_features_chem.get("accelerateType")
                meta_entry["esMeta"] = best_val
                meta_entry["esChemStyle"] = best_chem or "basic"
                meta_entry["esAccelType"] = best_accel
        else: # Standard player
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
