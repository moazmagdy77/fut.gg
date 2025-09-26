# club.3.clean.py

import json
from pathlib import Path
from collections import defaultdict
import copy
import numpy as np
import pandas as pd
import joblib
import warnings
from typing import Optional, Dict, Any, Tuple

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

# --- Helpers ---
def load_json_file(file_path: Path, default_val=None):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default_val

def _normalize_gender(gender_val, maps):
    """
    Convert numeric gender to text via maps["gender_map"].
    Default to 'Male' if unknown (most items are male).
    """
    try:
        g = maps.get("gender_map", {}).get(str(gender_val))
        if isinstance(g, str) and g:
            return g
    except Exception:
        pass
    return "Male"

def calculate_acceleration_type(accel, agility, strength, height, gender: str = "Male"):
    """
    New game rules (gender-specific height thresholds):

    Explosive:
      - Agility >= 65
      - (Agility - Strength) >= 10
      - Acceleration >= 80
      - Height <= 182 (Male) OR <= 162 (Female)

    Lengthy:
      - Strength >= 65
      - (Strength - Agility) >= 4
      - Acceleration >= 40
      - Height >= 183 (Male) OR >= 164 (Female)

    Controlled: otherwise.
    """
    try:
        if accel is None or agility is None or strength is None or height is None:
            return "CONTROLLED"
        accel = int(accel); agility = int(agility); strength = int(strength); height = int(height)
    except Exception:
        return "CONTROLLED"

    gender_str = (gender or "Male")
    is_female = str(gender_str).lower().startswith("f")

    # Height thresholds by gender
    exp_height_ok = (height <= 162) if is_female else (height <= 182)
    len_height_ok = (height >= 164) if is_female else (height >= 183)

    # Explosive
    if (agility >= 65 and (agility - strength) >= 10 and accel >= 80 and exp_height_ok):
        return "EXPLOSIVE"

    # Lengthy
    if (strength >= 65 and (strength - agility) >= 4 and accel >= 40 and len_height_ok):
        return "LENGTHY"

    # Otherwise
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

def parse_gg_rating_str(gg_rating_str_raw: Optional[str]) -> Dict[str, list]:
    parsed_ratings_by_role = defaultdict(list)
    if not gg_rating_str_raw:
        return parsed_ratings_by_role
    for part in gg_rating_str_raw.split('||'):
        try:
            chem_id_str, role_id_str, score_str = part.split(':')
            parsed_ratings_by_role[role_id_str].append({"chem_id_str": chem_id_str, "score": float(score_str)})
        except (ValueError, IndexError):
            continue
    return parsed_ratings_by_role

def resolve_anchor_source_eaid(evo_def: Dict[str, Any]) -> Optional[str]:
    for k in ("baseEaId", "originalEaId", "rootEaId", "rootDefinitionEaId", "baseItemEaId", "baseCardEaId"):
        v = evo_def.get(k)
        if v is not None:
            try:
                return str(int(v))
            except Exception:
                continue
    eid = evo_def.get("eaId")
    return str(int(eid)) if eid is not None else None

def to_player_like_from_ggdata(gg_data_obj: Optional[Dict[str, Any]], maps) -> Optional[Dict[str, Any]]:
    """Convert fut.gg 'data' object into the dict shape expected by prepare_features()."""
    if not gg_data_obj:
        return None
    d: Dict[str, Any] = {}
    # attributes
    for k, v in gg_data_obj.items():
        if isinstance(k, str) and k.startswith("attribute"):
            d[k] = v
    # simple fields
    d["height"] = gg_data_obj.get("height")
    d["weight"] = gg_data_obj.get("weight")
    d["skillMoves"] = gg_data_obj.get("skillMoves")
    d["weakFoot"] = gg_data_obj.get("weakFoot")
    # categorical
    d["foot"] = maps["foot_map"].get(str(gg_data_obj.get("foot")))
    d["bodyType"] = maps["bodytype_code_map"].get(str(gg_data_obj.get("bodytypeCode")))
    # gender
    d["gender"] = _normalize_gender(gg_data_obj.get("gender"), maps)
    # playstyles
    d["PS"]  = [maps["playstyles_map"].get(str(p)) for p in (gg_data_obj.get("playstyles") or []) if str(p) in maps["playstyles_map"]]
    d["PS+"] = [maps["playstyles_map"].get(str(p)) for p in (gg_data_obj.get("playstylesPlus") or []) if str(p) in maps["playstyles_map"]]
    # roles lists (names)
    d["roles+"]  = [maps["roles_plus_map"].get(str(r)) for r in (gg_data_obj.get("rolesPlus") or []) if str(r) in maps["roles_plus_map"]]
    d["roles++"] = [maps["roles_plus_plus_map"].get(str(r)) for r in (gg_data_obj.get("rolesPlusPlus") or []) if str(r) in maps["roles_plus_plus_map"]]
    return d

def get_es_anchors_for_role(es_meta_raw, role_name: str, maps) -> Tuple[Optional[float], Dict[str, float]]:
    """
    From EasySBC API blob, extract:
      - sub_anchor (chem=0) for the target role
      - a dict of chem=3 anchors per chemstyle name (lowercased)
    Always filter by playerRoleId to avoid cross-role contamination.
    """
    es_role_id = maps["roleNameToEsRoleId"].get(role_name)
    if not (es_meta_raw and es_role_id):
        return None, {}
    filtered = []
    for b in es_meta_raw:
        ratings = (b.get("data", {}) or {}).get("metaRatings", []) or []
        filtered.extend([r for r in ratings if str(r.get("playerRoleId")) == str(es_role_id)])
    if not filtered:
        return None, {}

    # chem=0 anchor
    r0 = next((r for r in filtered if r.get("chemistry") == 0 and r.get("metaRating") is not None), None)
    sub_anchor = float(r0["metaRating"]) if r0 else None

    # chem=3 per-style anchors
    es_style_map = maps["es_chem_style_names_map"]
    anchor_3_by_style = {}
    for r in filtered:
        if r.get("chemistry") == 3 and r.get("metaRating") is not None:
            chem_id = str(r.get("chemstyleId"))
            name = es_style_map.get(chem_id)
            if name:
                anchor_3_by_style[name.lower()] = float(r["metaRating"])
    return sub_anchor, anchor_3_by_style

# --- Model Manager ---
class ModelManager:
    def __init__(self, models_dir: Path):
        self.models = self._load_models(models_dir)

    def _load_models(self, models_dir: Path):
        models = {}
        if not models_dir.exists():
            print(f"⚠️ Models directory not found: {models_dir}")
            return models

        for pkl_file in models_dir.glob("*.pkl"):
            try:
                bundle = joblib.load(pkl_file)
                models[pkl_file.stem] = bundle
            except Exception as e:
                print(f"⚠️ Error loading model {pkl_file.name}: {e}")
        print(f"✅ Loaded {len(models)} models from disk.")
        return models

    def get_model(self, role_name: str, model_type: str):
        safe_role_name = role_name.replace(" ", "_").replace("-", "_")
        model_key = f"{safe_role_name}_{model_type}_model"
        return self.models.get(model_key)

# --- Feature building for models (ES + ggSub use same family) ---
def _familiarity_from_lists(role_name: str, roles_plus: list, roles_pp: list) -> int:
    if role_name in (roles_pp or []): return 2
    if role_name in (roles_plus or []): return 1
    return 0

def prepare_features(player_data: Dict[str, Any], maps, boosts: Dict[str, int] = {}, role_name: Optional[str] = None):
    """Build feature dict for a specific role (includes 'familiarity')."""
    features = {}
    base_attributes = {k: v for k, v in player_data.items() if k.startswith("attribute")}
    for attr in base_attributes:
        features[attr] = get_attribute_with_boost(base_attributes, attr, boosts)

    for key in ["height", "weight", "skillMoves", "weakFoot", "foot"]:
        features[key] = player_data.get(key)

    # ensure gender present for accelerateType calc
    features["gender"] = player_data.get("gender") or "Male"

    # playstyles as 0/1/2
    all_ps = list(maps.get("playstyles", {}).values())
    ps = set(player_data.get("PS", []) or []); ps_plus = set(player_data.get("PS+", []) or [])
    for name in all_ps:
        features[name] = 2 if name in ps_plus else (1 if name in ps else 0)

    # familiarity (0/1/2) depends on role
    if role_name is not None:
        roles_plus = player_data.get("roles+", []) or []
        roles_pp   = player_data.get("roles++", []) or []
        features["familiarity"] = _familiarity_from_lists(role_name, roles_plus, roles_pp)
    else:
        features["familiarity"] = 0

    # categorical
    features["bodytype"] = player_data.get("bodyType")
    features["accelerateType"] = calculate_acceleration_type(
        features.get("attributeAcceleration"),
        features.get("attributeAgility"),
        features.get("attributeStrength"),
        features.get("height"),
        features.get("gender")
    )
    return features

def _build_model_input(model_bundle: Dict[str, Any], feature_dict: Dict[str, Any]) -> pd.DataFrame:
    df = pd.DataFrame([feature_dict])
    df = pd.get_dummies(df, columns=["bodytype", "accelerateType", "foot"], dtype=int)
    feats = model_bundle["features"]
    for f in feats:
        if f not in df.columns:
            df[f] = 0
    df = df[feats].fillna(0).infer_objects(copy=False)
    return df

def _predict_absolute(model_bundle: Dict[str, Any], feature_dict: Dict[str, Any]) -> Optional[float]:
    try:
        X = _build_model_input(model_bundle, feature_dict)
        pred_scaled = model_bundle["model"].predict(
            model_bundle["feature_scaler"].transform(X)
        )
        y = model_bundle["target_scaler"].inverse_transform(np.array(pred_scaled).reshape(-1,1))[0,0]
        return float(y)
    except Exception:
        return None

# --- Anchored ES prediction for EVOS (unchanged behaviour) ---
def predict_es_with_anchor(
    model_bundle: Dict[str, Any],
    evo_features: Dict[str, Any],
    *,
    base_features: Optional[Dict[str, Any]],
    api_anchor: Optional[float],
    hard_min: Optional[float] = None,
    hard_max: float = 99.99
) -> Optional[float]:
    if not model_bundle:
        return None

    pred_abs_evo = _predict_absolute(model_bundle, evo_features)
    pred_abs_base = _predict_absolute(model_bundle, base_features) if base_features else None

    floors = []
    if api_anchor is not None:
        floors.append(float(api_anchor))
    if pred_abs_base is not None:
        floors.append(float(pred_abs_base))
    if hard_min is not None:
        floors.append(float(hard_min))
    floor_val = max(floors) if floors else 0.0

    anchored_via_delta = None
    try:
        if base_features is not None and "coef_unscaled" in model_bundle:
            Xe = _build_model_input(model_bundle, evo_features).values
            Xb = _build_model_input(model_bundle, base_features).values
            dX = np.maximum(0.0, Xe - Xb)  # monotone: evo never decreases
            w = np.array(model_bundle["coef_unscaled"], dtype=float).reshape(1, -1)
            if w.shape[1] == dX.shape[1]:
                delta_rating = float((dX * w).sum())
                anchored_via_delta = floor_val + max(0.0, delta_rating)
    except Exception:
        anchored_via_delta = None

    candidates = []
    if pred_abs_evo is not None:
        candidates.append(pred_abs_evo)
    if anchored_via_delta is not None:
        candidates.append(anchored_via_delta)
    if not candidates:
        return None

    pred = max(candidates)
    pred = max(pred, floor_val)
    pred = min(pred, hard_max)
    pred = max(0.0, pred)
    return round(float(pred), 2)

# --- ggMetaSub prediction for EVOS (anchored, never exceeds same-role ggMeta) ---
def predict_ggsub_with_anchor(
    model_bundle: Dict[str, Any],
    evo_features: Dict[str, Any],
    *,
    base_features: Optional[Dict[str, Any]],
    basic_anchor: Optional[float],
    same_role_ggmeta: Optional[float]
) -> Optional[float]:
    if not model_bundle:
        return None

    pred_abs_evo = _predict_absolute(model_bundle, evo_features)

    floors = []
    if basic_anchor is not None:
        floors.append(float(basic_anchor))
    if base_features is not None:
        try:
            base_abs = _predict_absolute(model_bundle, base_features)
            if base_abs is not None:
                floors.append(float(base_abs))
        except Exception:
            pass
    floor_val = max(floors) if floors else 0.0

    candidates = []
    if pred_abs_evo is not None:
        candidates.append(pred_abs_evo)
    if not candidates:
        return None

    pred = max(candidates)
    pred = max(pred, floor_val)
    if same_role_ggmeta is not None:
        pred = min(pred, float(same_role_ggmeta))  # enforce ggMetaSub ≤ ggMeta
    pred = min(pred, 99.99)
    pred = max(0.0, pred)
    return round(float(pred), 2)

def process_player(player_def: Dict[str, Any], is_evo: bool, model_manager: "ModelManager", maps):
    player_output = {"eaId": player_def.get("eaId"), "evolution": is_evo}
    base_attributes = {k: v for k, v in player_def.items() if k.startswith("attribute")}
    for key in ["commonName", "overall", "height", "weight", "skillMoves", "weakFoot"]:
        player_output[key] = player_def.get(key)
    player_output.update(base_attributes)

    # gender
    player_output["gender"] = _normalize_gender(player_def.get("gender"), maps)

    # positions and categorical labels
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
        player_output.get("height"),
        player_output.get("gender")
    )

    es_meta_raw = load_json_file(ES_META_DIR / f"{player_output['eaId']}_esMeta.json", [])

    # gg scores source
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
        if not role_name:
            continue

        meta_entry = {"role": role_name, "subAccelType": sub_accel_type}

        # gg best on-chem
        best_gg = max(scores, key=lambda x: x.get("score", 0), default=None)
        if best_gg:
            meta_entry["ggMeta"] = round(best_gg.get("score", 0.0), 2)
            chem_id = best_gg.get("chem_id_str") if is_evo else str(best_gg.get("chemistryStyle"))
            meta_entry["ggChemStyle"] = maps["gg_chem_style_names_map"].get(chem_id)
            boosts = maps["chem_style_boosts_map"].get((meta_entry.get("ggChemStyle") or "").lower(), {})
            meta_entry["ggAccelType"] = calculate_acceleration_type(
                get_attribute_with_boost(base_attributes, "attributeAcceleration", boosts),
                get_attribute_with_boost(base_attributes, "attributeAgility", boosts),
                get_attribute_with_boost(base_attributes, "attributeStrength", boosts),
                player_output.get("height"),
                player_output.get("gender")
            )

        # gg basic as sub (standard players use API "Basic"; evos predicted & anchored)
        if not is_evo:
            basic_gg = next(
                (s for s in scores if (maps["gg_chem_style_names_map"].get(str(s.get("chemistryStyle")), "") or "").lower() == 'basic'),
                None
            )
            if basic_gg and basic_gg.get("score") is not None:
                meta_entry["ggMetaSub"] = round(float(basic_gg["score"]), 2)
            elif "GK" in role_name and meta_entry.get("ggMeta") is not None:
                meta_entry["ggMetaSub"] = round(meta_entry["ggMeta"] * 0.95, 2)
        else:
            # Evo: predict ggMetaSub and anchor to base Basic + do not exceed same-role ggMeta
            gg_sub_model = model_manager.get_model(role_name, 'ggMetaSub')
            if gg_sub_model:
                base_anchor_eaid = resolve_anchor_source_eaid(player_def)  # may equal evo id
                base_gg_raw = load_json_file(GG_DATA_DIR / f"{base_anchor_eaid}_ggData.json")
                base_player_like = to_player_like_from_ggdata(
                    base_gg_raw.get("data") if base_gg_raw and "data" in base_gg_raw else None, maps
                )

                # Retrieve base Basic score for same role (anchor)
                basic_anchor = None
                base_gg_meta = load_json_file(GG_META_DIR / f"{base_anchor_eaid}_ggMeta.json")
                if base_gg_meta and "data" in base_gg_meta and "scores" in base_gg_meta["data"]:
                    basic_scores_by_role = defaultdict(list)
                    for s in base_gg_meta["data"]["scores"]:
                        basic_scores_by_role[str(s.get("role"))].append(s)
                    base_role_scores = basic_scores_by_role.get(role_id_str) or []
                    base_basic = next(
                        (s for s in base_role_scores if (maps["gg_chem_style_names_map"].get(str(s.get("chemistryStyle")), "") or "").lower() == 'basic'),
                        None
                    )
                    if base_basic and base_basic.get("score") is not None:
                        basic_anchor = float(base_basic["score"])

                evo_features_sub  = prepare_features(player_output, maps, boosts={}, role_name=role_name)
                base_features_sub = prepare_features(base_player_like, maps, boosts={}, role_name=role_name) if base_player_like else None
                meta_entry["ggMetaSub"] = predict_ggsub_with_anchor(
                    gg_sub_model,
                    evo_features_sub,
                    base_features=base_features_sub,
                    basic_anchor=basic_anchor,
                    same_role_ggmeta=meta_entry.get("ggMeta")
                )

        # ---------- ES handling ----------
        if is_evo:
            # Anchor info from base item
            base_anchor_eaid = resolve_anchor_source_eaid(player_def)  # may equal evo id
            base_es_meta_raw = load_json_file(ES_META_DIR / f"{base_anchor_eaid}_esMeta.json", [])
            sub_anchor, anchor3_by_style = get_es_anchors_for_role(base_es_meta_raw, role_name, maps)

            base_gg_raw = load_json_file(GG_DATA_DIR / f"{base_anchor_eaid}_ggData.json")
            base_player_like = to_player_like_from_ggdata(
                base_gg_raw.get("data") if base_gg_raw and "data" in base_gg_raw else None, maps
            )

            # esMetaSub (chem=0)
            es_sub_model = model_manager.get_model(role_name, 'esMetaSub')
            if es_sub_model:
                evo_features_sub  = prepare_features(player_output, maps, boosts={}, role_name=role_name)
                base_features_sub = prepare_features(base_player_like, maps, boosts={}, role_name=role_name) if base_player_like else None
                meta_entry["esMetaSub"] = predict_es_with_anchor(
                    es_sub_model,
                    evo_features_sub,
                    base_features=base_features_sub,
                    api_anchor=sub_anchor,
                    hard_min=sub_anchor,
                    hard_max=99.99
                )

            # esMeta (chem=3) — choose best chemstyle
            es_model = model_manager.get_model(role_name, 'esMeta')
            if es_model:
                best_val, best_chem, best_accel = None, None, sub_accel_type
                for chem_name, boosts in maps["chem_style_boosts_map"].items():
                    evo_features_chem  = prepare_features(player_output, maps, boosts=boosts, role_name=role_name)
                    base_features_chem = prepare_features(base_player_like, maps, boosts=boosts, role_name=role_name) if base_player_like else None
                    chem_anchor = anchor3_by_style.get(chem_name.lower()) if anchor3_by_style else None
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
        else:
            # Standard player: take API esMeta/esMetaSub (STRICTLY FILTERED BY playerRoleId)
            es_role_id = maps["roleNameToEsRoleId"].get(role_name)
            if es_role_id and es_meta_raw:
                # Flatten and filter to THIS role only
                filtered = []
                for b in es_meta_raw:
                    rs = (b.get("data", {}) or {}).get("metaRatings", []) or []
                    filtered.extend([r for r in rs if str(r.get("playerRoleId")) == str(es_role_id)])
                if filtered:
                    rating_0 = next((r for r in filtered if r.get("chemistry") == 0 and r.get("metaRating") is not None), None)
                    if rating_0:
                        meta_entry["esMetaSub"] = round(float(rating_0["metaRating"]), 2)
                    best_3 = max(
                        [r for r in filtered if r.get("chemistry") == 3 and r.get("isBestChemstyleAtChem")],
                        key=lambda x: x.get("metaRating", -1), default=None
                    )
                    if best_3 and best_3.get("metaRating") is not None:
                        meta_entry["esMeta"] = round(float(best_3["metaRating"]), 2)
                        chem_id_3 = str(best_3.get("chemstyleId"))
                        meta_entry["esChemStyle"] = maps["es_chem_style_names_map"].get(chem_id_3)
                        boosts_3 = maps["chem_style_boosts_map"].get((meta_entry.get("esChemStyle") or "").lower(), {})
                        meta_entry["esAccelType"] = calculate_acceleration_type(
                            get_attribute_with_boost(base_attributes, "attributeAcceleration", boosts_3),
                            get_attribute_with_boost(base_attributes, "attributeAgility", boosts_3),
                            get_attribute_with_boost(base_attributes, "attributeStrength", boosts_3),
                            player_output.get("height"),
                            player_output.get("gender")
                        )

        # Final ggMetaSub <= ggMeta safety for all players
        if meta_entry.get("ggMetaSub") is not None and meta_entry.get("ggMeta") is not None:
            if meta_entry["ggMetaSub"] > meta_entry["ggMeta"]:
                meta_entry["ggMetaSub"] = round(float(meta_entry["ggMeta"]), 2)

        # averages
        gg_meta, es_meta = meta_entry.get("ggMeta"), meta_entry.get("esMeta")
        gg_meta_sub, es_meta_sub = meta_entry.get("ggMetaSub"), meta_entry.get("esMetaSub")
        meta_entry["avgMeta"] = round((gg_meta + es_meta) / 2, 2) if (gg_meta is not None and es_meta is not None) else gg_meta or es_meta
        meta_entry["avgMetaSub"] = round((gg_meta_sub + es_meta_sub) / 2, 2) if (gg_meta_sub is not None and es_meta_sub is not None) else gg_meta_sub or es_meta_sub

        player_output["metaRatings"].append(meta_entry)

    return player_output

def main():
    print("🚀 Starting Player Data Processing Script...")
    maps_data = load_json_file(MAPS_FILE)
    if not maps_data:
        print("❌ Critical error: Could not load maps.json. Exiting.")
        return

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
        "role_id_to_name_map": maps_data.get("role", {}),
        "roleNameToEsRoleId": maps_data.get("roleNameToEsRoleId", {}),
        "playstyles": maps_data.get("playstyles", {}),
        "gender_map": maps_data.get("gender", {})  # NEW: numeric -> text
    }

    model_manager = ModelManager(MODELS_DIR)
    processed_players = []

    print("\n--- Processing Club Players ---")
    club_ids_data = load_json_file(CLUB_IDS_FILE)
    club_player_ids = [str(id) for id in club_ids_data] if isinstance(club_ids_data, list) else []
    print(f"ℹ️ Found {len(club_player_ids)} club players to process from {CLUB_IDS_FILE.name}.")
    for i, ea_id in enumerate(club_player_ids):
        if (i + 1) % 50 == 0:
            print(f"  - Player {i+1}/{len(club_player_ids)}")
        player_def_raw = load_json_file(GG_DATA_DIR / f"{ea_id}_ggData.json")
        if player_def_raw and "data" in player_def_raw:
            player = process_player(player_def_raw["data"], False, model_manager, maps)
            if player:
                processed_players.append(player)

    print("\n--- Processing Evo Players ---")
    evolab_data = load_json_file(EVOLAB_FILE)
    if evolab_data and "data" in evolab_data:
        evo_defs = [item["playerItemDefinition"] for item in evolab_data["data"] if "playerItemDefinition" in item]
        print(f"ℹ️ Found {len(evo_defs)} evo player definitions.")
        for i, evo_def in enumerate(evo_defs):
            if (i + 1) % 50 == 0:
                print(f"  - Evo {i+1}/{len(evo_defs)}")
            player = process_player(evo_def, True, model_manager, maps)
            if player:
                processed_players.append(player)

    print("\n--- Deduplicating Players ---")
    final_players = {}
    for player in processed_players:
        # keep evo if duplicates
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
