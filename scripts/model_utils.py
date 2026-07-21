# model_utils.py
# Shared model-loading, feature-building and ggMetaSub prediction used by both
# club.3.clean.py and build_all_players_summary.py, so the two stay in lock-step
# (previously the model logic lived only in club.3, so the all-players summary had
# no ggMetaSub at all).

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from shared_utils import _normalize_gender, calculate_acceleration_type, get_attribute_with_boost, average_optional


def compute_meta_entry(role_name, sub_accel_type, *, gg_score, gg_chem_style, gg_meta_sub,
                       es_entries, base_attributes, height, gender, es_chem_names, chem_boosts):
    """Assemble one per-role metaRating entry, shared by club.3.clean.py and
    build_all_players_summary.py so they can't drift. The caller resolves the gg
    pieces (which differ for evo vs non-evo) and passes them in:
      - gg_score / gg_chem_style: the best chem-style's ggMeta + style name (or None)
      - gg_meta_sub: predicted ggMetaSub (absolute or evo-anchored) or None
      - es_entries: this role's esMeta rows (chemistry 0..3), already filtered
    Computes gg/es accel types from the chem-style boosts, the on-chem/sub averages,
    and clamps ggMetaSub <= ggMeta (defensive)."""
    def accel(chem_style):
        b = chem_boosts.get((chem_style or "").lower(), {})
        return calculate_acceleration_type(
            get_attribute_with_boost(base_attributes, "attributeAcceleration", b),
            get_attribute_with_boost(base_attributes, "attributeAgility", b),
            get_attribute_with_boost(base_attributes, "attributeStrength", b),
            height, gender,
        )

    entry = {"role": role_name, "subAccelType": sub_accel_type}
    if gg_score is not None:
        entry["ggMeta"] = round(gg_score, 2)
        entry["ggChemStyle"] = gg_chem_style
        entry["ggAccelType"] = accel(gg_chem_style)
    if gg_meta_sub is not None:
        entry["ggMetaSub"] = gg_meta_sub

    if es_entries:
        r0 = next((r for r in es_entries if r.get("chemistry") == 0 and r.get("metaRating") is not None), None)
        if r0:
            entry["esMetaSub"] = round(float(r0["metaRating"]), 2)
        best3 = max([r for r in es_entries if r.get("chemistry") == 3], key=lambda x: x.get("metaRating", -1), default=None)
        if best3 and best3.get("metaRating") is not None:
            entry["esMeta"] = round(float(best3["metaRating"]), 2)
            entry["esChemStyle"] = es_chem_names.get(str(best3.get("chemstyleId")))
            entry["esAccelType"] = accel(entry.get("esChemStyle"))

    gm, gms = entry.get("ggMeta"), entry.get("ggMetaSub")
    if gms is not None and gm is not None and gms > gm:
        entry["ggMetaSub"] = round(gm, 2)
    entry["avgMeta"] = average_optional(entry.get("ggMeta"), entry.get("esMeta"))
    entry["avgMetaSub"] = average_optional(entry.get("ggMetaSub"), entry.get("esMetaSub"))
    return entry


class ModelManager:
    """Loads all *.pkl model bundles from a directory once and serves them by
    (role_name, model_type)."""
    def __init__(self, models_dir: Path):
        self.models = self._load_models(models_dir)

    def _load_models(self, models_dir: Path):
        models = {}
        if not models_dir.exists():
            return models
        for pkl_file in models_dir.glob("*.pkl"):
            try:
                models[pkl_file.stem] = joblib.load(pkl_file)
            except Exception:
                pass
        return models

    def get_model(self, role_name: str, model_type: str):
        safe_role_name = role_name.replace(" ", "_").replace("-", "_")
        return self.models.get(f"{safe_role_name}_{model_type}_model")


def _familiarity_from_lists(role_name, roles_plus, roles_pp):
    if role_name in (roles_pp or []):
        return 2
    if role_name in (roles_plus or []):
        return 1
    return 0


def prepare_features(player_data, maps, boosts={}, role_name=None):
    features = {}
    base_attributes = {k: v for k, v in player_data.items() if k.startswith("attribute")}
    for attr in base_attributes:
        features[attr] = get_attribute_with_boost(base_attributes, attr, boosts)
    for key in ["height", "weight", "skillMoves", "weakFoot", "foot"]:
        features[key] = player_data.get(key)
    features["gender"] = player_data.get("gender") or "Male"

    all_ps = list(maps.get("playstyles", {}).values())
    ps = set(player_data.get("PS", []) or [])
    ps_plus = set(player_data.get("PS+", []) or [])
    for name in all_ps:
        features[name] = 2 if name in ps_plus else (1 if name in ps else 0)

    if role_name is not None:
        features["familiarity"] = _familiarity_from_lists(role_name, player_data.get("roles+", []), player_data.get("roles++", []))
    else:
        features["familiarity"] = 0

    features["bodytype"] = player_data.get("bodyType")
    features["accelerateType"] = calculate_acceleration_type(
        features.get("attributeAcceleration"), features.get("attributeAgility"),
        features.get("attributeStrength"), features.get("height"), features.get("gender")
    )
    return features


def _build_model_input(model_bundle, feature_dict):
    df = pd.DataFrame([feature_dict])
    df = pd.get_dummies(df, columns=["bodytype", "accelerateType", "foot"], dtype=int)
    feats = model_bundle["features"]
    for f in feats:
        if f not in df.columns:
            df[f] = 0
    df = df[feats].fillna(0).infer_objects()
    return df


def _predict_absolute(model_bundle, feature_dict):
    try:
        X = _build_model_input(model_bundle, feature_dict)
        pred_scaled = model_bundle["model"].predict(model_bundle["feature_scaler"].transform(X))
        y = model_bundle["target_scaler"].inverse_transform(np.array(pred_scaled).reshape(-1, 1))[0, 0]
        return float(y)
    except Exception:
        return None


def predict_ggsub_absolute(model_bundle, features_no_chem, *, cap_to_ggmeta=None):
    if not model_bundle:
        return None
    pred = _predict_absolute(model_bundle, features_no_chem)
    if pred is None:
        return None
    if cap_to_ggmeta is not None:
        pred = min(pred, float(cap_to_ggmeta))
    pred = max(0.0, min(99.99, pred))
    return round(float(pred), 2)


def predict_ggsub_evo_anchored(model_bundle, evo_no_chem, *, base_no_chem, cap_to_ggmeta):
    if not model_bundle:
        return None
    evo_pred = _predict_absolute(model_bundle, evo_no_chem)
    base_pred = _predict_absolute(model_bundle, base_no_chem) if base_no_chem else None
    if evo_pred is None and base_pred is None:
        return None
    pred = evo_pred if evo_pred is not None else 0.0
    if base_pred is not None:
        pred = max(pred, base_pred)
    if cap_to_ggmeta is not None:
        pred = min(pred, float(cap_to_ggmeta))
    pred = max(0.0, min(99.99, pred))
    return round(float(pred), 2)


def to_player_like_from_ggdata(gg_data_obj, maps):
    if not gg_data_obj:
        return None
    d = {}
    for k, v in gg_data_obj.items():
        if isinstance(k, str) and k.startswith("attribute"):
            d[k] = v
    d["height"] = gg_data_obj.get("height")
    d["weight"] = gg_data_obj.get("weight")
    d["skillMoves"] = gg_data_obj.get("skillMoves")
    d["weakFoot"] = gg_data_obj.get("weakFoot")
    d["foot"] = maps.get("foot", {}).get(str(gg_data_obj.get("foot")))
    d["bodyType"] = maps.get("bodytypeCode", {}).get(str(gg_data_obj.get("bodytypeCode")))
    d["gender"] = _normalize_gender(gg_data_obj.get("gender"), maps)
    d["PS"] = [maps.get("playstyles", {}).get(str(p)) for p in (gg_data_obj.get("playstyles") or []) if str(p) in maps.get("playstyles", {})]
    d["PS+"] = [maps.get("playstyles", {}).get(str(p)) for p in (gg_data_obj.get("playstylesPlus") or []) if str(p) in maps.get("playstyles", {})]
    d["roles+"] = [maps.get("rolesPlus", {}).get(str(r)) for r in (gg_data_obj.get("rolesPlus") or []) if str(r) in maps.get("rolesPlus", {})]
    d["roles++"] = [maps.get("rolesPlusPlus", {}).get(str(r)) for r in (gg_data_obj.get("rolesPlusPlus") or []) if str(r) in maps.get("rolesPlusPlus", {})]
    return d
