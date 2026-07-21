# build_all_players_summary.py
# Builds a static summary JSON of ALL players from the raw data directory, including a
# modeled ggMetaSub per role (via model_utils + the step-7 models). Parallelized across
# CPU cores (mirrors club.3.clean.py) because the per-player model prediction is now the
# dominant cost. Run via retrain.py (AFTER step 7 so the models are fresh) or standalone.

import json
import sys
import io
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
except Exception:
    pass
import os
import time
import concurrent.futures
import multiprocessing
from pathlib import Path
from collections import defaultdict
from shared_utils import load_json_file, _normalize_gender, calculate_acceleration_type, get_attribute_with_boost, average_optional
from model_utils import ModelManager, prepare_features, predict_ggsub_absolute

# --- Configuration ---
BASE_DATA_DIR = Path(__file__).resolve().parent / '../data'
MODELS_DIR = Path(__file__).resolve().parent / '../models'
RAW_DATA_DIR = BASE_DATA_DIR / 'raw'
MAPS_FILE = BASE_DATA_DIR / 'maps.json'
OUTPUT_FILE = BASE_DATA_DIR / 'all_players_summary.json'

MIN_HEIGHT_CM = 195  # default; overridable via --min-height (matches club.3.clean.py)
for _i, _arg in enumerate(sys.argv):
    if _arg == '--min-height' and _i + 1 < len(sys.argv):
        try:
            MIN_HEIGHT_CM = int(sys.argv[_i + 1])
        except ValueError:
            pass
        break

# --- Worker globals (initialised once per process) ---
GLOBAL = {}


def init_worker(min_height_cm):
    maps = load_json_file(MAPS_FILE) or {}
    GLOBAL['min_height'] = min_height_cm
    GLOBAL['maps'] = maps
    GLOBAL['foot'] = maps.get("foot", {})
    GLOBAL['bodytype'] = maps.get("bodytypeCode", {})
    GLOBAL['position'] = maps.get("position", {})
    GLOBAL['playstyles'] = maps.get("playstyles", {})
    GLOBAL['rolesPlus'] = maps.get("rolesPlus", {})
    GLOBAL['rolesPP'] = maps.get("rolesPlusPlus", {})
    GLOBAL['role'] = maps.get("role", {})
    GLOBAL['ggChem'] = maps.get("ggChemistryStyleNames", {})
    GLOBAL['esChem'] = maps.get("esChemistryStyleNames", {})
    GLOBAL['roleToEs'] = maps.get("roleNameToEsRoleId", {})
    GLOBAL['chemBoosts'] = {
        item['name'].lower(): item['threeChemistryModifiers']
        for item in maps.get("ChemistryStylesBoosts", [])
        if 'name' in item and 'threeChemistryModifiers' in item
    }
    GLOBAL['models'] = ModelManager(MODELS_DIR)


def process_one(file_info):
    gg_path_str, sub = file_info
    maps = GLOBAL['maps']
    try:
        gg_raw = json.loads(Path(gg_path_str).read_text(encoding='utf-8'))
    except Exception:
        return None

    p = gg_raw.get("data")
    if not p:
        return None

    ea_id = str(p.get("eaId", ""))
    base_attributes = {k: v for k, v in p.items() if isinstance(k, str) and k.startswith("attribute")}
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

    numeric_positions = [str(pos) for pos in [p.get("position")] + (p.get("alternativePositionIds") or []) if pos is not None]
    player["positions"] = list(set(GLOBAL['position'].get(pos) for pos in numeric_positions if pos in GLOBAL['position']))
    player["foot"] = GLOBAL['foot'].get(str(p.get("foot")))
    player["bodyType"] = GLOBAL['bodytype'].get(str(p.get("bodytypeCode")))
    player["PS"] = [GLOBAL['playstyles'].get(str(ps)) for ps in (p.get("playstyles") or []) if str(ps) in GLOBAL['playstyles']]
    player["PS+"] = [GLOBAL['playstyles'].get(str(ps)) for ps in (p.get("playstylesPlus") or []) if str(ps) in GLOBAL['playstyles']]
    player["roles+"] = [GLOBAL['rolesPlus'].get(str(r)) for r in (p.get("rolesPlus") or []) if str(r) in GLOBAL['rolesPlus']]
    player["roles++"] = [GLOBAL['rolesPP'].get(str(r)) for r in (p.get("rolesPlusPlus") or []) if str(r) in GLOBAL['rolesPP']]
    rarity_data = p.get("rarity")
    player["rarity"] = rarity_data.get("name") if isinstance(rarity_data, dict) else None

    try:
        player_height = int(player.get("height") or 0)
    except (ValueError, TypeError):
        player_height = 0
    player["isTall"] = player_height >= GLOBAL['min_height']

    sub_accel_type = calculate_acceleration_type(
        base_attributes.get("attributeAcceleration"), base_attributes.get("attributeAgility"),
        base_attributes.get("attributeStrength"), player.get("height"), gender_text
    )

    # ggMeta scores by role (from the same subfolder as ggData)
    gg_scores_by_role = defaultdict(list)
    gg_meta_raw = load_json_file(RAW_DATA_DIR / sub / 'ggMeta' / f"{ea_id}_ggMeta.json")
    if gg_meta_raw and "data" in gg_meta_raw and "scores" in gg_meta_raw["data"]:
        for score in gg_meta_raw["data"]["scores"]:
            gg_scores_by_role[str(score.get("role"))].append(score)

    # esMeta indexed by role ONCE (the file repeats the same block ~10x)
    es_meta_raw = load_json_file(RAW_DATA_DIR / sub / 'esMeta' / f"{ea_id}_esMeta.json", [])
    es_by_role = defaultdict(list)
    for block in es_meta_raw:
        for r in ((block.get("data", {}) or {}).get("metaRatings", []) or []):
            es_by_role[str(r.get("playerRoleId"))].append(r)

    chem_boosts = GLOBAL['chemBoosts']
    models = GLOBAL['models']
    player["metaRatings"] = []

    for role_id_str, scores in gg_scores_by_role.items():
        role_name = GLOBAL['role'].get(role_id_str)
        if not role_name:
            continue

        meta_entry = {"role": role_name, "subAccelType": sub_accel_type}

        best_gg = max(scores, key=lambda x: x.get("score", 0), default=None)
        if best_gg:
            meta_entry["ggMeta"] = round(best_gg.get("score", 0.0), 2)
            meta_entry["ggChemStyle"] = GLOBAL['ggChem'].get(str(best_gg.get("chemistryStyle")))
            boosts = chem_boosts.get((meta_entry.get("ggChemStyle") or "").lower(), {})
            meta_entry["ggAccelType"] = calculate_acceleration_type(
                get_attribute_with_boost(base_attributes, "attributeAcceleration", boosts),
                get_attribute_with_boost(base_attributes, "attributeAgility", boosts),
                get_attribute_with_boost(base_attributes, "attributeStrength", boosts),
                player.get("height"), gender_text
            )

        # ggMetaSub via the anchored GG model (same model club.3 uses for non-evo)
        gg_sub_model = models.get_model(role_name, 'ggMetaSub')
        if gg_sub_model:
            features_no_chem = prepare_features(player, maps, boosts={}, role_name=role_name)
            meta_entry["ggMetaSub"] = predict_ggsub_absolute(gg_sub_model, features_no_chem, cap_to_ggmeta=meta_entry.get("ggMeta"))

        # ES Meta for this role
        es_role_id = GLOBAL['roleToEs'].get(role_name)
        if es_role_id:
            filtered = es_by_role.get(str(es_role_id), [])
            if filtered:
                r0 = next((r for r in filtered if r.get("chemistry") == 0 and r.get("metaRating") is not None), None)
                if r0:
                    meta_entry["esMetaSub"] = round(float(r0["metaRating"]), 2)
                best3 = max([r for r in filtered if r.get("chemistry") == 3], key=lambda x: x.get("metaRating", -1), default=None)
                if best3 and best3.get("metaRating") is not None:
                    meta_entry["esMeta"] = round(float(best3["metaRating"]), 2)
                    meta_entry["esChemStyle"] = GLOBAL['esChem'].get(str(best3.get("chemstyleId")))
                    es_boosts = chem_boosts.get((meta_entry.get("esChemStyle") or "").lower(), {})
                    meta_entry["esAccelType"] = calculate_acceleration_type(
                        get_attribute_with_boost(base_attributes, "attributeAcceleration", es_boosts),
                        get_attribute_with_boost(base_attributes, "attributeAgility", es_boosts),
                        get_attribute_with_boost(base_attributes, "attributeStrength", es_boosts),
                        player.get("height"), gender_text
                    )

        meta_entry["avgMeta"] = average_optional(meta_entry.get("ggMeta"), meta_entry.get("esMeta"))
        meta_entry["avgMetaSub"] = average_optional(meta_entry.get("ggMetaSub"), meta_entry.get("esMetaSub"))
        player["metaRatings"].append(meta_entry)

    return player


def main():
    multiprocessing.freeze_support()
    start = time.time()
    print(f"🚀 Building all-players summary (parallel)... [min_height={MIN_HEIGHT_CM}cm]")

    if not MAPS_FILE.exists():
        print("❌ Could not find maps.json. Exiting.")
        return

    # Scan for ggData files, keep the latest-modified per eaId across subfolders.
    player_latest = {}
    for sub in ['club - main', 'club - rest', 'training']:
        gg_dir = RAW_DATA_DIR / sub / 'ggData'
        if not gg_dir.exists():
            continue
        for f in gg_dir.glob("*_ggData.json"):
            ea_id = f.name.split('_')[0]
            try:
                mtime = f.stat().st_mtime
            except Exception:
                mtime = 0
            if ea_id not in player_latest or mtime > player_latest[ea_id][1]:
                player_latest[ea_id] = (str(f), mtime, sub)

    file_infos = [(v[0], v[2]) for v in sorted(player_latest.values(), key=lambda x: x[0])]
    print(f"ℹ️  Found {len(file_infos)} unique player files.")
    if not file_infos:
        print("⚠️ No player files found.")
        return

    max_workers = os.cpu_count() or 4
    print(f"🔥 Processing with {max_workers} CPU cores...")

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker, initargs=(MIN_HEIGHT_CM,)) as executor:
        for i, res in enumerate(executor.map(process_one, file_infos, chunksize=25)):
            if res:
                results.append(res)
            if (i + 1) % 2000 == 0 or (i + 1) == len(file_infos):
                print(f"  ⏳ {i + 1}/{len(file_infos)}...")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False)

    tall = sum(1 for p in results if p.get('isTall'))
    print(f"✅ Saved {len(results)} players ({tall} tall ≥{MIN_HEIGHT_CM}cm) to {OUTPUT_FILE.name} ({round(time.time() - start, 1)}s)")


if __name__ == "__main__":
    main()
