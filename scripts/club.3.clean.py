# club.3.clean.py (Turbo Edition)
# Optimization: Multiprocessing (All Cores) + Worker State Caching

import json
from pathlib import Path
from collections import defaultdict
import warnings
import os
import sys
import io
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
except Exception:
    pass
import concurrent.futures
import multiprocessing
from shared_utils import load_json_file, _normalize_gender, calculate_acceleration_type, get_attribute_with_boost, average_optional
from model_utils import ModelManager, prepare_features, predict_ggsub_absolute, predict_ggsub_evo_anchored, to_player_like_from_ggdata, compute_meta_entry

# Suppress warnings in main output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- Configuration ---
BASE_DATA_DIR = Path(__file__).resolve().parent / '../data'
MODELS_DIR = Path(__file__).resolve().parent / '../models'
RAW_DATA_DIR = BASE_DATA_DIR / 'raw'
PRICES_DIR  = RAW_DATA_DIR / 'prices'
EVOLAB_FILE = BASE_DATA_DIR / 'evolab.json'
MAPS_FILE = BASE_DATA_DIR / 'maps.json'
OUTPUT_FILE = BASE_DATA_DIR / 'club_final.json'
CLUB_IDS_FILE = BASE_DATA_DIR / 'all_club_ids.json'
EVO_ESMETA_FILE = RAW_DATA_DIR / 'evo_esmeta.json'

def get_raw_file_path(ea_id, file_type, raw_data_dir):
    for sub in ['club - main', 'club - rest', 'training']:
        p = raw_data_dir / sub / file_type / f"{ea_id}_{file_type}.json"
        if p.exists():
            return p
    return raw_data_dir / 'club - main' / file_type / f"{ea_id}_{file_type}.json"

# --- CLI Arguments ---
MIN_HEIGHT_CM = 195  # default
for i, arg in enumerate(sys.argv):
    if arg == '--min-height' and i + 1 < len(sys.argv):
        try:
            MIN_HEIGHT_CM = int(sys.argv[i + 1])
        except ValueError:
            pass
        break

# --- Global Worker State (Initialized once per process) ---
GLOBAL_MAPS = None
GLOBAL_MODEL_MANAGER = None
GLOBAL_EVO_ESMETA = None

# --- Helpers ---


def parse_gg_rating_str(gg_rating_str_raw):
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

def resolve_anchor_source_eaid(evo_def):
    # basePlayerEaId is the actual key fut.gg's evolab uses for the pre-evo base card;
    # the rest are defensive fallbacks. eaId is the last resort (today it equals
    # basePlayerEaId, so anchoring only worked by accident before this key was added).
    for k in ("basePlayerEaId", "baseEaId", "originalEaId", "rootEaId", "rootDefinitionEaId", "baseItemEaId", "baseCardEaId"):
        v = evo_def.get(k)
        if v is not None:
            try: return str(int(v))
            except Exception: continue
    eid = evo_def.get("eaId")
    return str(int(eid)) if eid is not None else None

# ModelManager, prepare_features, predict_ggsub_*, and to_player_like_from_ggdata
# now live in model_utils.py (shared with build_all_players_summary.py).

# --- Core Logic (Runs inside Worker) ---
def process_player(player_def, is_evo, model_manager, maps, evo_esmeta, min_height_cm=195):
    # This function contains the heavy lifting logic
    player_output = {"eaId": player_def.get("eaId"), "evolution": is_evo}
    base_attributes = {k: v for k, v in player_def.items() if k.startswith("attribute")}
    for key in ["commonName","overall","height","weight","skillMoves","weakFoot"]:
        player_output[key] = player_def.get(key)
    player_output.update(base_attributes)
    player_output["gender"] = _normalize_gender(player_def.get("gender"), maps)

    # Tag tall players
    try:
        player_height = int(player_output.get("height") or 0)
    except (ValueError, TypeError):
        player_height = 0
    player_output["isTall"] = player_height >= min_height_cm

    numeric_positions = [str(p) for p in [player_def.get("position")] + (player_def.get("alternativePositionIds") or []) if p is not None]
    player_output["positions"] = list(set([maps.get("position", {}).get(p) for p in numeric_positions if p in maps.get("position", {})]))
    player_output["foot"] = maps.get("foot", {}).get(str(player_def.get("foot")))
    player_output["PS"] = [maps.get("playstyles", {}).get(str(p)) for p in (player_def.get("playstyles") or []) if str(p) in maps.get("playstyles", {})]
    player_output["PS+"] = [maps.get("playstyles", {}).get(str(p)) for p in (player_def.get("playstylesPlus") or []) if str(p) in maps.get("playstyles", {})]
    player_output["roles+"]  = [maps.get("rolesPlus", {}).get(str(r)) for r in (player_def.get("rolesPlus") or []) if str(r) in maps.get("rolesPlus", {})]
    player_output["roles++"] = [maps.get("rolesPlusPlus", {}).get(str(r)) for r in (player_def.get("rolesPlusPlus") or []) if str(r) in maps.get("rolesPlusPlus", {})]
    player_output["bodyType"] = maps.get("bodytypeCode", {}).get(str(player_def.get("bodytypeCode")))
    rarity_data = player_def.get("rarity")
    player_output["rarity"] = rarity_data.get("name") if isinstance(rarity_data, dict) else None

    # Load Price
    price_file = PRICES_DIR / f"{player_output['eaId']}.json"
    if price_file.exists():
        price_data = load_json_file(price_file)
        if price_data:
            player_output["price"] = price_data.get("price")
            player_output["isExtinct"] = price_data.get("isExtinct")
    else:
        player_output["price"] = None
        player_output["isExtinct"] = None
    # (discardValue lives in tradeable_details.json and is surfaced by the Sell view;
    #  it was never written into the price files, so there's nothing to merge here.)

    sub_accel_type = calculate_acceleration_type(
        base_attributes.get("attributeAcceleration"), base_attributes.get("attributeAgility"),
        base_attributes.get("attributeStrength"), player_output.get("height"), player_output.get("gender")
    )

    if is_evo:
        # Exact evo esMeta from EasySBC — same metaRatings shape as the base esMeta API,
        # so it flows through the identical parser below (wrapped as a single block).
        evo_es = (evo_esmeta or {}).get(str(player_output['eaId']))
        es_meta_raw = [{"data": {"metaRatings": evo_es.get("metaRatings", [])}}] if evo_es else []
        gg_scores_by_role = parse_gg_rating_str(player_def.get("ggRatingStr"))
    else:
        es_meta_raw = load_json_file(get_raw_file_path(player_output['eaId'], 'esMeta', RAW_DATA_DIR), [])
        gg_meta_raw = load_json_file(get_raw_file_path(player_output['eaId'], 'ggMeta', RAW_DATA_DIR))
        gg_scores_by_role = defaultdict(list)
        if gg_meta_raw and "data" in gg_meta_raw and "scores" in gg_meta_raw["data"]:
            for score in gg_meta_raw["data"]["scores"]:
                gg_scores_by_role[str(score.get("role"))].append(score)

    player_output["metaRatings"] = []

    # Built ONCE per player (not per role): the chem-style boost map and the esMeta
    # entries indexed by role. The base esMeta file repeats the same block ~10x, so
    # scanning it inside the role loop was ~10x×(#roles) redundant work.
    chem_style_boosts_map = {
        item['name'].lower(): item['threeChemistryModifiers']
        for item in maps.get("ChemistryStylesBoosts", [])
        if 'name' in item and 'threeChemistryModifiers' in item
    }
    es_by_role = defaultdict(list)
    for b in es_meta_raw:
        for r in ((b.get("data", {}) or {}).get("metaRatings", []) or []):
            es_by_role[str(r.get("playerRoleId"))].append(r)

    # Iterate Roles
    for role_id_str, scores in gg_scores_by_role.items():
        role_name = maps.get("role", {}).get(role_id_str)
        if not role_name: continue

        # Resolve GG pieces (evo vs non-evo differ), then delegate assembly to the
        # shared compute_meta_entry (also used by build_all_players_summary).
        best_gg = max(scores, key=lambda x: x.get("score", 0), default=None)
        gg_score = best_gg.get("score", 0.0) if best_gg else None
        gg_chem_style = None
        if best_gg:
            chem_id = best_gg.get("chem_id_str") if is_evo else str(best_gg.get("chemistryStyle"))
            gg_chem_style = maps.get("ggChemistryStyleNames", {}).get(chem_id)
        gg_meta_cap = round(gg_score, 2) if gg_score is not None else None

        # ggMetaSub: evo -> anchored to the base card; non-evo -> absolute; GK fallback.
        gg_meta_sub = None
        gg_sub_model = model_manager.get_model(role_name, 'ggMetaSub')
        if gg_sub_model:
            features_no_chem = prepare_features(player_output, maps, boosts={}, role_name=role_name)
            if is_evo:
                base_anchor_eaid = resolve_anchor_source_eaid(player_def)
                base_gg_raw = load_json_file(get_raw_file_path(base_anchor_eaid, 'ggData', RAW_DATA_DIR))
                base_player_like = to_player_like_from_ggdata(base_gg_raw.get("data") if base_gg_raw else None, maps)
                base_features_sub = prepare_features(base_player_like, maps, boosts={}, role_name=role_name) if base_player_like else None
                gg_meta_sub = predict_ggsub_evo_anchored(gg_sub_model, features_no_chem, base_no_chem=base_features_sub, cap_to_ggmeta=gg_meta_cap)
            else:
                gg_meta_sub = predict_ggsub_absolute(gg_sub_model, features_no_chem, cap_to_ggmeta=gg_meta_cap)
        elif not is_evo and "GK" in role_name and gg_meta_cap:
            gg_meta_sub = round(gg_meta_cap * 0.95, 2)

        es_role_id = maps.get("roleNameToEsRoleId", {}).get(role_name)
        es_entries = es_by_role.get(str(es_role_id), []) if es_role_id else []

        player_output["metaRatings"].append(compute_meta_entry(
            role_name, sub_accel_type,
            gg_score=gg_score, gg_chem_style=gg_chem_style, gg_meta_sub=gg_meta_sub,
            es_entries=es_entries, base_attributes=base_attributes,
            height=player_output.get("height"), gender=player_output.get("gender"),
            es_chem_names=maps.get("esChemistryStyleNames", {}), chem_boosts=chem_style_boosts_map,
        ))

    return player_output

# --- Multiprocessing Wrapper ---
def init_worker(min_height_cm):
    """Initializes global state in each worker process."""
    global GLOBAL_MAPS, GLOBAL_MODEL_MANAGER, GLOBAL_MIN_HEIGHT, GLOBAL_EVO_ESMETA

    GLOBAL_MIN_HEIGHT = min_height_cm

    # Load Maps
    if GLOBAL_MAPS is None:
        GLOBAL_MAPS = load_json_file(MAPS_FILE)

    # Load Models
    if GLOBAL_MODEL_MANAGER is None:
        GLOBAL_MODEL_MANAGER = ModelManager(MODELS_DIR)

    # Load exact evo esMeta (from EasySBC via fetch.evo.esmeta.js), keyed by eaId
    if GLOBAL_EVO_ESMETA is None:
        GLOBAL_EVO_ESMETA = load_json_file(EVO_ESMETA_FILE, {}) or {}

def worker_task(payload):
    """
    Payload is a dict: {'type': 'club', 'id': ...} OR {'type': 'evo', 'data': ...}
    """
    try:
        if payload['type'] == 'club':
            ea_id = payload['id']
            data_file = get_raw_file_path(ea_id, 'ggData', RAW_DATA_DIR)
            player_def_raw = load_json_file(data_file)
            if not player_def_raw or "data" not in player_def_raw:
                return None
            return process_player(player_def_raw["data"], False, GLOBAL_MODEL_MANAGER, GLOBAL_MAPS, GLOBAL_EVO_ESMETA, GLOBAL_MIN_HEIGHT)

        elif payload['type'] == 'evo':
            # Evo data is passed directly
            return process_player(payload['data'], True, GLOBAL_MODEL_MANAGER, GLOBAL_MAPS, GLOBAL_EVO_ESMETA, GLOBAL_MIN_HEIGHT)
            
    except Exception as e:
        # Surface (don't silently drop) so a systemic regression shrinking club_final
        # is visible. Main separates these from real results before dedup.
        pid = payload.get('id') or (payload.get('data') or {}).get('eaId')
        return {'__error': f"{type(e).__name__}: {e}", '__id': pid}

# --- Main ---
def main():
    # Windows requires this for multiprocessing
    multiprocessing.freeze_support()
    
    print(f"🚀 Starting TURBO Cleaning Script (Multiprocessing)... [min_height={MIN_HEIGHT_CM}cm]")
    
    # Prepare Tasks
    tasks = []
    
    # 1. Club Players (Pass IDs)
    club_ids_data = load_json_file(CLUB_IDS_FILE)
    if club_ids_data:
        print(f"ℹ️ Queueing {len(club_ids_data)} club players...")
        for ea_id in club_ids_data:
            tasks.append({'type': 'club', 'id': str(ea_id)})
            
    # 2. Evo Players (Pass Data)
    evolab_data = load_json_file(EVOLAB_FILE)
    if evolab_data and "data" in evolab_data:
        evo_defs = [item["playerItemDefinition"] for item in evolab_data["data"] if "playerItemDefinition" in item]
        print(f"ℹ️ Queueing {len(evo_defs)} evo players...")
        for p_def in evo_defs:
            tasks.append({'type': 'evo', 'data': p_def})

    total_tasks = len(tasks)
    if total_tasks == 0:
        print("⚠️ No tasks found.")
        return

    # Use all CPU cores
    max_workers = os.cpu_count()
    print(f"🔥 processing with {max_workers} CPU cores...")
    
    results = []
    completed = 0
    
    # Executor with Initializer
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker, initargs=(MIN_HEIGHT_CM,)) as executor:
        # Submit all tasks
        futures = [executor.submit(worker_task, t) for t in tasks]
        
        # Monitor progress
        errors = []
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res and '__error' in res:
                errors.append(res)
            elif res:
                results.append(res)

            completed += 1
            if completed % 100 == 0 or completed == total_tasks:
                print(f"  - Processed {completed}/{total_tasks} players...")

        if errors:
            print(f"⚠️  {len(errors)} player(s) failed during processing. Examples:")
            for err in errors[:5]:
                print(f"    - id {err.get('__id')}: {err.get('__error')}")

    # Deduplicate by eaId. Evolutions share their base card's eaId, so the
    # `or player.get('evolution')` intentionally lets an evo overwrite its base card
    # (we surface the evolved version). A base card with no evo is kept as-is.
    print("\n--- Deduplicating ---")
    final_players = {}
    for player in results:
        if player['eaId'] not in final_players or player.get('evolution'):
            final_players[player['eaId']] = player
    
    # Filter: include only players with overall >= 75 OR isTall
    filtered_list = []
    skipped_count = 0
    for player in final_players.values():
        ovr = 0
        try:
            ovr = int(player.get('overall') or 0)
        except (ValueError, TypeError):
            pass
        is_tall = player.get('isTall', False)
        if ovr >= 75 or is_tall:
            filtered_list.append(player)
        else:
            skipped_count += 1
    
    final_list = filtered_list
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_list, f, indent=2, ensure_ascii=False)
    
    tall_count = sum(1 for p in final_list if p.get('isTall', False))
    print(f"\n🎉 Success! Processed {len(final_list)} unique players to {OUTPUT_FILE.name}")
    print(f"   │ {tall_count} tall players (≥{MIN_HEIGHT_CM}cm) included")
    if skipped_count > 0:
        print(f"   │ {skipped_count} players skipped (OVR < 75 and not tall)")

if __name__ == "__main__":
    main()