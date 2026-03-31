# shared_utils.py
# Contains core game logic and shared helpers to prevent logic drift across scripts.

import json
from pathlib import Path

def load_json_file(file_path: Path, default_val=None):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default_val

def _normalize_gender(gender_val, maps):
    try:
        g = maps.get("gender", {}).get(str(gender_val))
        if isinstance(g, str) and g:
            return g
    except Exception:
        pass
    return "Male"

def calculate_acceleration_type(accel, agility, strength, height, gender: str = "Male"):
    """
    Gender-specific height rules (new game update):

    Explosive:
      - Agility >= 65
      - (Agility - Strength) >= 10
      - Acceleration >= 80
      - Height <= 182 (Male) OR <= 162 (Female)
    Lengthy:
      - Strength >= 65
      - (Strength - Agility) >= 4
      - Acceleration >= 40
      - Height >= 185 (Male) OR >= 165 (Female)
    Controlled otherwise.
    """
    try:
        if accel is None or agility is None or strength is None or height is None:
            return "CONTROLLED"
        accel = int(accel); agility = int(agility); strength = int(strength); height = int(height)
    except Exception:
        return "CONTROLLED"

    is_female = str(gender or "Male").lower().startswith("f")
    exp_height_ok = (height <= 162) if is_female else (height <= 182)
    len_height_ok = (height >= 165) if is_female else (height >= 185)

    if (agility >= 65 and (agility - strength) >= 10 and accel >= 80 and exp_height_ok):
        return "EXPLOSIVE"
    if (strength >= 65 and (strength - agility) >= 4 and accel >= 40 and len_height_ok):
        return "LENGTHY"
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

def familiarity_for_role(role_name, player_def, maps):
    """
    0 = none, 1 = roles+, 2 = roles++
    """
    roles_plus_map = maps.get("rolesPlus", {})
    roles_pp_map   = maps.get("rolesPlusPlus", {})
    
    plus_names = {roles_plus_map.get(str(x)) for x in (player_def.get("rolesPlus") or [])}
    pp_names   = {roles_pp_map.get(str(x)) for x in (player_def.get("rolesPlusPlus") or [])}
    
    plus_names = {n for n in plus_names if n}
    pp_names   = {n for n in pp_names if n}
    
    if role_name in pp_names: return 2
    if role_name in plus_names: return 1
    return 0
