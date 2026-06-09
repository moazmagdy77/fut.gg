import streamlit as st
import pandas as pd
import json
from pathlib import Path

st.set_page_config(layout="wide")

# Define data directory
data_dir = Path(__file__).resolve().parents[1] / "data"

maps_path = data_dir / "maps.json"
maps_data = {}
chem_style_boosts = []
roles_order = []
positions_order = []
try:
    with open(maps_path, 'r', encoding='utf-8') as f:
        maps_data = json.load(f)
        chem_style_boosts = maps_data.get("ChemistryStylesBoosts", [])
        positions_map = maps_data.get("position", {})
        if positions_map:
            positions_order = [v for k, v in sorted(positions_map.items(), key=lambda item: int(item[0]))]
        rolesPlus = maps_data.get("rolesPlus", {})
        if rolesPlus:
            roles_order = [v for k, v in sorted(rolesPlus.items(), key=lambda item: int(item[0]))]
except Exception:
    pass

# This list helps define the order of attributes in the final display
attribute_filter_order = [
    "acceleration", "sprintSpeed", "positioning", "finishing", "shotPower", "longShots",
    "volleys", "penalties", "vision", "crossing", "fkAccuracy", "shortPassing",
    "longPassing", "curve", "agility", "balance", "reactions", "ballControl",
    "dribbling", "composure", "interceptions", "headingAccuracy", "defensiveAwareness",
    "standingTackle", "slidingTackle", "jumping", "stamina", "strength", "aggression",
    "gkDiving", "gkHandling", "gkKicking", "gkPositioning", "gkReflexes"
]

FORMATION_GROUPS = {
    "3-back": [
        "3-1-4-2", "3-4-1-2", "3-4-2-1", "3-4-3", "3-5-2",
    ],
    "4-back": [
        "4-1-2-1-2", "4-1-2-1-2 (2)", "4-1-3-2", "4-1-4-1",
        "4-2-1-3", "4-2-2-2", "4-2-3-1", "4-2-3-1 (2)", "4-2-4",
        "4-3-1-2", "4-3-2-1", "4-3-3", "4-3-3 (2)", "4-3-3 (3)",
        "4-3-3 (4)", "4-4-1-1 (2)", "4-4-2", "4-4-2 (2)", "4-5-1",
        "4-5-1 (2)",
    ],
    "5-back": [
        "5-2-1-2", "5-2-3", "5-3-2", "5-4-1",
    ],
}

FC26_FORMATIONS = {
    "3-1-4-2": [["ST", "ST"], ["LM", "CM", "CM", "RM"], ["CDM"], ["CB", "CB", "CB"], ["GK"]],
    "3-4-1-2": [["ST", "ST"], ["CAM"], ["LM", "CM", "CM", "RM"], ["CB", "CB", "CB"], ["GK"]],
    "3-4-2-1": [["ST"], ["LW", "RW"], ["LM", "CM", "CM", "RM"], ["CB", "CB", "CB"], ["GK"]],
    "3-4-3": [["LW", "ST", "RW"], ["LM", "CM", "CM", "RM"], ["CB", "CB", "CB"], ["GK"]],
    "3-5-2": [["ST", "ST"], ["CAM"], ["LM", "CDM", "CDM", "RM"], ["CB", "CB", "CB"], ["GK"]],
    "4-1-2-1-2": [["ST", "ST"], ["CAM"], ["LM", "RM"], ["CDM"], ["LB", "CB", "CB", "RB"], ["GK"]],
    "4-1-2-1-2 (2)": [["ST", "ST"], ["CAM"], ["CM", "CM"], ["CDM"], ["LB", "CB", "CB", "RB"], ["GK"]],
    "4-1-3-2": [["ST", "ST"], ["LM", "CM", "RM"], ["CDM"], ["LB", "CB", "CB", "RB"], ["GK"]],
    "4-1-4-1": [["ST"], ["LM", "CM", "CM", "RM"], ["CDM"], ["LB", "CB", "CB", "RB"], ["GK"]],
    "4-2-1-3": [["LW", "ST", "RW"], ["CAM"], ["CDM", "CDM"], ["LB", "CB", "CB", "RB"], ["GK"]],
    "4-2-2-2": [["ST", "ST"], ["CAM", "CAM"], ["CDM", "CDM"], ["LB", "CB", "CB", "RB"], ["GK"]],
    "4-2-3-1": [["ST"], ["CAM", "CAM", "CAM"], ["CDM", "CDM"], ["LB", "CB", "CB", "RB"], ["GK"]],
    "4-2-3-1 (2)": [["ST"], ["LM", "CAM", "RM"], ["CDM", "CDM"], ["LB", "CB", "CB", "RB"], ["GK"]],
    "4-2-4": [["LW", "ST", "ST", "RW"], ["CM", "CM"], ["LB", "CB", "CB", "RB"], ["GK"]],
    "4-3-1-2": [["ST", "ST"], ["CAM"], ["CM", "CM", "CM"], ["LB", "CB", "CB", "RB"], ["GK"]],
    "4-3-2-1": [["ST"], ["LW", "RW"], ["CM", "CM", "CM"], ["LB", "CB", "CB", "RB"], ["GK"]],
    "4-3-3": [["LW", "ST", "RW"], ["CM", "CM", "CM"], ["LB", "CB", "CB", "RB"], ["GK"]],
    "4-3-3 (2)": [["LW", "ST", "RW"], ["CM", "CM"], ["CDM"], ["LB", "CB", "CB", "RB"], ["GK"]],
    "4-3-3 (3)": [["LW", "ST", "RW"], ["CM"], ["CDM", "CDM"], ["LB", "CB", "CB", "RB"], ["GK"]],
    "4-3-3 (4)": [["LW", "ST", "RW"], ["CAM"], ["CM", "CM"], ["LB", "CB", "CB", "RB"], ["GK"]],
    "4-4-1-1 (2)": [["ST"], ["CAM"], ["LM", "CM", "CM", "RM"], ["LB", "CB", "CB", "RB"], ["GK"]],
    "4-4-2": [["ST", "ST"], ["LM", "CM", "CM", "RM"], ["LB", "CB", "CB", "RB"], ["GK"]],
    "4-4-2 (2)": [["ST", "ST"], ["LM", "CDM", "CDM", "RM"], ["LB", "CB", "CB", "RB"], ["GK"]],
    "4-5-1": [["ST"], ["CAM"], ["LM", "CM", "CM", "RM"], ["LB", "CB", "CB", "RB"], ["GK"]],
    "4-5-1 (2)": [["ST"], ["LM", "CAM", "RM"], ["CM", "CM"], ["LB", "CB", "CB", "RB"], ["GK"]],
    "5-2-1-2": [["ST", "ST"], ["CAM"], ["CM", "CM"], ["LB", "CB", "CB", "CB", "RB"], ["GK"]],
    "5-2-3": [["LW", "ST", "RW"], ["CM", "CM"], ["LB", "CB", "CB", "CB", "RB"], ["GK"]],
    "5-3-2": [["ST", "ST"], ["CM", "CM", "CM"], ["LB", "CB", "CB", "CB", "RB"], ["GK"]],
    "5-4-1": [["ST"], ["LM", "CM", "CM", "RM"], ["LB", "CB", "CB", "CB", "RB"], ["GK"]],
}

DEFAULT_ROLE_BY_POSITION = {
    "GK": "GK Goalkeeper",
    "RB": "RB Fullback",
    "CB": "CB Defender",
    "LB": "LB Fullback",
    "CDM": "CDM Holding",
    "CM": "CM Box to Box",
    "RM": "RM Winger",
    "LM": "LM Winger",
    "CAM": "CAM Playmaker",
    "RW": "RW Winger",
    "ST": "ST Advanced Forward",
    "LW": "LW Winger",
}

SIDE_PREFIX_BY_POSITION = {
    "ST": ("LST", "RST"),
    "CB": ("LCB", "RCB"),
    "CDM": ("LCDM", "RCDM"),
    "CM": ("LCM", "RCM"),
    "CAM": ("LCAM", "RCAM"),
}

def sort_by_reference(values, reference_order):
    values = [v for v in values if pd.notna(v)]
    if not reference_order:
        return sorted(values)
    value_order = {value: idx for idx, value in enumerate(reference_order)}
    return sorted(values, key=lambda value: (value_order.get(value, len(value_order)), str(value)))

def unique_positions_from_df(df_to_scan):
    found = []
    if "positions" not in df_to_scan.columns:
        return positions_order
    for positions in df_to_scan["positions"].dropna():
        if isinstance(positions, list):
            found.extend(positions)
    return sort_by_reference(sorted(set(found)), positions_order)

def role_options_for_position(position, available_roles):
    position_roles = [role for role in available_roles if str(role).startswith(f"{position} ")]
    return position_roles or available_roles

def default_role_for_position(position, available_roles):
    options = role_options_for_position(position, available_roles)
    preferred = DEFAULT_ROLE_BY_POSITION.get(position)
    if preferred in options:
        return preferred
    return options[0] if options else ""

def safe_widget_key(value):
    return "".join(ch if ch.isalnum() else "_" for ch in str(value))

def slot_labels_for_row(row_positions):
    counts = {position: row_positions.count(position) for position in row_positions}
    seen = {}
    labels = []
    for position in row_positions:
        seen[position] = seen.get(position, 0) + 1
        if counts[position] == 1:
            labels.append(position)
        elif counts[position] == 2 and position in SIDE_PREFIX_BY_POSITION:
            labels.append(SIDE_PREFIX_BY_POSITION[position][seen[position] - 1])
        elif counts[position] == 3 and position == "CB":
            labels.append(["LCB", "CB", "RCB"][seen[position] - 1])
        elif counts[position] == 3 and position == "CM":
            labels.append(["LCM", "CM", "RCM"][seen[position] - 1])
        elif counts[position] == 3 and position == "CAM":
            labels.append(["LCAM", "CAM", "RCAM"][seen[position] - 1])
        else:
            labels.append(f"{position} {seen[position]}")
    return labels

def centered_columns(slot_count):
    if slot_count == 1:
        cols = st.columns([2.2, 1.2, 2.2])
        return [cols[1]]
    if slot_count == 2:
        cols = st.columns([1.1, 1.2, 1.2, 1.1])
        return cols[1:3]
    if slot_count == 3:
        cols = st.columns([0.6, 1.2, 1.2, 1.2, 0.6])
        return cols[1:4]
    return st.columns(slot_count)

def clear_squad_builder_filters():
    st.session_state["squad_active_slot"] = None
    st.session_state["squad_applied_signature"] = ""
    st.session_state["squad_applied_position"] = ""
    st.session_state["squad_applied_role"] = ""
    st.session_state["position_filter"] = []
    st.session_state["role_filter"] = []

def apply_squad_slot_filter(slot_id, position, role):
    st.session_state["squad_active_slot"] = slot_id
    st.session_state["position_filter"] = [position]
    st.session_state["role_filter"] = [role]
    st.session_state["squad_applied_position"] = position
    st.session_state["squad_applied_role"] = role
    st.session_state["squad_applied_signature"] = f"{slot_id}|{position}|{role}"

def render_formation_grid():
    with st.expander("All FC26 Formations", expanded=False):
        for group_name, formation_names in FORMATION_GROUPS.items():
            st.markdown(f"#### {group_name}")
            for start in range(0, len(formation_names), 4):
                cols = st.columns(4)
                for col, formation_name in zip(cols, formation_names[start:start + 4]):
                    button_type = "primary" if st.session_state.get("squad_selected_formation") == formation_name else "secondary"
                    if col.button(formation_name, key=f"formation_button_{safe_widget_key(formation_name)}", type=button_type, use_container_width=True):
                        st.session_state["squad_pending_formation"] = formation_name
                        st.rerun()

def render_squad_builder(df_to_use, available_roles):
    pending_formation = st.session_state.pop("squad_pending_formation", None)
    if pending_formation in FC26_FORMATIONS:
        st.session_state["squad_selected_formation"] = pending_formation
        clear_squad_builder_filters()

    formation_names = [name for names in FORMATION_GROUPS.values() for name in names]
    selected_default = st.session_state.get("squad_selected_formation", "4-4-1-1 (2)")
    if selected_default not in formation_names:
        selected_default = "4-4-1-1 (2)"

    st.markdown("## Squad Builder")
    selector_col, count_col, clear_col = st.columns([3, 1, 1])
    selected_formation = selector_col.selectbox(
        "Formation",
        formation_names,
        index=formation_names.index(selected_default),
    )

    previous_formation = st.session_state.get("squad_last_formation")
    if previous_formation is None:
        st.session_state["squad_last_formation"] = selected_formation
        st.session_state["squad_selected_formation"] = selected_formation
    elif previous_formation != selected_formation:
        st.session_state["squad_last_formation"] = selected_formation
        st.session_state["squad_selected_formation"] = selected_formation
        clear_squad_builder_filters()
        st.rerun()
    else:
        st.session_state["squad_selected_formation"] = selected_formation

    active_slot = st.session_state.get("squad_active_slot")
    count_col.metric("Slots", sum(len(row) for row in FC26_FORMATIONS[selected_formation]))
    if clear_col.button("Clear Slot", use_container_width=True):
        clear_squad_builder_filters()
        st.rerun()

    render_formation_grid()

    with st.container(border=True):
        st.markdown(f"### {selected_formation}")
        for row_index, row_positions in enumerate(FC26_FORMATIONS[selected_formation]):
            slot_columns = centered_columns(len(row_positions))
            row_labels = slot_labels_for_row(row_positions)
            for slot_index, (slot_col, position, label) in enumerate(zip(slot_columns, row_positions, row_labels)):
                slot_id = f"{safe_widget_key(selected_formation)}_{row_index}_{slot_index}_{label}_{position}"
                role_key = f"squad_role_{slot_id}"
                role_options = role_options_for_position(position, available_roles)
                default_role = default_role_for_position(position, available_roles)
                if role_key not in st.session_state or st.session_state[role_key] not in role_options:
                    st.session_state[role_key] = default_role

                with slot_col:
                    selected_role = st.session_state[role_key]
                    button_type = "primary" if active_slot == slot_id else "secondary"
                    if st.button(label, key=f"squad_slot_{slot_id}", type=button_type, use_container_width=True):
                        apply_squad_slot_filter(slot_id, position, selected_role)
                        st.rerun()

                    selected_role = st.selectbox(
                        f"{label} role",
                        role_options,
                        key=role_key,
                        label_visibility="collapsed",
                    )

                    signature = f"{slot_id}|{position}|{selected_role}"
                    if active_slot == slot_id and st.session_state.get("squad_applied_signature") != signature:
                        apply_squad_slot_filter(slot_id, position, selected_role)
                        st.rerun()
            if row_index < len(FC26_FORMATIONS[selected_formation]) - 1:
                st.write("")

    active_position = st.session_state.get("squad_applied_position")
    active_role = st.session_state.get("squad_applied_role")
    if active_slot and active_position and active_role:
        match_count = df_to_use[
            df_to_use["positions"].apply(lambda vals: isinstance(vals, list) and active_position in vals)
            & (df_to_use["role"] == active_role)
        ]["__true_player_id"].nunique()
        st.caption(f"Selected: {active_position} / {active_role} - {match_count} matching club players before sidebar filters.")

def positions_match(player_positions, selected_positions):
    if not isinstance(player_positions, list):
        return False
    return bool(set(player_positions).intersection(selected_positions))

def apply_filters_to_df(source_df, active_filters):
    filtered = source_df.copy()
    for col, val in active_filters.items():
        if val is None or (isinstance(val, list) and not val):
            continue
        if col not in filtered.columns and col not in ["playstyles_all", "playstyles_plus_all"]:
            continue

        if col == "playstyles_all":
            def has_all_styles(row):
                combined = set((row.get('PS', []) or []) + (row.get('PS+', []) or []))
                return all(s in combined for s in val)
            filtered = filtered[filtered.apply(has_all_styles, axis=1)]
        elif col == "playstyles_plus_all":
            def has_all_ps_plus(ps_plus_list):
                return isinstance(ps_plus_list, list) and all(s in ps_plus_list for s in val)
            filtered = filtered[filtered['PS+'].apply(has_all_ps_plus)]
        elif col == "positions":
            selected_positions = set(val)
            filtered = filtered[filtered["positions"].apply(lambda positions: positions_match(positions, selected_positions))]
        elif isinstance(val, list):
            filtered = filtered[filtered[col].isin(val)]
        elif isinstance(val, tuple):
            filtered = filtered[filtered[col].between(val[0], val[1])]
        else:
            filtered = filtered[filtered[col] == val]
    return filtered

def load_data(file_path):
    """Loads and preprocesses the final JSON data."""
    try:
        with open(file_path, "r", encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        st.error(f"Error: `club_final.json` not found in the '{data_dir}' directory. Please ensure the processing script has run successfully.")
        return pd.DataFrame()
    except json.JSONDecodeError:
        st.error(f"Error: `club_final.json` is not a valid JSON file. Please check the file content.")
        return pd.DataFrame()

    df = pd.json_normalize(data, errors='ignore')

    # Rename attribute columns from 'attributeAcceleration' to 'acceleration'
    df.rename(columns={col: col.replace("attribute", "", 1)[0].lower() + col.replace("attribute", "", 1)[1:] 
                       for col in df.columns if col.startswith("attribute")}, inplace=True)

    # Now, safely explode the metaRatings column
    df = df.explode('metaRatings').reset_index(drop=True)
    meta_df = pd.json_normalize(df['metaRatings']).add_prefix('meta.')
    df = pd.concat([df.drop(columns=['metaRatings']), meta_df], axis=1)

    # Rename the flattened meta columns
    df.rename(columns={col: col.replace("meta.", "") for col in df.columns if col.startswith("meta.")}, inplace=True)
    
    if df.empty:
        st.warning("No data loaded or normalized from club_final.json.")
        return pd.DataFrame()

    df['player_origin_id'] = df['eaId'].astype(str) + '_' + df['evolution'].astype(str)
    df["__true_player_id"] = df["eaId"].astype(str)
    
    # Clean up data types
    int_cols = ['overall', 'height', 'weight', 'skillMoves', 'weakFoot'] + attribute_filter_order
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    # Calculate Responsiveness (Average of acceleration, sprintSpeed, agility, balance, reactions)
    resp_attrs = ['acceleration', 'sprintSpeed', 'agility', 'balance', 'reactions']
    if all(attr in df.columns for attr in resp_attrs):
        df['responsiveness'] = df[resp_attrs].mean(axis=1)
    else:
        df['responsiveness'] = 0.0

    float_cols = ['ggMeta', 'ggMetaSub', 'esMeta', 'esMetaSub', 'avgMeta', 'avgMetaSub', 'price', 'responsiveness', 'discardValue']
    for col in float_cols:
        if col in df.columns:
            if df[col].dtype == object:
                # Remove any formatting commas before conversion
                df[col] = df[col].astype(str).str.replace(',', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    # Recompute role familiarity flags based on the current role
    for col in ['roles+', 'roles++']:
        if col not in df.columns: df[col] = pd.Series([[] for _ in range(len(df))])
        df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])
        
    df['hasRolePlus'] = df.apply(lambda row: row.get('role') in row.get('roles+', []), axis=1)
    df['hasRolePlusPlus'] = df.apply(lambda row: row.get('role') in row.get('roles++', []), axis=1)

    def compute_lengthy(accel, agility, strength, height, gender):
        try:
            if pd.isna(accel) or pd.isna(agility) or pd.isna(strength) or pd.isna(height):
                return False
            accel = int(accel); agility = int(agility); strength = int(strength); height = int(height)
        except Exception:
            return False
            
        is_female = str(gender).lower().startswith("f") if pd.notna(gender) else False
        len_height_ok = (height >= 165) if is_female else (height >= 185)
        
        return (strength >= 65 and (strength - agility) >= 4 and accel >= 40 and len_height_ok)

    def get_lengthy_info(row):
        b_accel = row.get("acceleration", 0)
        b_agil = row.get("agility", 0)
        b_str = row.get("strength", 0)
        h = row.get("height", 0)
        g = row.get("gender", "Male")
        
        if compute_lengthy(b_accel, b_agil, b_str, h, g):
            return True, "0 (any)"
            
        for level_key, level_num in [("oneChemistryModifiers", 1), ("twoChemistryModifiers", 2), ("threeChemistryModifiers", 3)]:
            matching_chems = []
            for chem in chem_style_boosts:
                b = chem.get(level_key, {})
                na = min(int(b_accel) + int(b.get("attributeAcceleration", 0)), 99)
                nag = min(int(b_agil) + int(b.get("attributeAgility", 0)), 99)
                ns = min(int(b_str) + int(b.get("attributeStrength", 0)), 99)
                if compute_lengthy(na, nag, ns, h, g):
                    chem_name = chem.get("name", "").lower()
                    if chem_name:
                        matching_chems.append(chem_name)
            if matching_chems:
                return True, f"{level_num} ({'/'.join(sorted(set(matching_chems)))})"
        
        return False, ""

    if chem_style_boosts:
        info_series = df.apply(get_lengthy_info, axis=1)
        df['canBeLengthy'] = info_series.apply(lambda x: x[0])
        df['Chem points needed for Lengthy'] = info_series.apply(lambda x: x[1])
    else:
        df['canBeLengthy'] = False
        df['Chem points needed for Lengthy'] = ""

    return df

def load_all_players(file_path):
    """Loads the pre-built all players summary JSON."""
    try:
        with open(file_path, "r", encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        st.warning(f"All players data not found. Run `build_all_players_summary.py` to generate it.")
        return pd.DataFrame()
    
    df = pd.json_normalize(data, errors='ignore')
    
    # Explode metaRatings
    if 'metaRatings' in df.columns:
        df = df.explode('metaRatings').reset_index(drop=True)
        meta_df = pd.json_normalize(df['metaRatings']).add_prefix('meta.')
        df = pd.concat([df.drop(columns=['metaRatings']), meta_df], axis=1)
        df.rename(columns={col: col.replace("meta.", "") for col in df.columns if col.startswith("meta.")}, inplace=True)
    
    if df.empty:
        return pd.DataFrame()

    # Rename attribute columns from 'attributeAcceleration' to 'acceleration'
    df.rename(columns={col: col.replace("attribute", "", 1)[0].lower() + col.replace("attribute", "", 1)[1:] 
                       for col in df.columns if col.startswith("attribute")}, inplace=True)

    df["__true_player_id"] = df["eaId"].astype(str)
    
    int_cols = ['overall', 'height', 'weight', 'skillMoves', 'weakFoot'] + attribute_filter_order
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    resp_attrs = ['acceleration', 'sprintSpeed', 'agility', 'balance', 'reactions']
    if all(attr in df.columns for attr in resp_attrs):
        df['responsiveness'] = df[resp_attrs].mean(axis=1)

    float_cols = ['ggMeta', 'esMeta', 'esMetaSub', 'avgMeta', 'avgMetaSub', 'responsiveness']
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    for col in ['roles+', 'roles++']:
        if col not in df.columns: df[col] = pd.Series([[] for _ in range(len(df))])
        df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])
        
    df['hasRolePlus'] = df.apply(lambda row: row.get('role') in row.get('roles+', []), axis=1)
    df['hasRolePlusPlus'] = df.apply(lambda row: row.get('role') in row.get('roles++', []), axis=1)

    return df

df = load_data(data_dir / "club_final.json")
all_players_df = load_all_players(data_dir / "all_players_summary.json")
if df.empty:
    st.stop()

if "role" in df.columns:
    unique_roles = df["role"].dropna().unique().tolist()
    sorted_roles = sort_by_reference(unique_roles, roles_order)
else:
    sorted_roles = []

position_options = unique_positions_from_df(df)

# --- Display Logic ---
st.title("El Mostashar FC - Club Player Database")
render_squad_builder(df, sorted_roles)
st.markdown("---")

# --- Sidebar filters ---
st.sidebar.header("Filter Players")

with st.sidebar.expander("MetaRating Weights", expanded=True):
    st.info("Adjust the influence of ES vs GG models on the Average.")
    es_weight = st.slider("ES Meta Weight", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
    gg_weight = st.slider("GG Meta Weight", min_value=0.0, max_value=5.0, value=1.0, step=0.1)

if es_weight + gg_weight > 0:
    # Update avgMeta dynamically!
    has_both = (df['esMeta'] > 0) & (df['ggMeta'] > 0)
    df.loc[has_both, 'avgMeta'] = ((df.loc[has_both, 'esMeta'] * es_weight) + (df.loc[has_both, 'ggMeta'] * gg_weight)) / (es_weight + gg_weight)
    df.loc[(df['esMeta'] > 0) & (df['ggMeta'] <= 0), 'avgMeta'] = df['esMeta']
    df.loc[(df['ggMeta'] > 0) & (df['esMeta'] <= 0), 'avgMeta'] = df['ggMeta']

    has_both_sub = (df['esMetaSub'] > 0) & (df['ggMetaSub'] > 0)
    df.loc[has_both_sub, 'avgMetaSub'] = ((df.loc[has_both_sub, 'esMetaSub'] * es_weight) + (df.loc[has_both_sub, 'ggMetaSub'] * gg_weight)) / (es_weight + gg_weight)
    df.loc[(df['esMetaSub'] > 0) & (df['ggMetaSub'] <= 0), 'avgMetaSub'] = df['esMetaSub']
    df.loc[(df['ggMetaSub'] > 0) & (df['esMetaSub'] <= 0), 'avgMetaSub'] = df['ggMetaSub']
else:
    df['avgMeta'] = 0.0
    df['avgMetaSub'] = 0.0

filters = {}

def create_min_max_filter(container, column_name, label, step, key_suffix=""):
    if column_name in df.columns and pd.api.types.is_numeric_dtype(df[column_name]):
        numeric_col = df[column_name].dropna()
        if not numeric_col.empty and numeric_col.min() != numeric_col.max():
            is_int = pd.api.types.is_integer_dtype(df[column_name])
            min_val, max_val = (int(numeric_col.min()), int(numeric_col.max())) if is_int else (float(numeric_col.min()), float(numeric_col.max()))
            fmt = "%d" if is_int else "%.1f"
            
            c1, c2 = container.columns(2)
            user_min = c1.number_input(f"Min {label}", value=min_val, min_value=min_val, max_value=max_val, step=step, format=fmt, key=f"{column_name}_min{key_suffix}")
            user_max = c2.number_input(f"Max {label}", value=max_val, min_value=min_val, max_value=max_val, step=step, format=fmt, key=f"{column_name}_max{key_suffix}")
            
            if user_min > min_val or user_max < max_val:
                 filters[column_name] = (user_min, user_max)

st.sidebar.selectbox("Evolution", options=["All", True, False], format_func=lambda x: "Evo Players" if x is True else "Standard Players" if x is False else "All", key="evolution_filter")
if st.session_state.get("evolution_filter") != "All": filters["evolution"] = st.session_state.evolution_filter

with st.sidebar.expander("Base Characteristics"):
    create_min_max_filter(st, "overall", "Overall", 1)
    create_min_max_filter(st, "skillMoves", "Skill Moves", 1)
    create_min_max_filter(st, "weakFoot", "Weak Foot", 1)
    create_min_max_filter(st, "height", "Height (cm)", 1)
    create_min_max_filter(st, "weight", "Weight (kg)", 1)

with st.sidebar.expander("Detailed Meta Ratings"):
    # Clear session state naturally by linking the slider key to dynamic weights so they reset on change
    key_suf = f"_{es_weight}_{gg_weight}"
    create_min_max_filter(st, "avgMeta", "Avg On-Chem Meta", 0.1, key_suffix=key_suf)
    create_min_max_filter(st, "avgMetaSub", "Avg Sub Meta", 0.1, key_suffix=key_suf)
    create_min_max_filter(st, "ggMeta", "GG Meta", 0.1)
    create_min_max_filter(st, "ggMetaSub", "GG Meta (Sub)", 0.1)
    create_min_max_filter(st, "esMeta", "ES Meta", 0.1)
    create_min_max_filter(st, "esMetaSub", "ES Meta (Sub)", 0.1)

if position_options:
    st.sidebar.multiselect("Position (Any)", position_options, key="position_filter")
    if st.session_state.get("position_filter"): filters["positions"] = st.session_state.position_filter

if "role" in df.columns:
    st.sidebar.multiselect("Role (Any)", sorted_roles, key="role_filter")
    if st.session_state.get("role_filter"): filters["role"] = st.session_state.role_filter

if "foot" in df.columns:
    st.sidebar.multiselect("Foot", sorted(df["foot"].dropna().unique()), key="foot_filter")
    if st.session_state.get("foot_filter"): filters["foot"] = st.session_state.foot_filter

if "bodyType" in df.columns:
    st.sidebar.multiselect("Body Type", sorted(df["bodyType"].dropna().unique()), key="bodytype_filter")
    if st.session_state.get("bodytype_filter"): filters["bodyType"] = st.session_state.bodytype_filter

all_ps = set(s for l in df['PS'].dropna() if isinstance(l, list) for s in l)
all_ps.update(s for l in df['PS+'].dropna() if isinstance(l, list) for s in l)
if all_ps:
    st.sidebar.multiselect("PlayStyles (All Selected)", sorted(list(all_ps)), key="playstyles_all_filter")
    if st.session_state.get("playstyles_all_filter"): filters["playstyles_all"] = st.session_state.playstyles_all_filter

all_ps_plus = set(s for l in df['PS+'].dropna() if isinstance(l, list) for s in l)
if all_ps_plus:
    st.sidebar.multiselect("PlayStyles+ (All)", sorted(list(all_ps_plus)), key="playstyles_plus_all_filter")
    if st.session_state.get("playstyles_plus_all_filter"): filters["playstyles_plus_all"] = st.session_state.playstyles_plus_all_filter

with st.sidebar.expander("Accelerate Types"):
    if "ggAccelType" in df.columns: 
        unique_gg_accel = sorted(df["ggAccelType"].dropna().unique())
        if unique_gg_accel:
            selected_gg_accel = st.multiselect("GG Chem Accelerate Type", unique_gg_accel)
            if selected_gg_accel:
                filters["ggAccelType"] = selected_gg_accel

    if "esAccelType" in df.columns: 
        unique_es_accel = sorted(df["esAccelType"].dropna().unique())
        if unique_es_accel:
            selected_es_accel = st.multiselect("ES Chem Accelerate Type", unique_es_accel)
            if selected_es_accel:
                filters["esAccelType"] = selected_es_accel

    if "subAccelType" in df.columns: 
        unique_sub_accel = sorted(df["subAccelType"].dropna().unique())
        if unique_sub_accel:
            selected_sub_accel = st.multiselect("Sub Accelerate Type", unique_sub_accel)
            if selected_sub_accel:
                filters["subAccelType"] = selected_sub_accel

    st.checkbox("Can be Lengthy?", key="canBeLengthy_checkbox")
    if st.session_state.get("canBeLengthy_checkbox"): filters["canBeLengthy"] = True

st.sidebar.checkbox("Tall Players Only (≥195cm)", key="isTall_checkbox")
if st.session_state.get("isTall_checkbox"): filters["isTall"] = True

with st.sidebar.expander("Role Familiarity"):
    st.checkbox("Has Role+", key="hasRolePlus_checkbox")
    if st.session_state.get("hasRolePlus_checkbox"): filters["hasRolePlus"] = True
    st.checkbox("Has Role++", key="hasRolePlusPlus_checkbox")
    if st.session_state.get("hasRolePlusPlus_checkbox"): filters["hasRolePlusPlus"] = True

with st.sidebar.expander("In-Game Stats"):
    create_min_max_filter(st, "responsiveness", "Responsiveness", 0.1)
    for attr in attribute_filter_order:
        create_min_max_filter(st, attr, attr.replace("_", " ").title(), 1)

# --- Apply filters ---
filtered_df = apply_filters_to_df(df, filters)

# --- Default Sort ---
default_sort_column = "avgMeta"
if default_sort_column in filtered_df.columns:
    filtered_df = filtered_df.sort_values(by=default_sort_column, ascending=False)

if filters:
    st.subheader("Active Filters:")
    filter_tags = []
    for key, f_val in filters.items():
        display_name = key.replace("_", " ").title()
        if key == "playstyles_all": display_name = "PlayStyles (All Selected)" 
        elif key == "roles+_any": display_name = "Roles+ (Any)"
        elif key == "roles++_any": display_name = "Roles++ (Any)"
        elif key == "positions": display_name = "Position"
        
        if isinstance(f_val, list):
            val_str = ", ".join(map(str,f_val))
        elif isinstance(f_val, tuple):
            val_str = f"{f_val[0]} - {f_val[1]}"
        else:
            val_str = str(f_val)
        filter_tags.append(f"<span style='background-color:#333;color:#f5f5f5;padding:3px 7px;margin:2px;border-radius:12px;display:inline-block;font-size:0.9em;'>{display_name}: {val_str}</span>")
    st.markdown(" ".join(filter_tags), unsafe_allow_html=True)
    st.markdown("---")


# --- TABS ---
tab1, tab2, tab3 = st.tabs(["Club Squad", "Sell Now 💰", "All Players 🌍"])

with tab1:
    best_role_only = st.checkbox("Show best role only", value=True, key="best_role_only",
                                 help="When checked, shows each player once at their highest-rated role. Uncheck to see all roles.")
    # Include players with overall >= 75 OR tall players (isTall == True)
    squad_slot_filter_active = bool(st.session_state.get("squad_active_slot") and filters.get("positions") and filters.get("role"))
    if squad_slot_filter_active:
        tab1_df = filtered_df.copy()
    else:
        is_tall_mask = filtered_df["isTall"].fillna(False) if "isTall" in filtered_df.columns else pd.Series(False, index=filtered_df.index)
        ovr_mask = filtered_df["overall"].fillna(0).astype(int) >= 75 if "overall" in filtered_df.columns else pd.Series(True, index=filtered_df.index)
        tab1_df = filtered_df[ovr_mask | is_tall_mask]

    if best_role_only and not tab1_df.empty:
        tab1_df = tab1_df.loc[tab1_df.groupby("__true_player_id")["avgMeta"].idxmax()]
        tab1_df = tab1_df.sort_values(by="avgMeta", ascending=False)

    st.subheader("Top Player Ratings")
    col1, col2 = st.columns(2)

    def display_top_metric(container, df_to_use, metric_col, title, n=5):
        if metric_col in df_to_use.columns and not df_to_use.empty:
            df_no_na = df_to_use[df_to_use[metric_col] > 0].dropna(subset=[metric_col])
            if not df_no_na.empty:
                top_players = df_no_na.loc[df_no_na.groupby("__true_player_id")[metric_col].idxmax()].nlargest(n, metric_col)
                with container:
                    st.markdown(f"#### {title}")
                    for i, (_, row) in enumerate(top_players.iterrows()):
                        rank_display = f"**{i+1}.**"
                        if i < 3:
                            rank_display = ["🥇", "🥈", "🥉"][i]

                        def norm_style(val):
                            if val is None or (isinstance(val, float) and pd.isna(val)):
                                return "N/A"
                            s = str(val).strip()
                            return "N/A" if not s else s.title()

                        es_style = norm_style(row.get("esChemStyle"))
                        gg_style = norm_style(row.get("ggChemStyle"))

                        if es_style == "N/A" and gg_style != "N/A":
                            chem_display = gg_style
                        elif gg_style == "N/A":
                            chem_display = es_style
                        elif es_style == gg_style:
                            chem_display = es_style
                        else:
                            chem_display = f"{es_style}/{gg_style}"

                        label = (
                            f"{rank_display} {row.get('commonName', 'N/A')} "
                            f"({row.get('role', 'N/A')} - {chem_display})"
                        )
                        st.metric(label=label, value=f"{row.get(metric_col, 0.0):.2f}")


    display_top_metric(col1, tab1_df, "avgMeta", "Top Average On-Chem Meta", n=5)
    display_top_metric(col2, tab1_df, "avgMetaSub", "Top Average Sub Meta", n=5)

    st.markdown("---")
    st.markdown(f"### Player List ({tab1_df['player_origin_id'].nunique()} unique players, {len(tab1_df)} total entries)")

    columns_to_display = [
        "commonName", "role", "overall", "responsiveness", "avgMeta", 
        "ggMeta", "ggChemStyle", "ggAccelType", 
        "esMeta", "esChemStyle", "esAccelType",
        "avgMetaSub", "ggMetaSub", "esMetaSub", "subAccelType", "canBeLengthy", "Chem points needed for Lengthy",
        "hasRolePlusPlus", "hasRolePlus", "isTall", "skillMoves", "weakFoot", "foot", "height", "weight", "bodyType",
        "PS+", "PS", "positions", "roles++", "roles+"
    ] + attribute_filter_order

    final_display_columns = [col for col in columns_to_display if col in df.columns]

    display_df = tab1_df.copy()
    for col in ["hasRolePlus", "hasRolePlusPlus", "canBeLengthy", "isTall"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: "✅" if x else "❌")

    display_df.drop(columns=[c for c in ["player_origin_id", "__true_player_id"] if c in display_df.columns], inplace=True, errors="ignore")

    if display_df.empty:
        st.warning("No players found matching the selected filters.")
    else:
        st.dataframe(display_df[final_display_columns], use_container_width=True, hide_index=True)

with tab2:
    st.header("Sell Now")
    
    import os
    import datetime
    prices_dir = data_dir / "raw" / "prices"
    if prices_dir.exists():
        latest_file = max(prices_dir.glob('*.json'), key=os.path.getmtime, default=None)
        if latest_file:
            last_mod_time = os.path.getmtime(latest_file)
            dt_str = datetime.datetime.fromtimestamp(last_mod_time).strftime('%Y-%m-%d %H:%M:%S')
            st.info(f"Prices Last Updated at: **{dt_str}**")

    import json
    tradeable_file = data_dir / "tradeable_details.json"
    tradeable_details = []
    if tradeable_file.exists():
        try:
            with open(tradeable_file, "r", encoding="utf-8") as f:
                tradeable_details = json.load(f)
        except Exception:
            pass

    if tradeable_details:
        trad_df = pd.DataFrame(tradeable_details)
        prices_dir = data_dir / "raw" / "prices"
        
        def load_price(ea_id):
            clean_id = str(ea_id).split(".")[0]
            pfile = prices_dir / f"{clean_id}.json"
            if pfile.exists():
                try:
                    with open(pfile, "r", encoding="utf-8") as pf:
                        data = json.load(pf)
                        return data.get("price", 0)
                except:
                    pass
            return 0
            
        trad_df["price"] = trad_df["__true_player_id"].apply(load_price)
        trad_df["price"] = pd.to_numeric(trad_df["price"], errors="coerce").fillna(0).astype(int)
        trad_df["discardValue"] = pd.to_numeric(trad_df.get("discardValue", 0), errors="coerce").fillna(0).astype(int)
        
        sell_df = trad_df.copy()
    else:
        sell_df = pd.DataFrame()
        
    # Deduplicate by player ID to show one line per card but keep count
    if not sell_df.empty:
        item_counts = sell_df['__true_player_id'].value_counts()
        sell_df = sell_df.drop_duplicates(subset=['__true_player_id']).copy()
        sell_df['Item Count'] = sell_df['__true_player_id'].map(item_counts)
        sell_df['Total Value'] = sell_df['price'] * sell_df['Item Count'] * 0.95
        sell_df['Total Quick Sell Value'] = sell_df['discardValue'] * sell_df['Item Count']
        
    # Add Rarity filter
    if not sell_df.empty and 'rarity' in sell_df.columns:
        unique_rarities = sorted([r for r in sell_df['rarity'].dropna().unique() if str(r).strip() != ""])
        if unique_rarities:
            selected_rarities = st.multiselect("Rarity", unique_rarities, key="sell_now_rarity")
            if selected_rarities:
                sell_df = sell_df[sell_df['rarity'].isin(selected_rarities)]
    
    # Calculate sums
    total_price = sell_df['Total Value'].sum() if not sell_df.empty and 'Total Value' in sell_df.columns else 0
    total_discard = sell_df['Total Quick Sell Value'].sum() if 'Total Quick Sell Value' in sell_df.columns else 0
    
    # Show totals
    col1, col2 = st.columns(2)
    col1.metric("Total Market Value", f"{int(total_price):,}")
    col2.metric("Total Discard Value", f"{int(total_discard):,}")
    
    st.markdown("---")

    if sell_df.empty:
        st.info("No tradeable players found.")
    else:
        # Sort by price descending initially
        sell_df = sell_df.sort_values(by='price', ascending=False)
        
        st.markdown(f"Found **{len(sell_df)}** tradeable players.")

        # Prepare columns for display
        sell_cols = ["commonName", "overall", "Item Count", "price", "Total Value", "isExtinct"]
        if 'Total Quick Sell Value' in sell_df.columns:
            sell_cols.insert(5, "discardValue")
            sell_cols.insert(6, "Total Quick Sell Value")
        if 'rarity' in sell_df.columns:
            sell_cols.insert(2, "rarity")
        
        sell_display = sell_df[sell_cols].copy()
        
        # Use native Streamlit column configuration to format numbers properly without breaking sorting
        st.dataframe(
            sell_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "commonName": st.column_config.TextColumn("Name", width="medium"),
                "price": st.column_config.NumberColumn("Unit Price", format="%d", help="Market Price for a single card"),
                "Item Count": st.column_config.NumberColumn("Count", format="%d", help="Number of tradeable duplicates in club"),
                "Total Value": st.column_config.NumberColumn("Total Value", format="%d", help="Total market value after 5% EA tax"),
                "discardValue": st.column_config.NumberColumn("Unit Quick Sell", format="%d", help="Quick Sell Value for a single card"),
                "Total Quick Sell Value": st.column_config.NumberColumn("Total Quick Sell", format="%d", help="Total Quick Sell Value")
            }
        )

with tab3:
    st.header("All Players Database")
    
    if all_players_df.empty:
        st.info("No all players data available. Run the build script.")
    else:
        best_role_only_all = st.checkbox("Show best role only", value=True, key="best_role_only_all",
                                         help="When checked, shows each player once at their highest-rated role. Uncheck to see all roles.")
        
        # Apply filters to all_players_df
        filtered_all_df = all_players_df.copy()
        
        # Adjust meta weights for all_players_df
        if es_weight + gg_weight > 0 and not filtered_all_df.empty:
            has_both = (filtered_all_df['esMeta'] > 0) & (filtered_all_df['ggMeta'] > 0)
            filtered_all_df.loc[has_both, 'avgMeta'] = ((filtered_all_df.loc[has_both, 'esMeta'] * es_weight) + (filtered_all_df.loc[has_both, 'ggMeta'] * gg_weight)) / (es_weight + gg_weight)
            filtered_all_df.loc[(filtered_all_df['esMeta'] > 0) & (filtered_all_df['ggMeta'] <= 0), 'avgMeta'] = filtered_all_df['esMeta']
            filtered_all_df.loc[(filtered_all_df['ggMeta'] > 0) & (filtered_all_df['esMeta'] <= 0), 'avgMeta'] = filtered_all_df['ggMeta']

            # We only have esMetaSub in all_players_df, no ggMetaSub without model
            filtered_all_df['avgMetaSub'] = filtered_all_df['esMetaSub']
            
        filtered_all_df = apply_filters_to_df(filtered_all_df, filters)
        
        if best_role_only_all and not filtered_all_df.empty:
            if "avgMeta" in filtered_all_df.columns:
                filtered_all_df = filtered_all_df.loc[filtered_all_df.groupby("__true_player_id")["avgMeta"].idxmax()]
                filtered_all_df = filtered_all_df.sort_values(by="avgMeta", ascending=False)
                
        st.markdown(f"### Player List ({filtered_all_df['eaId'].nunique()} unique players, {len(filtered_all_df)} total entries)")

        columns_to_display_all = [
            "commonName", "role", "overall", "responsiveness", "avgMeta", 
            "ggMeta", "ggChemStyle", "ggAccelType", 
            "esMeta", "esChemStyle", "esAccelType",
            "avgMetaSub", "esMetaSub", "subAccelType", 
            "hasRolePlusPlus", "hasRolePlus", "isTall", "skillMoves", "weakFoot", "foot", "height", "weight", "bodyType",
            "PS+", "PS", "positions", "roles++", "roles+"
        ] + attribute_filter_order

        final_display_columns_all = [col for col in columns_to_display_all if col in filtered_all_df.columns]

        display_all_df = filtered_all_df.copy()
        for col in ["hasRolePlus", "hasRolePlusPlus", "isTall"]:
            if col in display_all_df.columns:
                display_all_df[col] = display_all_df[col].apply(lambda x: "✅" if x else "❌")

        display_all_df.drop(columns=[c for c in ["__true_player_id"] if c in display_all_df.columns], inplace=True, errors="ignore")

        if display_all_df.empty:
            st.warning("No players found matching the selected filters.")
        else:
            st.dataframe(display_all_df[final_display_columns_all], use_container_width=True, hide_index=True)
