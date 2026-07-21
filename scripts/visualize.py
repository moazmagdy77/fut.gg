import streamlit as st
import pandas as pd
import json
import os
import datetime
from pathlib import Path
from shared_utils import calculate_acceleration_type

st.set_page_config(layout="wide")

# Keep the Squad Builder formation pitch as a 2D grid on mobile. Streamlit
# otherwise stacks st.columns vertically on narrow screens, which destroys the
# on-pitch formation shape. Scoped to the pitch container only (st-key-squad_pitch).
st.markdown(
    """
    <style>
    .st-key-squad_pitch [data-testid="stHorizontalBlock"] { flex-wrap: nowrap !important; }
    .st-key-squad_pitch [data-testid="stColumn"] { min-width: 0 !important; }
    .st-key-squad_pitch [data-testid="stColumn"] > div { min-width: 0 !important; }
    .st-key-squad_pitch button { padding-left: 2px !important; padding-right: 2px !important; }
    .st-key-squad_pitch button p {
        font-size: 0.72rem !important; line-height: 1.1 !important;
        white-space: normal !important; overflow-wrap: anywhere;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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

# Compact column set shown by default in the player tables ("Show all attributes" reveals the rest).
COMPACT_DISPLAY_COLUMNS = [
    "commonName", "role", "overall", "responsiveness", "avgMeta",
    "ggMeta", "esMeta", "avgMetaSub", "ggChemStyle", "esChemStyle",
    "hasRolePlusPlus", "hasRolePlus", "isTall",
]

# Shared header/help/formatting for the player tables. Columns not present are ignored by Streamlit.
COLUMN_HELP = {
    "commonName": st.column_config.TextColumn("Name", width="medium"),
    "responsiveness": st.column_config.NumberColumn(
        "Responsiveness", format="%.1f",
        help="Average of Acceleration, Sprint Speed, Agility, Balance and Reactions (base / 0-chem)."),
    "avgMeta": st.column_config.NumberColumn(
        "Avg Meta", format="%.2f", help="Weighted average of ES and GG on-chem meta ratings (weights in the sidebar)."),
    "avgMetaSub": st.column_config.NumberColumn(
        "Avg Meta (Sub)", format="%.2f", help="Average sub meta rating (0 chem, no chem style)."),
    "ggMeta": st.column_config.NumberColumn("GG Meta", format="%.2f", help="fut.gg meta rating at the best chem style."),
    "esMeta": st.column_config.NumberColumn("ES Meta", format="%.2f", help="EasySBC meta rating at the best chem style."),
    "ggMetaSub": st.column_config.NumberColumn("GG Meta (Sub)", format="%.2f", help="fut.gg meta rating at 0 chem."),
    "esMetaSub": st.column_config.NumberColumn("ES Meta (Sub)", format="%.2f", help="EasySBC meta rating at 0 chem."),
    "overall": st.column_config.NumberColumn("OVR", format="%d"),
}

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
    with st.popover("📋 Browse all formations", width='stretch'):
        for group_name, formation_names in FORMATION_GROUPS.items():
            st.markdown(f"#### {group_name}")
            for start in range(0, len(formation_names), 4):
                cols = st.columns(4)
                for col, formation_name in zip(cols, formation_names[start:start + 4]):
                    button_type = "primary" if st.session_state.get("squad_selected_formation") == formation_name else "secondary"
                    if col.button(formation_name, key=f"formation_button_{safe_widget_key(formation_name)}", type=button_type, width='stretch'):
                        st.session_state["squad_pending_formation"] = formation_name
                        st.rerun()

@st.dialog("Confirm Add to Squad")
def confirm_add_dialog(player_info, slot_id, position, role):
    st.write(f"Do you want to add **{player_info['commonName']}** (Rating: {player_info['avgMeta']:.1f}) to the squad as **{position}** ({role})?")
    c1, c2 = st.columns(2)
    if c1.button("Confirm", type="primary", width='stretch'):
        player_info["selected_role"] = role
        st.session_state["squad_players"][slot_id] = player_info
        clear_squad_builder_filters()
        # Reset selection keys
        if "club_players_df" in st.session_state:
            st.session_state["club_players_df"] = {"selection": {"rows": [], "columns": []}}
        if "all_players_df" in st.session_state:
            st.session_state["all_players_df"] = {"selection": {"rows": [], "columns": []}}
        st.rerun()
    if c2.button("Cancel", width='stretch'):
        clear_squad_builder_filters()
        if "club_players_df" in st.session_state:
            st.session_state["club_players_df"] = {"selection": {"rows": [], "columns": []}}
        if "all_players_df" in st.session_state:
            st.session_state["all_players_df"] = {"selection": {"rows": [], "columns": []}}
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

    with st.expander("⚽ Squad Builder", expanded=True):
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
            st.session_state["squad_players"] = {}
            st.session_state["custom_squad_roles"] = {}
            st.rerun()
        else:
            st.session_state["squad_selected_formation"] = selected_formation

        active_slot = st.session_state.get("squad_active_slot")
        squad_players = st.session_state.setdefault("squad_players", {})
        count_col.metric("Slots Remaining", 11 - len(squad_players))
        if clear_col.button("Cancel selection", width='stretch', help="Clear the active slot's position/role filter."):
            clear_squad_builder_filters()
            st.rerun()

        st.caption("① Pick a formation → ② click a slot below → ③ pick a player from the list, then confirm.")
        render_formation_grid()

        with st.container(border=True, key="squad_pitch"):
            st.markdown(f"### {selected_formation}")
            for row_index, row_positions in enumerate(FC26_FORMATIONS[selected_formation]):
                slot_columns = centered_columns(len(row_positions))
                row_labels = slot_labels_for_row(row_positions)
                for slot_index, (slot_col, position, label) in enumerate(zip(slot_columns, row_positions, row_labels)):
                    slot_id = f"{safe_widget_key(selected_formation)}_{row_index}_{slot_index}_{label}_{position}"
                    role_key = f"squad_role_{slot_id}"
                    role_options = role_options_for_position(position, available_roles)
                    default_role = default_role_for_position(position, available_roles)
                
                    custom_roles = st.session_state.setdefault("custom_squad_roles", {})
                    if slot_id not in custom_roles or custom_roles[slot_id] not in role_options:
                        custom_roles[slot_id] = default_role
                
                    if role_key not in st.session_state or st.session_state[role_key] not in role_options:
                        st.session_state[role_key] = custom_roles[slot_id]

                    with slot_col:
                        selected_role = custom_roles[slot_id]
                        assigned_player = squad_players.get(slot_id)
                    
                        if assigned_player:
                            p_role = assigned_player.get('selected_role') or assigned_player.get('role', '')
                            button_label = f"{label}: {assigned_player['commonName']} ({assigned_player['avgMeta']:.1f} - {p_role})"
                        else:
                            button_label = label
                        button_type = "primary" if active_slot == slot_id else "secondary"
                    
                        if st.button(button_label, key=f"squad_slot_{slot_id}", type=button_type, width='stretch'):
                            apply_squad_slot_filter(slot_id, position, selected_role)
                            st.rerun()

                        if assigned_player:
                            if st.button("❌ Remove", key=f"remove_squad_slot_{slot_id}", type="secondary", width='stretch'):
                                del st.session_state["squad_players"][slot_id]
                                if active_slot == slot_id:
                                    clear_squad_builder_filters()
                                st.rerun()
                        else:
                            selected_role = st.selectbox(
                                f"{label} role",
                                role_options,
                                key=role_key,
                                label_visibility="collapsed",
                            )
                            custom_roles[slot_id] = selected_role

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
    # No initial .copy(): every branch reassigns via a boolean mask (which returns a
    # new frame) and callers copy before mutating, so source_df is never modified.
    filtered = source_df
    special_cols = ("playstyles_all", "playstyles_plus_all", "commonName_contains")
    for col, val in active_filters.items():
        if val is None or (isinstance(val, list) and not val):
            continue
        if col not in filtered.columns and col not in special_cols:
            continue

        if col == "commonName_contains":
            filtered = filtered[filtered["commonName"].fillna("").astype(str).str.contains(str(val), case=False, na=False)]
        elif col == "playstyles_all":
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

def recompute_avg_meta(target_df, es_weight, gg_weight):
    """Recompute avgMeta / avgMetaSub from the ES/GG weights, in place.

    On-chem uses esMeta+ggMeta. Sub uses esMetaSub+ggMetaSub when a ggMetaSub
    column exists (club_final); otherwise it falls back to esMetaSub alone
    (all_players_summary has no modeled ggMetaSub yet)."""
    total = es_weight + gg_weight
    if total <= 0 or target_df.empty:
        return target_df

    if 'esMeta' in target_df.columns and 'ggMeta' in target_df.columns:
        both = (target_df['esMeta'] > 0) & (target_df['ggMeta'] > 0)
        target_df.loc[both, 'avgMeta'] = (
            target_df.loc[both, 'esMeta'] * es_weight + target_df.loc[both, 'ggMeta'] * gg_weight
        ) / total
        target_df.loc[(target_df['esMeta'] > 0) & (target_df['ggMeta'] <= 0), 'avgMeta'] = target_df['esMeta']
        target_df.loc[(target_df['ggMeta'] > 0) & (target_df['esMeta'] <= 0), 'avgMeta'] = target_df['ggMeta']

    if 'esMetaSub' in target_df.columns and 'ggMetaSub' in target_df.columns:
        both_sub = (target_df['esMetaSub'] > 0) & (target_df['ggMetaSub'] > 0)
        target_df.loc[both_sub, 'avgMetaSub'] = (
            target_df.loc[both_sub, 'esMetaSub'] * es_weight + target_df.loc[both_sub, 'ggMetaSub'] * gg_weight
        ) / total
        target_df.loc[(target_df['esMetaSub'] > 0) & (target_df['ggMetaSub'] <= 0), 'avgMetaSub'] = target_df['esMetaSub']
        target_df.loc[(target_df['ggMetaSub'] > 0) & (target_df['esMetaSub'] <= 0), 'avgMetaSub'] = target_df['ggMetaSub']
    elif 'esMetaSub' in target_df.columns:
        target_df['avgMetaSub'] = target_df['esMetaSub']
    return target_df

def _normalize_players_df(df):
    """Shared normalization for both club_final and all_players_summary frames.

    Explodes the per-role `metaRatings` list, flattens attribute/meta columns,
    coerces dtypes, and computes role-familiarity flags. Callers add their own
    extras (club: player_origin_id + lengthy info). The float/int coercion lists
    are supersets — missing columns are simply skipped — so one helper serves both
    schemas without behavior change."""
    if 'metaRatings' in df.columns:
        df = df.explode('metaRatings').reset_index(drop=True)
        df = df[df['metaRatings'].notna()].reset_index(drop=True)
        meta_df = pd.json_normalize(df['metaRatings']).add_prefix('meta.')
        df = pd.concat([df.drop(columns=['metaRatings']), meta_df], axis=1)
        df.rename(columns={col: col.replace("meta.", "") for col in df.columns if col.startswith("meta.")}, inplace=True)

    if df.empty:
        return df

    # attributeAcceleration -> acceleration, etc.
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
    else:
        df['responsiveness'] = 0.0

    float_cols = ['ggMeta', 'ggMetaSub', 'esMeta', 'esMetaSub', 'avgMeta', 'avgMetaSub', 'price', 'responsiveness', 'discardValue']
    for col in float_cols:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace(',', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    for col in ['roles+', 'roles++']:
        if col not in df.columns:
            df[col] = pd.Series([[] for _ in range(len(df))])
        df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])

    df['hasRolePlus'] = df.apply(lambda row: row.get('role') in row.get('roles+', []), axis=1)
    df['hasRolePlusPlus'] = df.apply(lambda row: row.get('role') in row.get('roles++', []), axis=1)
    return df

@st.cache_data
def load_data(file_path, mtime):
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

    df = _normalize_players_df(pd.json_normalize(data, errors='ignore'))
    if df.empty:
        st.warning("No data loaded or normalized from club_final.json.")
        return pd.DataFrame()

    df['player_origin_id'] = df['eaId'].astype(str) + '_' + df['evolution'].astype(str)

    def compute_lengthy(accel, agility, strength, height, gender):
        # Delegate to the canonical AcceleRATE logic in shared_utils so the viewer
        # can't drift from the pipeline. Lengthy and Explosive are mutually
        # exclusive, so this equals a standalone lengthy-only check.
        gender = gender if pd.notna(gender) else "Male"
        return calculate_acceleration_type(accel, agility, strength, height, gender) == "LENGTHY"

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

@st.cache_data
def load_all_players(file_path, mtime):
    """Loads the pre-built all players summary JSON."""
    try:
        with open(file_path, "r", encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        st.warning(f"All players data not found. Run `build_all_players_summary.py` to generate it.")
        return pd.DataFrame()
    
    return _normalize_players_df(pd.json_normalize(data, errors='ignore'))

if "squad_players" not in st.session_state:
    st.session_state["squad_players"] = {}

club_file = data_dir / "club_final.json"
all_players_file = data_dir / "all_players_summary.json"

club_mtime = os.path.getmtime(club_file) if club_file.exists() else 0
all_players_mtime = os.path.getmtime(all_players_file) if all_players_file.exists() else 0

fodder_prices_file = data_dir / "fodder_prices.json"
club_analyzer_file = data_dir / "club-analyzer.html"
fodder_prices_mtime = os.path.getmtime(fodder_prices_file) if fodder_prices_file.exists() else 0
club_analyzer_mtime = os.path.getmtime(club_analyzer_file) if club_analyzer_file.exists() else 0

df = load_data(club_file, club_mtime).copy()

# Exclude players already in the starting squad. all_players_df is loaded lazily
# inside the All Players view so its ~45k-row frame isn't copied on every rerun.
assigned_ea_ids = {str(p["eaId"]) for p in st.session_state["squad_players"].values()}
if assigned_ea_ids and not df.empty:
    df = df[~df["__true_player_id"].astype(str).isin(assigned_ea_ids)]

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

if st.sidebar.button("Clear Cache & Reload Data", width='stretch'):
    st.cache_data.clear()
    st.rerun()

st.sidebar.text_input("🔎 Search player by name", key="name_search", placeholder="e.g. Mbappé")

with st.sidebar.expander("MetaRating Weights", expanded=True):
    st.info("Adjust the influence of ES vs GG models on the Average.")
    es_weight = st.slider("ES Meta Weight", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
    gg_weight = st.slider("GG Meta Weight", min_value=0.0, max_value=5.0, value=1.0, step=0.1)

# Guard the degenerate case: both weights at 0 would zero out every rating and
# silently break sorting / "best role". Fall back to equal weighting with a notice.
if es_weight + gg_weight <= 0:
    st.sidebar.warning("Both meta weights are 0 — falling back to equal weighting.")
    es_weight = gg_weight = 1.0

# Update avgMeta / avgMetaSub dynamically from the ES/GG weights
recompute_avg_meta(df, es_weight, gg_weight)

filters = {}

# Player name search (rendered at the top of the sidebar; applied here)
_name_query = st.session_state.get("name_search", "")
if _name_query and _name_query.strip():
    filters["commonName_contains"] = _name_query.strip()

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
    # Render Min/Max inputs only for attributes the user picks, instead of
    # instantiating ~70 number_inputs (2 per attribute) on every rerun.
    attr_choices = [a for a in attribute_filter_order if a in df.columns]
    picked_attrs = st.multiselect(
        "Filter by attribute", attr_choices,
        format_func=lambda a: a.replace("_", " ").title(),
        key="attr_filter_picker",
        help="Pick attributes to add Min/Max range filters for.",
    )
    for attr in picked_attrs:
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
        elif key == "commonName_contains": display_name = "Name contains"

        if isinstance(f_val, list):
            val_str = ", ".join(map(str,f_val))
        elif isinstance(f_val, tuple):
            val_str = f"{f_val[0]} - {f_val[1]}"
        else:
            val_str = str(f_val)
        filter_tags.append(f"<span style='background-color:#333;color:#f5f5f5;padding:3px 7px;margin:2px;border-radius:12px;display:inline-block;font-size:0.9em;'>{display_name}: {val_str}</span>")
    st.markdown(" ".join(filter_tags), unsafe_allow_html=True)
    st.markdown("---")


def render_player_table(view_df, columns_full, bool_cols, table_key, show_all_key):
    """Render a selectable player table shared by the Club and All Players views:
    a compact/full column toggle, ✅/❌ for boolean flags, single-row selection,
    and (when a squad slot is active) the confirm-add dialog. `view_df` must be the
    same frame the row order comes from so the selection index maps back correctly."""
    show_all = st.checkbox("Show all attributes", value=False, key=show_all_key,
                           help="Reveal every attribute column. Off shows a compact set.")
    cols = columns_full if show_all else COMPACT_DISPLAY_COLUMNS
    final_cols = [c for c in cols if c in view_df.columns]

    display = view_df.copy()
    for col in bool_cols:
        if col in display.columns:
            display[col] = display[col].apply(lambda x: "✅" if x else "❌")
    display.drop(columns=[c for c in ["player_origin_id", "__true_player_id"] if c in display.columns],
                 inplace=True, errors="ignore")

    if display.empty:
        st.warning("No players found matching the selected filters.")
        return

    event = st.dataframe(
        display[final_cols],
        width='stretch', hide_index=True,
        on_select="rerun", selection_mode="single-row",
        key=table_key, column_config=COLUMN_HELP,
    )
    active_slot = st.session_state.get("squad_active_slot")
    if active_slot and event.selection.rows:
        sel = view_df.iloc[event.selection.rows[0]]
        player_info = {
            "eaId": str(sel["__true_player_id"]) if "__true_player_id" in sel else str(sel["eaId"]),
            "commonName": sel["commonName"],
            "avgMeta": sel["avgMeta"],
            "overall": sel["overall"],
            "role": sel["role"],
            "position": sel["positions"] if isinstance(sel["positions"], list) else [sel["positions"]],
        }
        confirm_add_dialog(player_info, active_slot,
                           st.session_state.get("squad_applied_position"),
                           st.session_state.get("squad_applied_role"))


# --- Fodder Value helpers ---
GOLD_COMMON_PRICE = 350  # 75+ commons — the overview API starts at 81; near-floor & stable
GOLD_RARE_PRICE = 600    # 75–80 rares


@st.cache_data
def load_fodder_prices(file_path, mtime):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return (json.load(f) or {}).get("byRating", {})
    except Exception:
        return {}


@st.cache_data
def load_fodder_source(html_path, mtime):
    """Parse club-analyzer.html into per-card rows (rating/location/untradeable/rarity/
    discard) for the Fodder valuation. Mirrors club.1.get.ids.py's regex parse."""
    import re
    try:
        txt = Path(html_path).read_text(encoding="utf-8")
    except Exception:
        return pd.DataFrame()
    rows = re.findall(r'<tr>(.*?)</tr>', txt, re.DOTALL)
    if not rows:
        return pd.DataFrame()
    headers = [re.sub(r'<.*?>', '', h).strip() for h in re.findall(r'<th.*?>(.*?)</th>', rows[0], re.DOTALL)]

    def idx(name):
        try:
            return headers.index(name)
        except ValueError:
            return -1

    i_r, i_loc, i_unt, i_rar, i_disc = idx("Rating"), idx("Location"), idx("Untradeable"), idx("Rarity"), idx("Discard Value")
    recs = []
    for row in rows[1:]:
        cols = [re.sub(r'<.*?>', '', c).strip() for c in re.findall(r'<td.*?>(.*?)</td>', row, re.DOTALL)]
        if not cols or not (0 <= i_loc < len(cols)):
            continue
        try:
            rating = int(cols[i_r]) if 0 <= i_r < len(cols) else 0
        except ValueError:
            rating = 0
        disc = 0
        if 0 <= i_disc < len(cols):
            dt = cols[i_disc].replace(",", "").replace(".", "")
            if dt.isdigit():
                disc = int(dt)
        recs.append({
            "rating": rating,
            "location": cols[i_loc],
            "untradeable": (cols[i_unt].lower() == "true") if 0 <= i_unt < len(cols) else False,
            "rarity": cols[i_rar] if 0 <= i_rar < len(cols) else "",
            "discard": disc,
        })
    return pd.DataFrame(recs)


def build_fodder_table(src, prices, include_starting=True):
    """Reproduce the fodder valuation: per rating/tier, Value (cheapest BIN) × Count of
    owned cards. Club = untradeable-in-club, SS = SBC storage (per the source sheet)."""
    required = {"location", "untradeable", "rating", "rarity", "discard"}
    if src.empty or not required.issubset(src.columns):
        return pd.DataFrame()

    club = src[(src["location"] == "CLUB") & (src["untradeable"])]
    ss = src[src["location"] == "SBCSTORAGE"]
    rows = []

    # Bronze / Silver: tradeable club cards valued at their quick-sell (discard) total.
    tc = src[(src["location"] == "CLUB") & (~src["untradeable"]) & (src["discard"] > 0)]
    rows.append({"Rating": "Bronze (Tradeable)", "Value": None, "Count": None, "Club": None, "SS": None,
                 "Starting": None, "Total Value": int(tc[tc["rating"] < 65]["discard"].sum())})
    rows.append({"Rating": "Silvers (Tradeable)", "Value": None, "Count": None, "Club": None, "SS": None,
                 "Starting": None, "Total Value": int(tc[(tc["rating"] >= 65) & (tc["rating"] < 75)]["discard"].sum())})

    def add(label, value, club_mask, ss_mask):
        c = int(club_mask.sum())
        s = int(ss_mask.sum())
        starting = 0  # not tracked in the club export
        billable = (c + s) if include_starting else (c + s - starting)
        total = int(round((value or 0) * billable))
        rows.append({"Rating": label, "Value": value, "Count": c + s, "Club": c, "SS": s, "Starting": starting, "Total Value": total})

    add("Gold Commons", GOLD_COMMON_PRICE,
        (club["rating"] >= 75) & (club["rarity"] == "Common"), (ss["rating"] >= 75) & (ss["rarity"] == "Common"))
    add("Gold Rares", GOLD_RARE_PRICE,
        (club["rating"] >= 75) & (club["rating"] < 81) & (club["rarity"] == "Rare"),
        (ss["rating"] >= 75) & (ss["rating"] < 81) & (ss["rarity"] == "Rare"))
    for r in range(81, 100):
        val = prices.get(str(r))
        if r in (81, 82):  # only rares exist as fodder at these ratings (per the sheet)
            cm = (club["rating"] == r) & (club["rarity"] == "Rare")
            sm = (ss["rating"] == r) & (ss["rarity"] == "Rare")
        else:
            cm = (club["rating"] == r)
            sm = (ss["rating"] == r)
        add(str(r), val, cm, sm)
    return pd.DataFrame(rows)


# --- VIEWS ---
# A segmented control instead of st.tabs: st.tabs executes ALL tab bodies every
# rerun (incl. the heavy ~45k-row All Players processing). This runs only the
# selected view, so switching/filtering stays fast.
VIEW_CLUB, VIEW_SELL, VIEW_ALL, VIEW_FODDER = "Club Squad", "Sell Now 💰", "All Players 🌍", "Fodder Value 🍞"
active_view = st.segmented_control(
    "View", [VIEW_CLUB, VIEW_SELL, VIEW_ALL, VIEW_FODDER],
    default=VIEW_CLUB, key="active_view", label_visibility="collapsed",
)
if active_view is None:
    active_view = VIEW_CLUB

if active_view == VIEW_CLUB:
    if st.session_state.get("squad_active_slot"):
        _ap = st.session_state.get("squad_applied_position"); _ar = st.session_state.get("squad_applied_role")
        st.info(f"🎯 Filling **{_ap}** ({_ar}) — select a player row below to add them (or use 'Cancel selection' in the Squad Builder).")
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

    render_player_table(
        tab1_df, columns_to_display,
        ["hasRolePlus", "hasRolePlusPlus", "canBeLengthy", "isTall"],
        "club_players_df", "show_all_cols_club",
    )

elif active_view == VIEW_SELL:
    st.header("Sell Now")

    prices_dir = data_dir / "raw" / "prices"
    if prices_dir.exists():
        latest_file = max(prices_dir.glob('*.json'), key=os.path.getmtime, default=None)
        if latest_file:
            last_mod_time = os.path.getmtime(latest_file)
            dt_str = datetime.datetime.fromtimestamp(last_mod_time).strftime('%Y-%m-%d %H:%M:%S')
            st.info(f"Prices Last Updated at: **{dt_str}**")

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

        def load_price(ea_id):
            clean_id = str(ea_id).split(".")[0]
            pfile = prices_dir / f"{clean_id}.json"
            if pfile.exists():
                try:
                    with open(pfile, "r", encoding="utf-8") as pf:
                        data = json.load(pf)
                        return data.get("price", 0)
                except Exception:
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
            width='stretch',
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

elif active_view == VIEW_ALL:
    st.header("All Players Database")

    # Loaded lazily here (not at module top) so its ~45k-row frame is only built
    # and copied when this view is actually open.
    all_players_df = load_all_players(all_players_file, all_players_mtime)
    if assigned_ea_ids and not all_players_df.empty:
        all_players_df = all_players_df[~all_players_df["__true_player_id"].astype(str).isin(assigned_ea_ids)]
    
    if all_players_df.empty:
        st.info("No all players data available. Run the build script.")
    else:
        best_role_only_all = st.checkbox("Show best role only", value=True, key="best_role_only_all",
                                         help="When checked, shows each player once at their highest-rated role. Uncheck to see all roles.")
        
        # Apply filters to all_players_df
        filtered_all_df = all_players_df.copy()
        
        # Adjust meta weights for all_players_df (helper falls back to esMetaSub
        # for the Sub column until build_all_players_summary provides ggMetaSub).
        recompute_avg_meta(filtered_all_df, es_weight, gg_weight)
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
            "avgMetaSub", "ggMetaSub", "esMetaSub", "subAccelType",
            "hasRolePlusPlus", "hasRolePlus", "isTall", "skillMoves", "weakFoot", "foot", "height", "weight", "bodyType",
            "PS+", "PS", "positions", "roles++", "roles+"
        ] + attribute_filter_order

        render_player_table(
            filtered_all_df, columns_to_display_all,
            ["hasRolePlus", "hasRolePlusPlus", "isTall"],
            "all_players_df", "show_all_cols_all",
        )

elif active_view == VIEW_FODDER:
    st.header("Fodder Value 🍞")
    prices = load_fodder_prices(fodder_prices_file, fodder_prices_mtime)
    src = load_fodder_source(club_analyzer_file, club_analyzer_mtime)
    if src.empty:
        st.info("No `club-analyzer.html` found — can't value fodder. Run the club pipeline first.")
    elif not prices:
        st.warning("No fodder prices yet. Run `node scripts/fetch.fodder.prices.js` (or the pipeline's Fodder step).")
    else:
        if fodder_prices_mtime:
            st.caption(f"Prices as of {datetime.datetime.fromtimestamp(fodder_prices_mtime).strftime('%Y-%m-%d %H:%M')}")
        include_starting = st.toggle(
            "Include Starting?", value=True,
            help="Starting-XI cards aren't tracked in the club export, so Starting = 0 and this is currently informational.",
        )
        ft = build_fodder_table(src, prices, include_starting)
        if ft.empty:
            st.warning("Could not compute fodder counts from club-analyzer.html.")
        else:
            grand_count = int(ft["Count"].fillna(0).sum())
            grand_total = int(ft["Total Value"].fillna(0).sum())
            c1, c2 = st.columns(2)
            c1.metric("Total Fodder Value", f"{grand_total:,}")
            c2.metric("Cards Counted", f"{grand_count:,}")
            st.caption("Value = cheapest BIN per rating (fut.gg). **Club** = untradeable, in club · **SS** = SBC storage · "
                       "Bronze/Silver = quick-sell total of tradeable club cards.")
            st.dataframe(
                ft, width='stretch', hide_index=True,
                column_config={
                    "Rating": st.column_config.TextColumn("Rating / Tier", width="medium"),
                    "Value": st.column_config.NumberColumn("Value", format="%d", help="Cheapest BIN at this rating"),
                    "Count": st.column_config.NumberColumn("Count", format="%d", help="Club + SS"),
                    "Club": st.column_config.NumberColumn("Club", format="%d", help="Untradeable, in club"),
                    "SS": st.column_config.NumberColumn("SS", format="%d", help="In SBC storage"),
                    "Starting": st.column_config.NumberColumn("Starting", format="%d"),
                    "Total Value": st.column_config.NumberColumn("Total Value", format="%d", help="Value × Count"),
                },
            )
