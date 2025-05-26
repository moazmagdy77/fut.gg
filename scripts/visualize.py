import streamlit as st
import pandas as pd
import json
from pathlib import Path

st.set_page_config(layout="wide")

# Initialize session state for the clear filters flag if it doesn't exist
if "clear_filters_button_clicked" not in st.session_state:
    st.session_state.clear_filters_button_clicked = False

# Define data directory
data_dir = Path(__file__).resolve().parents[1] / "data"

attribute_filter_order = [
    "acceleration", "sprintSpeed", "positioning", "finishing", "shotPower", "longShots",
    "volleys", "penalties", "vision", "crossing", "fkAccuracy", "shortPassing",
    "longPassing", "curve", "agility", "balance", "reactions", "ballControl",
    "dribbling", "composure", "interceptions", "headingAccuracy", "defensiveAwareness",
    "standingTackle", "slidingTackle", "jumping", "stamina", "strength", "aggression",
    "gkDiving", "gkHandling", "gkKicking", "gkPositioning", "gkReflexes"
]

# Load and normalize JSON
try:
    with open(data_dir / "club_final.json", "r") as f:
        data = json.load(f)
except FileNotFoundError:
    st.error(f"Error: `club_final.json` not found in the '{data_dir}' directory. Please ensure the Python processing script has run successfully.")
    st.stop()
except json.JSONDecodeError:
    st.error(f"Error: `club_final.json` is not a valid JSON file. Please check the file content.")
    st.stop()


for idx, p in enumerate(data):
    p["player_origin_id"] = f"{p.get('eaId', 'unknown')}_{idx}"
    p["debug_index"] = idx

df = pd.json_normalize(data) # type: ignore

if df.empty:
    st.warning("No data loaded from club_final.json. The file might be empty or structured incorrectly.")
    st.stop()

if 'commonName' in df.columns:
    df.dropna(subset=['commonName'], inplace=True)
    df = df[df['commonName'].astype(str).str.strip() != '']
else:
    st.warning("`commonName` column not found in the data. Cannot filter out empty names.")


if df.empty:
    st.warning("No players with valid common names found in the data.")
    st.stop()

df["player_origin_id"] = df["player_origin_id"].astype(str)
df["eaId"] = df["eaId"].astype(str)
df["__true_player_id"] = df["eaId"].fillna(df["commonName"])

if "evolution" not in df.columns:
    df["evolution"] = False

df.rename(columns={col: col.replace("attribute", "", 1)[0].lower() + col.replace("attribute", "", 1)[1:]
                   for col in df.columns if col.startswith("attribute")}, inplace=True)

meta_ratings_col_name = "metaRatings"
if meta_ratings_col_name in df.columns:
    df[meta_ratings_col_name] = df[meta_ratings_col_name].apply(lambda x: x if isinstance(x, list) else ([{}] if not x else [x]))
    df = df.explode(meta_ratings_col_name, ignore_index=True)
    df[meta_ratings_col_name] = df[meta_ratings_col_name].apply(lambda x: x if isinstance(x, dict) else {})

    df["role"] = df[meta_ratings_col_name].apply(lambda x: x.get("role", "N/A"))
    df["esMetaSub"] = pd.to_numeric(df[meta_ratings_col_name].apply(lambda x: x.get("esMetaSub")), errors='coerce').fillna(0)
    df["esMeta"] = pd.to_numeric(df[meta_ratings_col_name].apply(lambda x: x.get("esMeta")), errors='coerce').fillna(0)
    df["esChemStyle"] = df[meta_ratings_col_name].apply(lambda x: x.get("esChemStyle", "None"))
    df["esAccelType"] = df[meta_ratings_col_name].apply(lambda x: x.get("esAccelType", "Unknown"))
    df["ggMeta"] = pd.to_numeric(df[meta_ratings_col_name].apply(lambda x: x.get("ggMeta")), errors='coerce').fillna(0)
    df["ggChemStyle"] = df[meta_ratings_col_name].apply(lambda x: x.get("ggChemStyle", "None"))
    df["ggAccelType"] = df[meta_ratings_col_name].apply(lambda x: x.get("ggAccelType", "Unknown"))
    df["subAccelType"] = df[meta_ratings_col_name].apply(lambda x: x.get("subAccelType", "CONTROLLED"))

    df = df.drop(columns=[meta_ratings_col_name])
else:
    placeholder_cols = ["role", "ggMeta", "ggChemStyle", "ggAccelType",
                         "esMeta", "esChemStyle", "esAccelType", "esMetaSub", "subAccelType"]
    for col in placeholder_cols:
        if col not in df.columns:
            if "Meta" in col or "Rank" in col : df[col] = 0.0
            elif "ChemStyle" in col : df[col] = "None"
            elif "AccelType" in col or "subAccelType" in col : df[col] = "Unknown"
            else: df[col] = "N/A"

df["height"] = pd.to_numeric(df["height"], errors='coerce').fillna(df["height"].dropna().astype(float).mean() if not df["height"].dropna().empty else 0).astype(int)
df["weight"] = pd.to_numeric(df["weight"], errors='coerce').fillna(df["weight"].dropna().astype(float).mean() if not df["weight"].dropna().empty else 0).astype(int)
df["overall"] = pd.to_numeric(df["overall"], errors='coerce').fillna(0).astype(int)
df["skillMoves"] = pd.to_numeric(df["skillMoves"], errors='coerce').fillna(0).astype(int)
df["weakFoot"] = pd.to_numeric(df["weakFoot"], errors='coerce').fillna(0).astype(int)

for attr_col_name in attribute_filter_order:
    if attr_col_name in df.columns:
        df[attr_col_name] = pd.to_numeric(df[attr_col_name], errors='coerce').fillna(0).astype(int)

df["__player_id"] = df["player_origin_id"] + "_" + df["role"].astype(str)

def recompute_role_flags(row):
    current_role = row.get("role", "N/A")
    player_roles_plus = row.get("roles+", [])
    player_roles_plus_plus = row.get("roles++", [])

    if not isinstance(player_roles_plus, list): player_roles_plus = []
    if not isinstance(player_roles_plus_plus, list): player_roles_plus_plus = []

    has_role_plus = current_role in player_roles_plus
    has_role_plus_plus = current_role in player_roles_plus_plus
    return pd.Series({"hasRolePlus": has_role_plus, "hasRolePlusPlus": has_role_plus_plus})

if "roles+" in df.columns and "roles++" in df.columns:
    role_flags = df.apply(recompute_role_flags, axis=1)
    df["hasRolePlus"] = role_flags["hasRolePlus"]
    df["hasRolePlusPlus"] = role_flags["hasRolePlusPlus"]
else:
    df["hasRolePlus"] = False
    df["hasRolePlusPlus"] = False


# --- Sidebar filters ---
st.sidebar.header("Filter Players")

# Clear All Filters Button and Logic
if st.sidebar.button("🧹 Clear All Filters", key="clear_filters_main_button"):
    st.session_state.clear_filters_button_clicked = True
    st.rerun()

if st.session_state.get("clear_filters_button_clicked", False):
    # Reset Selectbox (Evolution)
    if "evolution_filter" in st.session_state: # Ensure key exists before trying to set
        st.session_state.evolution_filter = "All"

    # Reset Min/Max filters (General Numeric and In-Game Stats)
    min_max_cols_to_reset = ["overall", "height", "weight", "skillMoves", "weakFoot",
                             "ggMeta", "esMetaSub", "esMeta"] + \
                            [attr for attr in attribute_filter_order if attr in df.columns]

    for col_name in min_max_cols_to_reset:
        key_prefix = col_name.lower().replace(" ", "_").replace("+", "plus").replace("-","").replace("(","").replace(")","")
        min_key = f"{key_prefix}_min"
        max_key = f"{key_prefix}_max"

        if col_name in df.columns and not df[col_name].isnull().all():
            numeric_col = pd.to_numeric(df[col_name], errors='coerce').dropna()
            if not numeric_col.empty:
                min_val_data = numeric_col.min()
                max_val_data = numeric_col.max()
                is_float_heuristic = df[col_name].dtype == 'float' or "meta" in col_name.lower() # Adjusted heuristic

                default_min = float(min_val_data) if is_float_heuristic else int(round(min_val_data))
                default_max = float(max_val_data) if is_float_heuristic else int(round(max_val_data))

                if min_key in st.session_state: st.session_state[min_key] = default_min
                if max_key in st.session_state: st.session_state[max_key] = default_max
            # else: # Column is all NaN after coerce, or empty
            #     if min_key in st.session_state: st.session_state[min_key] = 0 # Or some other safe default
            #     if max_key in st.session_state: st.session_state[max_key] = 100 # Or some other safe default

    # Reset Multiselects
    multiselect_keys_to_reset = [
        "role_filter", "roles+_any_filter", "roles++_any_filter",
        "playstyles_all_filter", "ggAccelType_filter",
        "esAccelType_filter", "subAccelType_filter"
    ]
    for ms_key in multiselect_keys_to_reset:
        if ms_key in st.session_state:
            st.session_state[ms_key] = []

    # Reset Checkboxes
    checkbox_keys_to_reset = ["hasRolePlus_checkbox", "hasRolePlusPlus_checkbox"]
    for cb_key in checkbox_keys_to_reset:
        if cb_key in st.session_state:
            st.session_state[cb_key] = False

    st.session_state.clear_filters_button_clicked = False # Reset the flag


filters = {}

if "evolution" in df.columns:
    unique_evo_vals = sorted(list(df["evolution"].unique()))
    options = ["All"] + unique_evo_vals
    # Use st.session_state.evolution_filter to make it responsive to the clear button
    st.sidebar.selectbox("Evolution", options=options, index=0, key="evolution_filter")
    if "evolution_filter" in st.session_state and st.session_state.evolution_filter != "All":
        filters["evolution"] = st.session_state.evolution_filter


def create_min_max_filter(container, column_name, label, default_step=1, default_format_str="%d"):
    if column_name in df.columns and not df[column_name].isnull().all():
        numeric_col = pd.to_numeric(df[column_name], errors='coerce').dropna()
        if not numeric_col.empty:
            min_val_data = numeric_col.min()
            max_val_data = numeric_col.max()

            if min_val_data == max_val_data: return

            key_prefix = column_name.lower().replace(" ", "_").replace("+", "plus").replace("-","").replace("(","").replace(")","")
            is_target_float = isinstance(default_step, float) or "meta" in column_name.lower() # Adjusted heuristic

            current_format_str = default_format_str
            if is_target_float and default_format_str == "%d": current_format_str = "%.1f"
            elif not is_target_float and default_format_str != "%d": current_format_str = "%d"

            s_min_val = float(min_val_data) if is_target_float else int(round(min_val_data))
            s_max_val = float(max_val_data) if is_target_float else int(round(max_val_data))

            if container is None:
                st.error(f"Error: Filter container for '{label}' is None. Cannot create inputs.")
                return

            col1, col2 = container.columns(2)
            # The number_input will use st.session_state values if keys exist,
            # otherwise, it will use the 'value' parameter.
            # Our clear button logic pre-populates session_state for these keys.
            user_min_val = col1.number_input(f"Min {label}", min_value=s_min_val, max_value=s_max_val,
                                          value=s_min_val, # Default if not in session_state
                                          step=default_step, format=current_format_str, key=f"{key_prefix}_min")
            user_max_val = col2.number_input(f"Max {label}", min_value=s_min_val, max_value=s_max_val,
                                          value=s_max_val, # Default if not in session_state
                                          step=default_step, format=current_format_str, key=f"{key_prefix}_max")

            # Check against actual data min/max to decide if filter is active
            # user_min_val and user_max_val will be from session_state if set by clear button or user
            if float(user_min_val) > float(s_min_val) or float(user_max_val) < float(s_max_val):
                 filters[column_name] = (float(user_min_val), float(user_max_val))


create_min_max_filter(st.sidebar, "overall", "Overall")
create_min_max_filter(st.sidebar, "height", "Height (cm)")
create_min_max_filter(st.sidebar, "weight", "Weight (kg)")
create_min_max_filter(st.sidebar, "skillMoves", "Skill Moves")
create_min_max_filter(st.sidebar, "weakFoot", "Weak Foot")
create_min_max_filter(st.sidebar, "ggMeta", "GG Meta", default_step=0.1, default_format_str="%.1f")
create_min_max_filter(st.sidebar, "esMetaSub", "ES Meta (Sub)", default_step=0.1, default_format_str="%.1f")
create_min_max_filter(st.sidebar, "esMeta", "ES Meta (Chem)", default_step=0.1, default_format_str="%.1f")


if "role" in df.columns:
    unique_roles = sorted(df["role"].dropna().unique())
    if unique_roles:
        st.sidebar.multiselect("Role", options=unique_roles, key="role_filter")
        if "role_filter" in st.session_state and st.session_state.role_filter:
            filters["role"] = st.session_state.role_filter

for role_col_name, role_label in [("roles+", "Roles+"), ("roles++", "Roles++")]:
    if role_col_name in df.columns:
        all_role_items = set()
        df[role_col_name].dropna().apply(lambda L: all_role_items.update(L if isinstance(L, list) else []))
        if all_role_items:
            current_key = f"{role_col_name}_any_filter"
            st.sidebar.multiselect(f"{role_label} (Any)", sorted(list(all_role_items)), key=current_key)
            if current_key in st.session_state and st.session_state[current_key]:
                filters[f"{role_col_name}_any"] = st.session_state[current_key]

all_ps_styles = set()
if 'PS' in df.columns:
    df['PS'].dropna().apply(lambda styles: all_ps_styles.update(styles if isinstance(styles, list) else []))
if 'PS+' in df.columns:
    df['PS+'].dropna().apply(lambda styles: all_ps_styles.update(styles if isinstance(styles, list) else []))

if all_ps_styles:
    st.sidebar.multiselect("PlayStyles (All Selected)", sorted(list(all_ps_styles)), key="playstyles_all_filter")
    if "playstyles_all_filter" in st.session_state and st.session_state.playstyles_all_filter:
        filters["playstyles_all"] = st.session_state.playstyles_all_filter

accel_types_map = {
    "ggAccelType": "GG Chem Accelerate Type",
    "esAccelType": "ES Chem Accelerate Type",
    "subAccelType": "Sub Accelerate Type"
}
for accel_col, accel_label in accel_types_map.items():
    if accel_col in df.columns:
        unique_accel = sorted(df[accel_col].dropna().unique())
        if unique_accel:
            st.sidebar.multiselect(accel_label, unique_accel, key=f"{accel_col}_filter")
            if f"{accel_col}_filter" in st.session_state and st.session_state[f"{accel_col}_filter"]:
                filters[accel_col] = st.session_state[f"{accel_col}_filter"]

with st.sidebar.expander("Role Familiarity", expanded=True):
    if "hasRolePlus" in df.columns:
        st.checkbox("Has Role Plus", key="hasRolePlus_checkbox")
        if "hasRolePlus_checkbox" in st.session_state and st.session_state.hasRolePlus_checkbox:
            filters["hasRolePlus"] = True # Value is True if checked
    if "hasRolePlusPlus" in df.columns:
        st.checkbox("Has Role Plus Plus", key="hasRolePlusPlus_checkbox")
        if "hasRolePlusPlus_checkbox" in st.session_state and st.session_state.hasRolePlusPlus_checkbox:
            filters["hasRolePlusPlus"] = True # Value is True if checked


igs_expander_container = st.sidebar.expander("In-Game Stats", expanded=False)
for attr_col in attribute_filter_order:
    create_min_max_filter(igs_expander_container, attr_col, attr_col.replace("_", " ").title())


# Apply filters (identical logic as before)
filtered_df = df.copy()
for col, val in filters.items():
    if col == "playstyles_all":
        def has_all_selected_styles_combined(row):
            if not val: return True
            combined_player_styles = set()
            ps_list = row.get("PS", [])
            ps_plus_list = row.get("PS+", [])
            if isinstance(ps_list, list): combined_player_styles.update(ps_list)
            if isinstance(ps_plus_list, list): combined_player_styles.update(ps_plus_list)
            return all(s in combined_player_styles for s in val)
        if not filtered_df.empty: # Ensure df is not empty before apply
             filtered_df = filtered_df[filtered_df.apply(has_all_selected_styles_combined, axis=1)]
        continue
    elif col == "roles+_any" or col == "roles++_any":
        actual_col_name = col.split('_')[0]
        if actual_col_name in filtered_df.columns:
            def has_any_selected_style_single_list(row_styles_list_item):
                if not isinstance(row_styles_list_item, list): return False
                return any(s_item in row_styles_list_item for s_item in val)
            try:
                if not filtered_df.empty:
                     mask = filtered_df[actual_col_name].apply(has_any_selected_style_single_list)
                     filtered_df = filtered_df[mask]
            except KeyError as e:
                st.warning(f"Skipping filter for '{actual_col_name}' due to unexpected issue: {e}")
        continue

    if isinstance(val, list):
        if col in filtered_df.columns and not filtered_df.empty:
            # Check if the column contains lists (for 'isin' behavior on list elements)
            if filtered_df[col].dropna().apply(lambda x: isinstance(x, list)).any():
                filtered_df = filtered_df[filtered_df[col].apply(lambda x: any(i in x for i in val) if isinstance(x, list) else False)]
            else: # Standard isin for non-list columns
                filtered_df = filtered_df[filtered_df[col].isin(val)]
    elif isinstance(val, tuple) and len(val) == 2:
        if col in filtered_df.columns and not filtered_df.empty:
            numeric_series = pd.to_numeric(filtered_df[col], errors='coerce')
            filtered_df = filtered_df[numeric_series.between(val[0], val[1], inclusive='both') & numeric_series.notna()]
    else: # Direct value match
        if col in filtered_df.columns and not filtered_df.empty:
            filtered_df = filtered_df[filtered_df[col] == val]


default_sort_column = "ggMeta"
if default_sort_column in filtered_df.columns and not filtered_df[default_sort_column].isnull().all():
    filtered_df = filtered_df.sort_values(by=default_sort_column, ascending=False)
elif "overall" in filtered_df.columns and not filtered_df["overall"].isnull().all(): # Fallback
    filtered_df = filtered_df.sort_values(by="overall", ascending=False)

columns_to_display = [
    "commonName", "role", "overall",
    "ggMeta", "ggChemStyle", "ggAccelType",
    "esMeta", "esChemStyle", "esAccelType",
    "esMetaSub","subAccelType",
    "hasRolePlus", "hasRolePlusPlus",
    "skillMoves", "weakFoot",
    "PS+", "PS", "roles+", "roles++",
    "positions", "foot",
    "bodyType",
    "height", "weight"
] + [col for col in attribute_filter_order if col in filtered_df.columns]

final_display_columns = [col for col in columns_to_display if col in filtered_df.columns]

if "hasRolePlus" in filtered_df.columns:
    filtered_df["hasRolePlus"] = filtered_df["hasRolePlus"].apply(lambda x: "✅" if x else "❌")
if "hasRolePlusPlus" in filtered_df.columns:
    filtered_df["hasRolePlusPlus"] = filtered_df["hasRolePlusPlus"].apply(lambda x: "✅" if x else "❌")

st.title("El Mostashar FC - Club Player Database")

if filters:
    st.subheader("Active Filters:")
    filter_tags = []
    for key, f_val in filters.items():
        display_name = key.replace("_", " ").title()
        if key == "playstyles_all": display_name = "PlayStyles (All Selected)"
        elif key == "roles+_any": display_name = "Roles+ (Any)"
        elif key == "roles++_any": display_name = "Roles++ (Any)"
        # Make display name for accel types more friendly
        elif key == "ggAccelType": display_name = "GG Chem Accelerate Type"
        elif key == "esAccelType": display_name = "ES Chem Accelerate Type"
        elif key == "subAccelType": display_name = "Sub Accelerate Type"


        if isinstance(f_val, list): val_str = ", ".join(map(str,f_val))
        elif isinstance(f_val, tuple): val_str = f"{f_val[0]} - {f_val[1]}"
        else: val_str = str(f_val)
        filter_tags.append(f"<span style='background-color:#333;color:#f5f5f5;padding:3px 7px;margin:2px;border-radius:12px;display:inline-block;font-size:0.9em;'>{display_name}: {val_str}</span>")
    st.markdown(" ".join(filter_tags), unsafe_allow_html=True)
    st.markdown("---")

st.subheader("Top Meta Ratings")
col1, col2, col3 = st.columns(3)

def display_top_metric(container, df_to_use, metric_col, title):
    if metric_col in df_to_use.columns and not df_to_use.empty:
        # Ensure metric_col is numeric for idxmax and nlargest
        df_to_use[metric_col] = pd.to_numeric(df_to_use[metric_col], errors='coerce')
        df_to_use_no_na = df_to_use.dropna(subset=[metric_col])

        if not df_to_use_no_na.empty:
            top_players_unique = df_to_use_no_na.loc[df_to_use_no_na.groupby("__true_player_id")[metric_col].idxmax()]
            top_df = top_players_unique.nlargest(3, metric_col)
            with container:
                st.markdown(f"**{title}**")
                for i, (_, row) in enumerate(top_df.iterrows()):
                    medal = ["🥇", "🥈", "🥉"][i] if i < 3 else ""
                    st.metric(label=f"{medal} {row.get('commonName', 'N/A')}", value=f'{row.get(metric_col, 0.0):.2f}')
        # else:
        #     with container: st.markdown(f"**{title}**\n\nNo data for this metric.");

display_top_metric(col1, filtered_df, "ggMeta", "Top GG Meta")
display_top_metric(col2, filtered_df, "esMeta", "Top EasySBC Meta (Full Chem)")
display_top_metric(col3, filtered_df, "esMetaSub", "Top EasySBC Meta (Sub)")


st.markdown(f"### Showing {filtered_df['player_origin_id'].nunique()} unique players ({len(filtered_df)} role entries)")

columns_to_drop_before_display = ["__true_player_id", "player_origin_id", "debug_index", "__player_id"]
display_df = filtered_df.drop(columns=[col for col in columns_to_drop_before_display if col in filtered_df.columns], errors="ignore")

if display_df.empty:
    st.warning("No players found matching the selected filters.")
else:
    if final_display_columns :
        st.dataframe(display_df[final_display_columns], use_container_width=True, hide_index=True)
    else:
        st.dataframe(display_df, use_container_width=True, hide_index=True)