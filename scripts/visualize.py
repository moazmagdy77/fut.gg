import streamlit as st
import pandas as pd
import json
from pathlib import Path

st.set_page_config(layout="wide")

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
# Expects club_final.json from the python_processing_script_v1
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
    p["player_origin_id"] = f"{p.get('eaId', 'unknown')}_{idx}" # Unique ID for merging later if needed
    p["debug_index"] = idx # For debugging if necessary

df = pd.json_normalize(data) # type: ignore

if df.empty:
    st.warning("No data loaded from club_final.json. The file might be empty or structured incorrectly.")
    st.stop()

# Filter out rows with empty or NaN commonName early
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


# Add evolution column if it doesn't exist, default to False
if "evolution" not in df.columns:
    df["evolution"] = False


# Rename attribute columns (e.g., attributeAcceleration to acceleration)
df.rename(columns={col: col.replace("attribute", "", 1)[0].lower() + col.replace("attribute", "", 1)[1:] 
                   for col in df.columns if col.startswith("attribute")}, inplace=True)

# Explode metaRatings properly
meta_ratings_col_name = "metaRatings" # This matches the output of python_processing_script_v1
if meta_ratings_col_name in df.columns:
    df[meta_ratings_col_name] = df[meta_ratings_col_name].apply(lambda x: x if isinstance(x, list) else ([{}] if not x else [x])) # Ensure it's a list, handle None/empty
    df = df.explode(meta_ratings_col_name, ignore_index=True)
    
    # Ensure the exploded column is treated as a dictionary
    df[meta_ratings_col_name] = df[meta_ratings_col_name].apply(lambda x: x if isinstance(x, dict) else {})
    
    # Extract fields using the keys from python_processing_script_v1's meta_entry output
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
                        "esMeta", "esChemStyle", "esAccelType""esMetaSub", "subAccelType"]
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
filters = {}

if "evolution" in df.columns:
    unique_evo_vals = sorted(list(df["evolution"].unique()))
    options = ["All"] + unique_evo_vals
    selected_evolution = st.sidebar.selectbox("Evolution", options=options, index=0)
    if selected_evolution != "All":
        filters["evolution"] = selected_evolution

def create_min_max_filter(container, column_name, label, default_step=1, default_format_str="%d"):
    """Creates min/max number_input filters within the given container (e.g., st.sidebar or an expander)."""
    if column_name in df.columns and not df[column_name].isnull().all():
        numeric_col = pd.to_numeric(df[column_name], errors='coerce').dropna()
        if not numeric_col.empty:
            min_val_data = numeric_col.min() 
            max_val_data = numeric_col.max() 
            
            if min_val_data == max_val_data:
                # container.text(f"{label}: {min_val_data} (single value in dataset)") # Less verbose
                return

            key_prefix = column_name.lower().replace(" ", "_").replace("+", "plus").replace("-","").replace("(","").replace(")","")
            is_target_float = isinstance(default_step, float)
            
            current_format_str = default_format_str
            if is_target_float and default_format_str == "%d": current_format_str = "%.1f" 
            elif not is_target_float and default_format_str != "%d": current_format_str = "%d"

            if is_target_float:
                s_min_val = float(min_val_data)
                s_max_val = float(max_val_data)
            else: 
                s_min_val = int(round(min_val_data))
                s_max_val = int(round(max_val_data))
            
            val_min_for_input = s_min_val 
            val_max_for_input = s_max_val
            
            if container is None: # Should not happen if called correctly
                st.error(f"Error: Filter container for '{label}' is None. Cannot create inputs.")
                return

            col1, col2 = container.columns(2) 
            user_min = col1.number_input(f"Min {label}", min_value=s_min_val, max_value=s_max_val, value=val_min_for_input, step=default_step, format=current_format_str, key=f"{key_prefix}_min")
            user_max = col2.number_input(f"Max {label}", min_value=s_min_val, max_value=s_max_val, value=val_max_for_input, step=default_step, format=current_format_str, key=f"{key_prefix}_max")
            
            if float(user_min) > float(min_val_data) or float(user_max) < float(max_val_data):
                 filters[column_name] = (float(user_min), float(user_max))

# General Numeric Filters (directly in sidebar)
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
        selected_roles = st.sidebar.multiselect("Role", options=unique_roles)
        if selected_roles:
            filters["role"] = selected_roles

# Filters for roles+ and roles++
for role_col_name, role_label in [("roles+", "Roles+"), ("roles++", "Roles++")]:
    if role_col_name in df.columns:
        all_role_items = set()
        df[role_col_name].dropna().apply(lambda L: all_role_items.update(L if isinstance(L, list) else []))
        if all_role_items:
            selected_role_items = st.sidebar.multiselect(f"{role_label} (Any)", sorted(list(all_role_items)))
            if selected_role_items:
                filters[f"{role_col_name}_any"] = selected_role_items


all_ps_styles = set()
if 'PS' in df.columns:
    df['PS'].dropna().apply(lambda styles: all_ps_styles.update(styles if isinstance(styles, list) else []))
if 'PS+' in df.columns:
    df['PS+'].dropna().apply(lambda styles: all_ps_styles.update(styles if isinstance(styles, list) else []))

if all_ps_styles:
    selected_playstyles = st.sidebar.multiselect("PlayStyles (All Selected)", sorted(list(all_ps_styles))) 
    if selected_playstyles:
        filters["playstyles_all"] = selected_playstyles 
            
if "ggAccelType" in df.columns: 
    unique_gg_accel = sorted(df["ggAccelType"].dropna().unique())
    if unique_gg_accel:
        selected_gg_accel = st.sidebar.multiselect("GG Chem Accelerate Type", unique_gg_accel)
        if selected_gg_accel:
            filters["ggAccelType"] = selected_gg_accel

if "esAccelType" in df.columns: 
    unique_es_accel = sorted(df["esAccelType"].dropna().unique())
    if unique_es_accel:
        selected_es_accel = st.sidebar.multiselect("ES Chem Accelerate Type", unique_es_accel)
        if selected_es_accel:
            filters["esAccelType"] = selected_es_accel

if "subAccelType" in df.columns: 
    unique_sub_accel = sorted(df["subAccelType"].dropna().unique())
    if unique_sub_accel:
        selected_sub_accel = st.sidebar.multiselect("Sub Accelerate Type", unique_sub_accel)
        if selected_sub_accel:
            filters["subAccelType"] = selected_sub_accel


with st.sidebar.expander("Role Familiarity", expanded=True):
    if "hasRolePlus" in df.columns:
        has_role_plus_checkbox = st.checkbox("Has Role Plus")
        if has_role_plus_checkbox:
            filters["hasRolePlus"] = True
    if "hasRolePlusPlus" in df.columns:
        has_role_plus_plus_checkbox = st.checkbox("Has Role Plus Plus")
        if has_role_plus_plus_checkbox:
            filters["hasRolePlusPlus"] = True

igs_expander_container = st.sidebar.expander("In-Game Stats", expanded=False)
for attr_col in attribute_filter_order:
    create_min_max_filter(igs_expander_container, attr_col, attr_col.replace("_", " ").title())


# Apply filters
filtered_df = df.copy()
for col, val in filters.items():
    if col == "playstyles_all": 
        def has_all_selected_styles_combined(row):
            # If no playstyles are selected in the filter, all players pass this specific filter
            if not val: # 'val' is the list of selected playstyles from the filter
                return True
            
            combined_player_styles = set() 
            ps_list = row.get("PS", [])
            ps_plus_list = row.get("PS+", [])

            if isinstance(ps_list, list): 
                combined_player_styles.update(ps_list)
            if isinstance(ps_plus_list, list): 
                combined_player_styles.update(ps_plus_list)
            
            # Check if all styles in 'val' (selected by user) are present in the player's combined styles
            return all(s in combined_player_styles for s in val)
        filtered_df = filtered_df[filtered_df.apply(has_all_selected_styles_combined, axis=1)]
        continue
    elif col == "roles+_any" or col == "roles++_any": 
        actual_col_name = col.split('_')[0] 
        if actual_col_name in filtered_df.columns: 
            def has_any_selected_style_single_list(row_styles_list):
                if not isinstance(row_styles_list, list): return False
                return any(s in row_styles_list for s in val)
            filtered_df = filtered_df[filtered_df[actual_col_name].apply(has_any_selected_style_single_list)]
        continue

    if isinstance(val, list): 
        if filtered_df[col].dropna().apply(lambda x: isinstance(x, list)).any():
            filtered_df = filtered_df[filtered_df[col].apply(lambda x: any(i in x for i in val) if isinstance(x, list) else False)]
        else:
            filtered_df = filtered_df[filtered_df[col].isin(val)]
    elif isinstance(val, tuple) and len(val) == 2: 
        numeric_series = pd.to_numeric(filtered_df[col], errors='coerce')
        filtered_df = filtered_df[numeric_series.between(val[0], val[1], inclusive='both') & numeric_series.notna()]

    else: 
        filtered_df = filtered_df[filtered_df[col] == val]


# Default sort by ggMeta descending
default_sort_column = "ggMeta"
if default_sort_column in filtered_df.columns:
    filtered_df = filtered_df.sort_values(by=default_sort_column, ascending=False)
elif "overall" in filtered_df.columns: # Fallback
    filtered_df = filtered_df.sort_values(by="overall", ascending=False)


# Columns to display
columns_to_display = [
    "commonName", "role", "overall",
    "ggMeta", "ggChemStyle", "ggAccelType",  
    "esMeta", "esChemStyle", "esAccelType", 
    "esMetaSub","subAccelType", 
    "hasRolePlus", "hasRolePlusPlus",
    "skillMoves", "weakFoot", 
    "PS+", "PS", "roles+", "roles++", 
    "positions", "foot",
    "bodyType", "accelerateType", 
    "height", "weight"
] + [col for col in attribute_filter_order if col in filtered_df.columns]

final_display_columns = [col for col in columns_to_display if col in filtered_df.columns]


# Formatting for boolean flags
if "hasRolePlus" in filtered_df.columns:
    filtered_df["hasRolePlus"] = filtered_df["hasRolePlus"].apply(lambda x: "✅" if x else "❌")
if "hasRolePlusPlus" in filtered_df.columns:
    filtered_df["hasRolePlusPlus"] = filtered_df["hasRolePlusPlus"].apply(lambda x: "✅" if x else "❌")


# Display
st.title("El Mostashar FC - Club Player Database")

if filters:
    st.subheader("Active Filters:")
    filter_tags = []
    for key, f_val in filters.items():
        display_name = key.replace("_", " ").title()
        if key == "playstyles_all": display_name = "PlayStyles (All Selected)" 
        elif key == "roles+_any": display_name = "Roles+ (Any)"
        elif key == "roles++_any": display_name = "Roles++ (Any)"
        
        if isinstance(f_val, list):
            val_str = ", ".join(map(str,f_val))
        elif isinstance(f_val, tuple):
            val_str = f"{f_val[0]} - {f_val[1]}"
        else:
            val_str = str(f_val)
        filter_tags.append(f"<span style='background-color:#333;color:#f5f5f5;padding:3px 7px;margin:2px;border-radius:12px;display:inline-block;font-size:0.9em;'>{display_name}: {val_str}</span>")
    st.markdown(" ".join(filter_tags), unsafe_allow_html=True)
    st.markdown("---")

# Top 3 Players Display
st.subheader("Top Meta Ratings")
col1, col2, col3 = st.columns(3)

# Top 3 GG Meta
if "ggMeta" in filtered_df.columns and not filtered_df.empty:
    top_gg_players_unique = filtered_df.loc[filtered_df.groupby("__true_player_id")["ggMeta"].idxmax()]
    top_gg_df = top_gg_players_unique.nlargest(3, "ggMeta")
    with col1:
        st.markdown("**Top GG Meta**")
        for i, (_, row) in enumerate(top_gg_df.iterrows()):
            medal = ["🥇", "🥈", "🥉"][i] if i < 3 else "" 
            st.metric(label=f"{medal} {row.get('commonName', 'N/A')}", value=f'{row.get("ggMeta", 0.0):.2f}')

# Top 3 ES Meta (3 Chem)
if "esMeta" in filtered_df.columns and not filtered_df.empty: 
    top_es_players_unique = filtered_df.loc[filtered_df.groupby("__true_player_id")["esMeta"].idxmax()]
    top_es_df = top_es_players_unique.nlargest(3, "esMeta")
    with col2:
        st.markdown("**Top EasySBC Meta (Full Chem)**")
        for i, (_, row) in enumerate(top_es_df.iterrows()):
            medal = ["🥇", "🥈", "�"][i] if i < 3 else ""
            st.metric(label=f"{medal} {row.get('commonName', 'N/A')}", value=f'{row.get("esMeta", 0.0):.2f}')

# Top 3 ES Meta (0 Chem)
if "esMetaSub" in filtered_df.columns and not filtered_df.empty:
    top_es0_players_unique = filtered_df.loc[filtered_df.groupby("__true_player_id")["esMetaSub"].idxmax()]
    top_es0_df = top_es0_players_unique.nlargest(3, "esMetaSub")
    with col3:
        st.markdown("**Top EasySBC Meta (Sub)**")
        for i, (_, row) in enumerate(top_es0_df.iterrows()):
            medal = ["🥇", "🥈", "🥉"][i] if i < 3 else ""
            st.metric(label=f"{medal} {row.get('commonName', 'N/A')}", value=f'{row.get("esMetaSub", 0.0):.2f}')


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

�