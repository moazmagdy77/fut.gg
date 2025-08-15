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

# This list helps define the order of attributes in the final display
attribute_filter_order = [
    "acceleration", "sprintSpeed", "positioning", "finishing", "shotPower", "longShots",
    "volleys", "penalties", "vision", "crossing", "fkAccuracy", "shortPassing",
    "longPassing", "curve", "agility", "balance", "reactions", "ballControl",
    "dribbling", "composure", "interceptions", "headingAccuracy", "defensiveAwareness",
    "standingTackle", "slidingTackle", "jumping", "stamina", "strength", "aggression",
    "gkDiving", "gkHandling", "gkKicking", "gkPositioning", "gkReflexes"
]

@st.cache_data
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

    # --- FIX: Dynamically determine all meta columns to ensure none are missed ---
    all_meta_keys = set()
    for p in data:
        if p.get('metaRatings'):
            for r in p['metaRatings']:
                all_meta_keys.update(r.keys())
    
    # Define the columns that are at the top level of each player object
    top_level_cols = list(data[0].keys() if data else [])
    top_level_cols.remove('metaRatings')

    df = pd.json_normalize(data, record_path='metaRatings', meta=top_level_cols, errors='ignore')
    
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

    float_cols = ['ggMeta', 'ggMetaSub', 'esMeta', 'esMetaSub', 'avgMeta', 'avgMetaSub']
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    # Recompute role familiarity flags based on the current role
    for col in ['roles+', 'roles++']:
        if col not in df.columns: df[col] = pd.Series([[] for _ in range(len(df))])
        df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])
        
    df['hasRolePlus'] = df.apply(lambda row: row.get('role') in row.get('roles+', []), axis=1)
    df['hasRolePlusPlus'] = df.apply(lambda row: row.get('role') in row.get('roles++', []), axis=1)

    return df

df = load_data(data_dir / "club_final.json")
if df.empty:
    st.stop()

# --- Sidebar filters ---
st.sidebar.header("Filter Players")

if st.sidebar.button("🧹 Clear All Filters", key="clear_filters_main_button"):
    st.session_state.clear_filters_button_clicked = True
    st.rerun()

if st.session_state.get("clear_filters_button_clicked", False):
    st.session_state.clear_filters_button_clicked = False
    for key in st.session_state.keys():
        if key.endswith(('_min', '_max', '_filter', '_checkbox')):
            del st.session_state[key]
    st.rerun()

filters = {}

def create_min_max_filter(container, column_name, label, step):
    if column_name in df.columns and pd.api.types.is_numeric_dtype(df[column_name]):
        numeric_col = df[column_name].dropna()
        if not numeric_col.empty and numeric_col.min() != numeric_col.max():
            is_int = pd.api.types.is_integer_dtype(df[column_name])
            min_val, max_val = (int(numeric_col.min()), int(numeric_col.max())) if is_int else (float(numeric_col.min()), float(numeric_col.max()))
            fmt = "%d" if is_int else "%.1f"
            
            c1, c2 = container.columns(2)
            user_min = c1.number_input(f"Min {label}", value=min_val, min_value=min_val, max_value=max_val, step=step, format=fmt, key=f"{column_name}_min")
            user_max = c2.number_input(f"Max {label}", value=max_val, min_value=min_val, max_value=max_val, step=step, format=fmt, key=f"{column_name}_max")
            
            if user_min > min_val or user_max < max_val:
                 filters[column_name] = (user_min, user_max)

st.sidebar.selectbox("Evolution", options=["All", True, False], format_func=lambda x: "Evo Players" if x is True else "Standard Players" if x is False else "All", key="evolution_filter")
if st.session_state.get("evolution_filter") != "All": filters["evolution"] = st.session_state.evolution_filter

create_min_max_filter(st.sidebar, "avgMeta", "Avg On-Chem Meta", 0.1)
create_min_max_filter(st.sidebar, "avgMetaSub", "Avg Sub Meta", 0.1)
create_min_max_filter(st.sidebar, "overall", "Overall", 1)
create_min_max_filter(st.sidebar, "skillMoves", "Skill Moves", 1)
create_min_max_filter(st.sidebar, "weakFoot", "Weak Foot", 1)
create_min_max_filter(st.sidebar, "height", "Height (cm)", 1)
create_min_max_filter(st.sidebar, "weight", "Weight (kg)", 1)

with st.sidebar.expander("Detailed Meta Ratings"):
    create_min_max_filter(st, "ggMeta", "GG Meta", 0.1)
    create_min_max_filter(st, "ggMetaSub", "GG Meta (Sub)", 0.1)
    create_min_max_filter(st, "esMeta", "ES Meta", 0.1)
    create_min_max_filter(st, "esMetaSub", "ES Meta (Sub)", 0.1)

# (Other filter widgets remain the same...)
# ...

with st.sidebar.expander("In-Game Stats"):
    for attr in attribute_filter_order:
        create_min_max_filter(st, attr, attr.replace("_", " ").title(), 1)

# --- Apply filters and sort (logic unchanged) ---
# ...

# --- Display Logic ---
st.title("El Mostashar FC - Club Player Database")

# ... (Active filter display logic unchanged) ...

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
                    if i < 3: rank_display = ["🥇", "🥈", "🥉"][i]
                    label = f"{rank_display} {row.get('commonName', 'N/A')} ({row.get('role', 'N/A')})"
                    st.metric(label=label, value=f'{row.get(metric_col, 0.0):.2f}')

display_top_metric(col1, filtered_df, "avgMeta", "Top Average On-Chem Meta", n=5)
display_top_metric(col2, filtered_df, "avgMetaSub", "Top Average Sub Meta", n=5)

st.markdown("---")
st.markdown(f"### Player List ({filtered_df['player_origin_id'].nunique()} unique players, {len(filtered_df)} total entries)")

columns_to_display = [
    "commonName", "role", "overall", "avgMeta", 
    "ggMeta", "ggChemStyle", "ggAccelType", 
    "esMeta", "esChemStyle", "esAccelType",
    "avgMetaSub", "subAccelType", "ggMetaSub", "esMetaSub",
    "hasRolePlusPlus", "hasRolePlus", "skillMoves", "weakFoot", "foot", "height", "weight", "bodyType",
    "PS+", "PS", "positions", "roles++", "roles+"
] + attribute_filter_order

final_display_columns = [col for col in columns_to_display if col in df.columns]

display_df = filtered_df.copy()
for col in ["hasRolePlus", "hasRolePlusPlus"]:
    if col in display_df.columns:
        display_df[col] = display_df[col].apply(lambda x: "✅" if x else "❌")

display_df.drop(columns=[c for c in ["player_origin_id", "__true_player_id"] if c in display_df.columns], inplace=True, errors="ignore")

if display_df.empty:
    st.warning("No players found matching the selected filters.")
else:
    st.dataframe(display_df[final_display_columns], use_container_width=True, hide_index=True)