import streamlit as st
import pandas as pd
import json
from pathlib import Path

st.set_page_config(layout="wide")

# Define data directory
data_dir = Path(__file__).resolve().parents[2] / "data"

# Load metadata from visual.csv
visual_df = pd.read_csv(data_dir / "visual.csv")

# Rename attribute fields
visual_df["field"] = visual_df["field"].apply(lambda x: x.replace("attribute", "", 1) if isinstance(x, str) and x.startswith("attribute") else x)
visual_df["field"] = visual_df["field"].apply(lambda x: x[0].lower() + x[1:] if isinstance(x, str) and x else x)

non_attribute_filter_order = [
    "rolesPlusPlus", "rolesPlus", "skillMoves", "weakFoot", "playstylesPlus", "playstyles",
    "positions", "foot", "bodytype", "accelerateType", "height", "weight", "overall"
]

attribute_filter_order = [
    "acceleration", "sprintSpeed", "positioning", "finishing", "shotPower", "longShots",
    "volleys", "penalties", "vision", "crossing", "fkAccuracy", "shortPassing",
    "longPassing", "curve", "agility", "balance", "reactions", "ballControl",
    "dribbling", "composure", "interceptions", "headingAccuracy", "defensiveAwareness",
    "standingTackle", "slidingTackle", "jumping", "stamina", "strength", "aggression",
    "gkDiving", "gkHandling", "gkKicking", "gkPositioning", "gkReflexes"
]

attribute_fields = visual_df[visual_df["field"].isin(attribute_filter_order)].copy()
attribute_fields["sort_index"] = attribute_fields["field"].apply(lambda x: attribute_filter_order.index(x))
attribute_fields = attribute_fields.sort_values(by="sort_index")

non_attribute_fields = visual_df[visual_df["field"].isin(non_attribute_filter_order)].copy()
non_attribute_fields["sort_index"] = non_attribute_fields["field"].apply(lambda x: non_attribute_filter_order.index(x))
non_attribute_fields = non_attribute_fields.sort_values(by="sort_index") 

# Load and normalize JSON data
with open(data_dir / "final.json", "r") as f:
    data = json.load(f)

for idx, p in enumerate(data):
    p["player_origin_id"] = f"{p.get('eaId', 'unknown')}_{idx}"
    p["debug_index"] = idx

# Ensure all players have a proper metaRatings format
def ensure_meta_list(p):
    if not isinstance(p.get("metaRatings"), list):
        if isinstance(p.get("metaRatings"), dict):
            p["metaRatings"] = [p["metaRatings"]]
        else:
            p["metaRatings"] = [{"archetype": "N/A", "metaRating": 0}]
    elif len(p["metaRatings"]) == 0:
        p["metaRatings"] = [{"archetype": "N/A", "metaRating": 0}]
    return p

data = [ensure_meta_list(p) for p in data]
df = pd.json_normalize(data)
df["player_origin_id"] = df["player_origin_id"].astype(str)

df["eaId"] = df["eaId"].astype(str)
df["__true_player_id"] = df["eaId"].fillna(df["commonName"])

# Rename attribute columns
df.rename(columns={col: col.replace("attribute", "", 1)[0].lower() + col.replace("attribute", "", 1)[1:] for col in df.columns if col.startswith("attribute")}, inplace=True)

# Expand metaRatings into one row per archetype and metaRating
if "metaRatings" in df.columns:
    df["metaRatings"] = df["metaRatings"].apply(lambda x: x if isinstance(x, list) else [{}])
    multi_meta = df[df["metaRatings"].apply(lambda x: isinstance(x, list) and len(x) > 1)]
    df = df.explode("metaRatings", ignore_index=True)
    df["metaRatings"] = df["metaRatings"].apply(lambda x: x if isinstance(x, dict) else {})
    df["archetype"] = df["metaRatings"].apply(lambda x: x.get("archetype"))
    df["metarating"] = df["metaRatings"].apply(lambda x: x.get("metaRating"))

    df["archetype"] = df["archetype"].fillna("N/A")
    df["metarating"] = df["metarating"].fillna(0)

    df = df.drop(columns=["metaRatings"])

df["height"] = pd.to_numeric(df["height"], errors="coerce")
df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
df["__player_id"] = df["player_origin_id"] + "_" + df["archetype"].astype(str)

# Define the rolesPlusPlus → archetype mapping
rpp_to_archetype = {
    "GK Goalkeeper": "goalkeeper", "GK Sweeper Keeper": "sweeper_keeper",
    "RB Fullback": "fullback", "RB Falseback": "falseback", "RB Wingback": "wingback", "RB Attacking Wingback": "attacking_wingback",
    "LB Fullback": "fullback", "LB Falseback": "falseback", "LB Wingback": "wingback", "LB Attacking Wingback": "attacking_wingback",
    "CB Defender": "defender", "CB Stopper": "stopper", "CB Ball-Playing Defender": "ball_playing_defender",
    "CDM Holding": "holding", "CDM Centre-Half": "centre_half", "CDM Deep-Lying Playmaker": "deep_lying_playmaker", "CDM Wide Half": "wide_half",
    "CM Box to Box": "box_to_box", "CM Holding": "holding", "CM Deep-Lying Playmaker": "deep_lying_playmaker", "CM Playmaker": "playmaker", "CM Half-Winger": "half_winger",
    "RM Winger": "winger", "RM Wide Midfielder": "wide_midfielder", "RM Wide Playmaker": "wide_playmaker", "RM Inside Forward": "inside_forward",
    "LM Winger": "winger", "LM Wide Midfielder": "wide_midfielder", "LM Wide Playmaker": "wide_playmaker", "LM Inside Forward": "inside_forward",
    "CAM Playmaker": "playmaker", "CAM Shadow Striker": "shadow_striker", "CAM Half Winger": "half_winger", "CAM Classic 10": "classic_ten",
    "RW Winger": "winger", "RW Inside Forward": "inside_forward", "RW Wide Playmaker": "wide_playmaker",
    "LW Winger": "winger", "LW Inside Forward": "inside_forward", "LW Wide Playmaker": "wide_playmaker",
    "ST Advanced Forward": "advanced_forward", "ST Poacher": "poacher", "ST False 9": "false_nine", "ST Target Forward": "target_forward"
}

# Sidebar for filtering
st.sidebar.header("Filter Players")
filters = {}

evolution_values = [True, False]
selected_evolution = st.sidebar.selectbox("evolution", options=["All"] + evolution_values)
if selected_evolution != "All":
    filters["evolution"] = selected_evolution

# Define attribute filter order
attribute_filter_order = [
    "acceleration", "sprintSpeed", "positioning", "finishing", "shotPower", "longShots",
    "volleys", "penalties", "vision", "crossing", "fkAccuracy", "shortPassing",
    "longPassing", "curve", "agility", "balance", "reactions", "ballControl",
    "dribbling", "composure", "interceptions", "headingAccuracy", "defensiveAwareness",
    "standingTackle", "slidingTackle", "jumping", "stamina", "strength", "aggression",
    "gkDiving", "gkHandling", "gkKicking", "gkPositioning", "gkReflexes"
]
attribute_fields["sort_index"] = attribute_fields["field"].apply(lambda x: attribute_filter_order.index(x) if x in attribute_filter_order else -1)
attribute_fields = attribute_fields.sort_values(by="sort_index")

# Process non-attribute fields first
for _, row in non_attribute_fields.iterrows():
    col = row["field"]
    if col not in df.columns:
        continue

    filter_type = row["filter type"]
    if filter_type == "dropdown/selection":
        series = df[col].dropna()
        if series.apply(lambda x: isinstance(x, list)).any():
            unique_vals = sorted(set(item for sublist in series if isinstance(sublist, list) for item in sublist))
        else:
            unique_vals = sorted(series.unique())
        selected_vals = st.sidebar.multiselect(f"{col}", unique_vals)
        if selected_vals:
            filters[col] = selected_vals
    elif filter_type == "min/max boxes in one line":
        min_val = row["min"] if not pd.isna(row["min"]) else df[col].min()
        max_val = row["max"] if not pd.isna(row["max"]) else df[col].max()
        min_val, max_val = int(min_val), int(max_val)
        selected_range = st.sidebar.slider(f"{col}", min_val, max_val, (min_val, max_val))
        filters[col] = selected_range
    if col == "archetype":
        continue
    if col in ["height", "weight"] and col not in filters:
        min_val = int(df[col].min())
        max_val = int(df[col].max())
        selected_range = st.sidebar.slider(f"{col}", min_val, max_val, (min_val, max_val))
        filters[col] = selected_range

# Auto-control archetype filter if rolesPlusPlus is selected
selected_rpp = filters.get("rolesPlusPlus", [])
if selected_rpp:
    mapped_archetypes = sorted(set(rpp_to_archetype[r] for r in selected_rpp if r in rpp_to_archetype))
    filters["archetype"] = mapped_archetypes

with st.sidebar.expander("In-Game Stats", expanded=False):
    # Process attribute fields
    for _, row in attribute_fields.iterrows():
        col = row["field"]
        if col not in df.columns:
            continue

        filter_type = row["filter type"]
        if filter_type == "dropdown/selection":
            series = df[col].dropna()
            if series.apply(lambda x: isinstance(x, list)).any():
                unique_vals = sorted(set(item for sublist in series if isinstance(sublist, list) for item in sublist))
            else:
                unique_vals = sorted(series.unique())
            selected_vals = st.multiselect(f"{col}", unique_vals)
            if selected_vals:
                filters[col] = selected_vals
        elif filter_type == "min/max boxes in one line":
            min_val = row["min"] if not pd.isna(row["min"]) else df[col].min()
            max_val = row["max"] if not pd.isna(row["max"]) else df[col].max()
            min_val, max_val = int(min_val), int(max_val)
            selected_range = st.slider(f"{col}", min_val, max_val, (min_val, max_val))
            filters[col] = selected_range

# Include all players by default, even those with metaRating = 0
df["metarating"] = pd.to_numeric(df["metarating"], errors="coerce").fillna(0)

# Apply filters
filtered_df = df.copy()
for col, val in filters.items():
    if col == "evolution":
        filtered_df = filtered_df[filtered_df[col] == val]
        continue
    if col in ["playstyles", "playstylesPlus"]:
        filtered_df = filtered_df[filtered_df.apply(
            lambda row: any(i in (row.get("playstyles") or []) for i in val) or any(i in (row.get("playstylesPlus") or []) for i in val),
            axis=1
        )]
    elif isinstance(val, list):
        if df[col].apply(lambda x: isinstance(x, list)).any():
            filtered_df = filtered_df[filtered_df[col].apply(lambda x: any(i in x for i in val) if isinstance(x, list) else False)]
        else:
            filtered_df = filtered_df[filtered_df[col].isin(val)]
    elif isinstance(val, tuple) and len(val) == 2:
        filtered_df = filtered_df[filtered_df[col].between(val[0], val[1])]

# Custom column order
column_order = [
    "commonName", "archetype", "metarating", "rolesPlusPlus", "rolesPlus",
    "skillMoves", "weakFoot", "playstylesPlus", "playstyles", "positions", "foot",
    "bodytype", "accelerateType", "height", "weight", "overall",
    "acceleration", "sprintSpeed", "positioning", "finishing",
    "shotPower", "longShots", "volleys", "penalties",
    "vision", "crossing", "fkAccuracy", "shortPassing",
    "longPassing", "curve", "agility", "balance",
    "reactions", "ballControl", "dribbling", "composure",
    "interceptions", "headingAccuracy", "defensiveAwareness",
    "standingTackle", "slidingTackle", "jumping", "stamina",
    "strength", "aggression", "gkDiving", "gkHandling",
    "gkKicking", "gkPositioning", "gkReflexes"
]
existing_columns = [col for col in column_order if col in filtered_df.columns]
remaining_columns = [col for col in filtered_df.columns if col not in existing_columns]
filtered_df = filtered_df[existing_columns + remaining_columns]

# Display
st.title("Mostashar Moza Player Database")
st.markdown(f"### Showing {filtered_df['player_origin_id'].nunique()} unique players")
sort_by = "metarating" if "metarating" in filtered_df.columns else "overall"
filtered_df = filtered_df.drop(columns=["__true_player_id", "player_origin_id", "debug_index", "__player_id"], errors="ignore")
st.dataframe(filtered_df.sort_values(by=sort_by, ascending=False), use_container_width=True, hide_index=True)