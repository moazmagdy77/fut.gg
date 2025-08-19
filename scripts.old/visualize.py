import streamlit as st
import pandas as pd
import json
from pathlib import Path

st.set_page_config(layout="wide")

# Define data directory
data_dir = Path(__file__).resolve().parents[1] / "data"

# Load metadata
visual_df = pd.read_csv(data_dir / "visual.csv")

# Process visual fields
visual_df["field"] = visual_df["field"].apply(lambda x: x.replace("attribute", "", 1) if isinstance(x, str) and x.startswith("attribute") else x)
visual_df["field"] = visual_df["field"].apply(lambda x: x[0].lower() + x[1:] if isinstance(x, str) and x else x)

non_attribute_filter_order = [
    "overall", "rolesPlusPlus", "rolesPlus", "skillMoves", "weakFoot", "playstylesPlus", "playstyles",
    "positions", "foot", "bodytype", "accelerateType", "height", "weight"
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

# Load and normalize JSON
with open(data_dir / "final.json", "r") as f:
    data = json.load(f)

for idx, p in enumerate(data):
    p["player_origin_id"] = f"{p.get('eaId', 'unknown')}_{idx}"
    p["debug_index"] = idx

# Ensure all players have a metaRatings list
def ensure_meta_list(p):
    if not isinstance(p.get("metaRatings"), list):
        if isinstance(p.get("metaRatings"), dict):
            p["metaRatings"] = [p["metaRatings"]]
        else:
            p["metaRatings"] = [{"archetype": "N/A", "metaRating_0chem": 0, "metaRating_3chem": 0}]
    elif len(p["metaRatings"]) == 0:
        p["metaRatings"] = [{"archetype": "N/A", "metaRating_0chem": 0, "metaRating_3chem": 0}]
    return p

data = [ensure_meta_list(p) for p in data]
df = pd.json_normalize(data)
df["player_origin_id"] = df["player_origin_id"].astype(str)
df["eaId"] = df["eaId"].astype(str)
df["__true_player_id"] = df["eaId"].fillna(df["commonName"])

# Rename attribute columns
df.rename(columns={col: col.replace("attribute", "", 1)[0].lower() + col.replace("attribute", "", 1)[1:] for col in df.columns if col.startswith("attribute")}, inplace=True)

# Explode metaRatings properly
if "metaRatings" in df.columns:
    df["metaRatings"] = df["metaRatings"].apply(lambda x: x if isinstance(x, list) else [{}])
    df = df.explode("metaRatings", ignore_index=True)
    df["metaRatings"] = df["metaRatings"].apply(lambda x: x if isinstance(x, dict) else {})
    df["archetype"] = df["metaRatings"].apply(lambda x: x.get("archetype", "N/A"))
    df["metaRating_0chem"] = df["metaRatings"].apply(lambda x: x.get("metaR_0chem", 0))
    df["metaRating_3chem"] = df["metaRatings"].apply(lambda x: x.get("metaR_3chem", 0))
    df["bestChemStyle"] = df["metaRatings"].apply(lambda x: x.get("bestChemStyle", "None"))
    df["accelerateType_chem"] = df["metaRatings"].apply(lambda x: x.get("accelerateType_chem", "Unknown"))
    df["hasRolePlus"] = df["metaRatings"].apply(lambda x: x.get("hasRolePlus", False))
    df["hasRolePlusPlus"] = df["metaRatings"].apply(lambda x: x.get("hasRolePlusPlus", False))
    df = df.drop(columns=["metaRatings"])

df["height"] = pd.to_numeric(df["height"], errors="coerce")
df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
df["height"] = df["height"].fillna(0).astype(int)
df["weight"] = df["weight"].fillna(0).astype(int)
df["__player_id"] = df["player_origin_id"] + "_" + df["archetype"].astype(str)

# Add mapping from rolesPlusPlus to archetype keys
rpp_to_archetype = {
    "GK Goalkeeper": "goalkeeper", "GK Sweeper Keeper": "sweeper_keeper",
    "RB Fullback": "fullback", "RB Falseback": "falseback", "RB Wingback": "wingback", "RB Attacking Wingback": "attacking_wingback",
    "LB Fullback": "fullback", "LB Falseback": "falseback", "LB Wingback": "wingback", "LB Attacking Wingback": "attacking_wingback",
    "CB Defender": "defender", "CB Stopper": "stopper", "CB Ball-Playing Defender": "ball_playing_defender",
    "CDM Holding": "holding", "CDM Centre-Half": "centre_half", "CDM Deep-Lying Playmaker": "deep_lying_playmaker", "CDM Wide Half": "wide_half",
    "CM Box to Box": "box_to_box", "CM Holding": "holding", "CM Deep-Lying Playmaker": "deep_lying_playmaker", "CM Playmaker": "playmaker", "CM Half-Winger": "half_ winger",
    "RM Winger": "winger", "RM Wide Midfielder": "wide_midfielder", "RM Wide Playmaker": "wide_playmaker", "RM Inside Forward": "inside_forward",
    "LM Winger": "winger", "LM Wide Midfielder": "wide_midfielder", "LM Wide Playmaker": "wide_playmaker", "LM Inside Forward": "inside_forward",
    "CAM Playmaker": "playmaker", "CAM Shadow Striker": "shadow_striker", "CAM Half Winger": "half_winger", "CAM Classic 10": "classic_ten",
    "RW Winger": "winger", "RW Inside Forward": "inside_forward", "RW Wide Playmaker": "wide_playmaker",
    "LW Winger": "winger", "LW Inside Forward": "inside_forward", "LW Wide Playmaker": "wide_playmaker",
    "ST Advanced Forward": "advanced_forward", "ST Poacher": "poacher", "ST False 9": "false_nine", "ST Target Forward": "target_forward"
}

# Sidebar filters
st.sidebar.header("Filter Players")
filters = {}

# Evolution filter (if exists)
evolution_values = [True, False]
selected_evolution = st.sidebar.selectbox("evolution", options=["All"] + evolution_values)
if selected_evolution != "All":
    filters["evolution"] = selected_evolution

# Height filter
if "height" in df.columns and not df["height"].isnull().all():
    min_height = int(df["height"].min())
    max_height = int(df["height"].max())
    selected_height = st.sidebar.slider("Height (cm)", min_height, max_height, (min_height, max_height))
    filters["height"] = selected_height

# Weight filter
if "weight" in df.columns and not df["weight"].isnull().all():
    min_weight = int(df["weight"].min())
    max_weight = int(df["weight"].max())
    selected_weight = st.sidebar.slider("Weight (kg)", min_weight, max_weight, (min_weight, max_weight))
    filters["weight"] = selected_weight

# Archetype filter
archetypes = sorted(df["archetype"].dropna().unique())
selected_archetypes = st.sidebar.multiselect("Archetype", options=archetypes)
if selected_archetypes:
    filters["archetype"] = selected_archetypes

# Process non-attribute fields
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
        selected_range = st.sidebar.slider(f"{col}", int(min_val), int(max_val), (int(min_val), int(max_val)))
        filters[col] = selected_range

# Recompute hasRolePlus and hasRolePlusPlus dynamically based on archetype and rolesPlus/rolesPlusPlus
def recompute_role_flags(row):
    roles_plus = row.get("rolesPlus", [])
    roles_plus_plus = row.get("rolesPlusPlus", [])
    archetype = row.get("archetype", "N/A")
    mapped_roles_plus = [rpp_to_archetype.get(role, None) for role in roles_plus]
    mapped_roles_plus_plus = [rpp_to_archetype.get(role, None) for role in roles_plus_plus]
    has_role_plus = archetype in mapped_roles_plus
    has_role_plus_plus = archetype in mapped_roles_plus_plus
    return pd.Series({"hasRolePlus": has_role_plus, "hasRolePlusPlus": has_role_plus_plus})

role_flags = df.apply(recompute_role_flags, axis=1)
df["hasRolePlus"] = role_flags["hasRolePlus"]
df["hasRolePlusPlus"] = role_flags["hasRolePlusPlus"]

# Additional filters for hasRolePlus and hasRolePlusPlus
with st.sidebar.expander("Role Familiarity", expanded=True):
    has_role_plus = st.checkbox("Has Role Plus")
    has_role_plus_plus = st.checkbox("Has Role Plus Plus")
    if has_role_plus:
        filters["hasRolePlus"] = True
    if has_role_plus_plus:
        filters["hasRolePlusPlus"] = True

with st.sidebar.expander("In-Game Stats", expanded=False):
    for _, row in attribute_fields.iterrows():
        col = row["field"]
        if col not in df.columns:
            continue
        filter_type = row["filter type"]
        if filter_type == "dropdown/selection":
            unique_vals = sorted(df[col].dropna().unique())
            selected_vals = st.multiselect(f"{col}", unique_vals)
            if selected_vals:
                filters[col] = selected_vals
        elif filter_type == "min/max boxes in one line":
            min_val = df[col].min()
            max_val = df[col].max()
            selected_range = st.slider(f"{col}", int(min_val), int(max_val), (int(min_val), int(max_val)))
            filters[col] = selected_range

# Apply filters
filtered_df = df.copy()
for col, val in filters.items():
    if col == "evolution":
        filtered_df = filtered_df[filtered_df[col] == val]
    elif isinstance(val, list):
        if col in ["playstylesPlus", "playstyles"]:
            # Special handling: players must have ALL selected styles across playstylesPlus and playstyles
            def has_all_styles(row):
                combined_styles = []
                if isinstance(row.get("playstylesPlus"), list):
                    combined_styles.extend(row.get("playstylesPlus"))
                if isinstance(row.get("playstyles"), list):
                    combined_styles.extend(row.get("playstyles"))
                return all(v in combined_styles for v in val)

            filtered_df = filtered_df[filtered_df.apply(has_all_styles, axis=1)]
        else:
            if df[col].apply(lambda x: isinstance(x, list)).any():
                filtered_df = filtered_df[filtered_df[col].apply(lambda x: any(i in x for i in val) if isinstance(x, list) else False)]
            else:
                filtered_df = filtered_df[filtered_df[col].isin(val)]
    elif isinstance(val, tuple) and len(val) == 2:
        filtered_df = filtered_df[filtered_df[col].between(val[0], val[1])]
    else:
        filtered_df = filtered_df[filtered_df[col] == val]

# Sort option
sort_by = st.sidebar.selectbox("Sort By", options=["metaRating_3chem", "metaRating_0chem", "overall"], index=0)

# Columns to display
columns_to_display = [
    "commonName", "archetype", "metaRating_0chem", "metaRating_3chem",
    "bestChemStyle", "accelerateType_chem", "hasRolePlus", "hasRolePlusPlus",
    "skillMoves", "weakFoot", "playstylesPlus", "playstyles", "positions", "foot",
    "bodytype", "accelerateType", "height", "weight", "overall"
] + [col for col in attribute_filter_order if col in filtered_df.columns]

existing_columns = [col for col in columns_to_display if col in filtered_df.columns]
columns_to_hide = ["evolutionIgsBoost", "premiumSeasonPassLevel", "standardSeasonPassLevel", "totalIgsBoost"]
filtered_df = filtered_df.drop(columns=[col for col in columns_to_hide if col in filtered_df.columns], errors="ignore")
remaining_columns = [col for col in filtered_df.columns if col not in existing_columns]
filtered_df = filtered_df[existing_columns + remaining_columns]

def format_boolean(value):
    if value is True:
        return "✅"
    elif value is False:
        return "❌"
    else:
        return ""

filtered_df["hasRolePlus"] = filtered_df["hasRolePlus"].apply(format_boolean)
filtered_df["hasRolePlusPlus"] = filtered_df["hasRolePlusPlus"].apply(format_boolean)

# Display
st.title("El Mostashar Player Database - Rady Club")
st.markdown(f"### Showing {filtered_df['player_origin_id'].nunique()} unique players")
filtered_df = filtered_df.drop(columns=["__true_player_id", "player_origin_id", "debug_index", "__player_id"], errors="ignore")
st.dataframe(filtered_df.sort_values(by=sort_by, ascending=False), use_container_width=True, hide_index=True)