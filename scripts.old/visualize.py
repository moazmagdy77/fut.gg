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
with open(data_dir / "final.json", "r") as f:
    data = json.load(f)

for idx, p in enumerate(data):
    p["player_origin_id"] = f"{p.get('eaId', 'unknown')}_{idx}"
    p["debug_index"] = idx

df = pd.json_normalize(data)
df["player_origin_id"] = df["player_origin_id"].astype(str)
df["eaId"] = df["eaId"].astype(str)
df["__true_player_id"] = df["eaId"].fillna(df["commonName"])

# Rename attribute columns
df.rename(columns={col: col.replace("attribute", "", 1)[0].lower() + col.replace("attribute", "", 1)[1:] for col in df.columns if col.startswith("attribute")}, inplace=True)

# Explode metaRatings properly
if "finalMetaRatings" in df.columns:
    df["finalMetaRatings"] = df["finalMetaRatings"].apply(lambda x: x if isinstance(x, list) else [{}])
    df = df.explode("finalMetaRatings", ignore_index=True)
    df["finalMetaRatings"] = df["finalMetaRatings"].apply(lambda x: x if isinstance(x, dict) else {})
    df["role"] = df["finalMetaRatings"].apply(lambda x: x.get("role", "N/A"))
    df["esMetaSub"] = df["finalMetaRatings"].apply(lambda x: x.get("esMetaSub", 0))
    df["esMetaChem"] = df["finalMetaRatings"].apply(lambda x: x.get("esMetaChem", 0))
    df["esChemS"] = df["finalMetaRatings"].apply(lambda x: x.get("esChemS", "None"))
    df["accelTypeEsChem"] = df["finalMetaRatings"].apply(lambda x: x.get("accelTypeEsChem", "Unknown"))
    df["ggMeta"] = df["finalMetaRatings"].apply(lambda x: x.get("ggMeta", 0))
    df["ggRank"] = df["finalMetaRatings"].apply(lambda x: x.get("ggRank", 999))
    df["ggChemS"] = df["finalMetaRatings"].apply(lambda x: x.get("ggChemS", "None"))
    df["accelTypeGgChem"] = df["finalMetaRatings"].apply(lambda x: x.get("accelTypeGgChem", "Unknown"))
    df = df.drop(columns=["finalMetaRatings"])

df["height"] = pd.to_numeric(df["height"], errors="coerce")
df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
df["height"] = df["height"].fillna(0).astype(int)
df["weight"] = df["weight"].fillna(0).astype(int)

filters = {}

st.sidebar.header("Filter Players")

# Evolution filter (if exists)
evolution_values = [True, False]
selected_evolution = st.sidebar.selectbox("evolution", options=["All"] + evolution_values)
if selected_evolution != "All":
    filters["evolution"] = selected_evolution

# Role filter
if "role" in df.columns:
    unique_roles = sorted(df["role"].dropna().unique())
    selected_roles = st.sidebar.multiselect("Role", unique_roles)
    if selected_roles:
        filters["role"] = selected_roles

# Overall rating filter
if "overall" in df.columns and not df["overall"].isnull().all():
    min_ovr = int(df["overall"].min())
    max_ovr = int(df["overall"].max())
    left_col, right_col = st.sidebar.columns([1, 1])
    with left_col:
        min_input = st.number_input("Min Overall", min_ovr, max_ovr, value=min_ovr, key="overall_min")
    with right_col:
        max_input = st.number_input("Max Overall", min_ovr, max_ovr, value=max_ovr, key="overall_max")
    if min_input > min_ovr or max_input < max_ovr:
        filters["overall"] = (min_input, max_input)

# Height filter
if "height" in df.columns and not df["height"].isnull().all():
    min_height = int(df["height"].min())
    max_height = int(df["height"].max())
    left_col, right_col = st.sidebar.columns([1, 1])
    with left_col:
        min_input = st.number_input("Min Height (cm)", min_height, max_height, value=min_height, key="height_min")
    with right_col:
        max_input = st.number_input("Max Height (cm)", min_height, max_height, value=max_height, key="height_max")
    if min_input > min_height or max_input < max_height:
        filters["height"] = (min_input, max_input)


# Weight filter
if "weight" in df.columns and not df["weight"].isnull().all():
    min_weight = int(df["weight"].min())
    max_weight = int(df["weight"].max())
    left_col, right_col = st.sidebar.columns([1, 1])
    with left_col:
        min_input = st.number_input("Min Weight (kg)", min_weight, max_weight, value=min_weight, key="weight_min")
    with right_col:
        max_input = st.number_input("Max Weight (kg)", min_weight, max_weight, value=max_weight, key="weight_max")
    if min_input > min_weight or max_input < max_weight:
        filters["weight"] = (min_input, max_input)

# Skill Moves filter
if "skillMoves" in df.columns and not df["skillMoves"].isnull().all():
    min_skill = int(df["skillMoves"].min())
    max_skill = int(df["skillMoves"].max())
    left_col, right_col = st.sidebar.columns([1, 1])
    with left_col:
        min_input = st.number_input("Min Skill Moves", min_skill, max_skill, value=min_skill, key="skillMoves_min")
    with right_col:
        max_input = st.number_input("Max Skill Moves", min_skill, max_skill, value=max_skill, key="skillMoves_max")
    if min_input > min_skill or max_input < max_skill:
        filters["skillMoves"] = (min_input, max_input)

# Weak Foot filter
if "weakFoot" in df.columns and not df["weakFoot"].isnull().all():
    min_weak = int(df["weakFoot"].min())
    max_weak = int(df["weakFoot"].max())
    left_col, right_col = st.sidebar.columns([1, 1])
    with left_col:
        min_input = st.number_input("Min Weak Foot", min_weak, max_weak, value=min_weak, key="weakFoot_min")
    with right_col:
        max_input = st.number_input("Max Weak Foot", min_weak, max_weak, value=max_weak, key="weakFoot_max")
    if min_input > min_weak or max_input < max_weak:
        filters["weakFoot"] = (min_input, max_input)

# Combined PlayStyle and PlayStyle+ filter
if "PS" in df.columns or "PS+" in df.columns:
    combined_styles = set()
    for col in ["PS", "PS+"]:
        if col in df.columns:
            combined_styles.update(x for sublist in df[col].dropna() if isinstance(sublist, list) for x in sublist)
    combined_styles = sorted(combined_styles)
    selected_styles = st.sidebar.multiselect("PlayStyle (PS & PS+)", combined_styles)
    if selected_styles:
        filters["PS_combined"] = selected_styles

# Acceleration Type (Base) filter
if "accelerateType" in df.columns:
    unique_accel_base = sorted(df["accelerateType"].dropna().unique())
    selected_accel_base = st.sidebar.multiselect("Acceleration Type (Base)", unique_accel_base)
    if selected_accel_base:
        filters["accelerateType"] = selected_accel_base

# Acceleration Type (GG) filter
if "accelTypeGgChem" in df.columns:
    unique_accels = sorted(df["accelTypeGgChem"].dropna().unique())
    selected_accels = st.sidebar.multiselect("Acceleration Type (GG)", unique_accels)
    if selected_accels:
        filters["accelTypeGgChem"] = selected_accels

# Acceleration Type (ES) filter
if "accelTypeEsChem" in df.columns:
    unique_accels_es = sorted(df["accelTypeEsChem"].dropna().unique())
    selected_accels_es = st.sidebar.multiselect("Acceleration Type (ES)", unique_accels_es)
    if selected_accels_es:
        filters["accelTypeEsChem"] = selected_accels_es

if "ggMeta" in df.columns and not df["ggMeta"].isnull().all():
    min_gg_meta = float(df["ggMeta"].min())
    max_gg_meta = float(df["ggMeta"].max())
    left_col, right_col = st.sidebar.columns([1, 1])
    with left_col:
        min_input = st.number_input("Min GG Meta", min_gg_meta, max_gg_meta, value=min_gg_meta, key="ggMeta_min")
    with right_col:
        max_input = st.number_input("Max GG Meta", min_gg_meta, max_gg_meta, value=max_gg_meta, key="ggMeta_max")
    if min_input > min_gg_meta or max_input < max_gg_meta:
        filters["ggMeta"] = (min_input, max_input)

if "ggRank" in df.columns and not df["ggRank"].isnull().all():
    min_gg_rank = int(df["ggRank"].min())
    max_gg_rank = int(df["ggRank"].max())
    left_col, right_col = st.sidebar.columns([1, 1])
    with left_col:
        min_input = st.number_input("Min GG Rank", min_gg_rank, max_gg_rank, value=min_gg_rank, key="ggRank_min")
    with right_col:
        max_input = st.number_input("Max GG Rank", min_gg_rank, max_gg_rank, value=max_gg_rank, key="ggRank_max")
    if min_input > min_gg_rank or max_input < max_gg_rank:
        filters["ggRank"] = (min_input, max_input)

with st.sidebar.expander("In-Game Stats", expanded=False):
    for attr in attribute_filter_order:
        if attr in df.columns:
            min_val = int(df[attr].min())
            max_val = int(df[attr].max())
            left_col, right_col = st.columns([1, 1])
            with left_col:
                min_input = st.number_input(f"Min {attr}", min_val, max_val, value=min_val, key=f"{attr}_min")
            with right_col:
                max_input = st.number_input(f"Max {attr}", min_val, max_val, value=max_val, key=f"{attr}_max")
            if min_input > min_val or max_input < max_val:
                filters[attr] = (min_input, max_input)


# Ensure PS and PS+ are explicitly preserved as lists after explode and transformation
if "PS" not in df.columns:
    df["PS"] = [[] for _ in range(len(df))]
if "PS+" not in df.columns:
    df["PS+"] = [[] for _ in range(len(df))]

# Apply filters
filtered_df = df.copy()

# Apply combined PlayStyle filter
values = filters.get("PS_combined", [])

def has_all_styles(row):
    combined_styles = []
    if isinstance(row.get("PS+"), list):
        combined_styles.extend(row.get("PS+"))
    if isinstance(row.get("PS"), list):
        combined_styles.extend(row.get("PS"))
    return all(v in combined_styles for v in values)

if values:
    filtered_df = filtered_df[filtered_df.apply(has_all_styles, axis=1)]

for col, val in filters.items():
    if col == "PS_combined":
        continue  # Already handled earlier, skip to avoid KeyError
    if col == "evolution":
        filtered_df = filtered_df[filtered_df[col] == val]
    elif isinstance(val, list):
        if col in ["PS+", "PS"]:
            # Special handling: players must have ALL selected styles across PS+ and PS
            def has_all_styles(row):
                combined_styles = []
                if isinstance(row.get("PS+"), list):
                    combined_styles.extend(row.get("PS+"))
                if isinstance(row.get("PS"), list):
                    combined_styles.extend(row.get("PS"))
                return all(v in combined_styles for v in val)

            filtered_df = filtered_df[filtered_df.apply(has_all_styles, axis=1)]
        else:
            if df[col].apply(lambda x: isinstance(x, list)).any():
                filtered_df = filtered_df[filtered_df[col].apply(lambda x: any(i in x for i in val) if isinstance(x, list) else False)]
            else:
                filtered_df = filtered_df[filtered_df[col].isin(val)]
    elif isinstance(val, tuple) and len(val) == 2:
        # Only apply height/weight filter if explicitly set (i.e. user has moved the slider)
        if col in ["height", "weight", "overall", "skillMoves", "weakFoot"]:
            # Only filter if the user has changed the slider from the min/max range
            min_val = int(df[col].min())
            max_val = int(df[col].max())
            # If the selected range is not the full available range, apply filter, else skip
            if val[0] > min_val or val[1] < max_val:
                # Exclude zero or less (which are originally nulls)
                filtered_df = filtered_df[(filtered_df[col] >= val[0]) & (filtered_df[col] <= val[1]) & (filtered_df[col] > 0)]
            # else, don't filter (preserve all, including 0/null)
        else:
            filtered_df = filtered_df[filtered_df[col].between(val[0], val[1])]
    else:
        filtered_df = filtered_df[filtered_df[col] == val]

# Columns to display
columns_to_display = [
    "commonName", "role",
    "esMetaSub", "ggMeta",
    "esMetaChem", "ggRank",
    "esChemS", "ggChemS",
    "accelTypeEsChem", "accelTypeGgChem",
    "skillMoves", "weakFoot", "PS+", "PS", "positions", "foot",
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

# Display
st.title("El Mostashar Player Database")
st.markdown(f"### Showing {filtered_df['player_origin_id'].nunique()} unique players")

if filters:
    st.subheader("Active Filters")
    for key, val in filters.items():
        display_key = "PlayStyle" if key == "PS_combined" else key
        if isinstance(val, list):
            for v in val:
                st.markdown(
                    f"<span style='background-color:#333;color:#f5f5f5;padding:4px 8px;margin-right:5px;border-radius:12px;display:inline-block;'>{display_key}: {v}</span>",
                    unsafe_allow_html=True
                )
        elif isinstance(val, tuple):
            st.markdown(
                f"<span style='background-color:#333;color:#f5f5f5;padding:4px 8px;margin-right:5px;border-radius:12px;display:inline-block;'>{display_key}: {val[0]} - {val[1]}</span>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<span style='background-color:#333;color:#f5f5f5;padding:4px 8px;margin-right:5px;border-radius:12px;display:inline-block;'>{display_key}: {val}</span>",
                unsafe_allow_html=True
            )



st.subheader("Top Meta Ratings")
col1, col2, col3 = st.columns(3)

# Top 3 GG Meta
if "ggMeta" in filtered_df.columns and not filtered_df.empty:
    top_gg_df = filtered_df.nlargest(3, "ggMeta")
    with col1:
        st.markdown("**Top GG Meta**")
        for i, (_, row) in enumerate(top_gg_df.iterrows()):
            medal = ["🥇", "🥈", "🥉"][i] if i < 3 else ""
            st.metric(label=f"{medal} {row['commonName']}", value=f'{row["ggMeta"]:.2f}')

# Top 3 ES Meta (3 Chem)
if "esMetaChem" in filtered_df.columns and not filtered_df.empty:
    top_es_df = filtered_df.nlargest(3, "esMetaChem")
    with col2:
        st.markdown("**Top EasySBC Meta (Full Chem)**")
        for i, (_, row) in enumerate(top_es_df.iterrows()):
            medal = ["🥇", "🥈", "🥉"][i] if i < 3 else ""
            st.metric(label=f"{medal} {row['commonName']}", value=f'{row["esMetaChem"]:.2f}')

# Top 3 ES Meta (0 Chem)
if "esMetaSub" in filtered_df.columns and not filtered_df.empty:
    top_es0_df = filtered_df.nlargest(3, "esMetaSub")
    with col3:
        st.markdown("**Top EasySBC Meta (Sub)**")
        for i, (_, row) in enumerate(top_es0_df.iterrows()):
            medal = ["🥇", "🥈", "🥉"][i] if i < 3 else ""
            st.metric(label=f"{medal} {row['commonName']}", value=f'{row["esMetaSub"]:.2f}')

filtered_df = filtered_df.drop(columns=["__true_player_id", "player_origin_id", "debug_index"], errors="ignore")

if filtered_df.empty:
    st.warning("No players found matching the selected filters.")
else:
    st.dataframe(filtered_df.sort_values(by="ggMeta", ascending=False), use_container_width=True, hide_index=True)