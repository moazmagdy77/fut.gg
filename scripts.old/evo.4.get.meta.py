import pandas as pd
import json
from pathlib import Path

# Define data directory
data_dir = Path(__file__).resolve().parents[1] / "data"

# Load data
predicted_df = pd.read_csv(data_dir / "predicted_metaratings.csv", dtype={"eaId": int})
predicted_df.drop(columns=["position"], inplace=True, errors="ignore")

with open(data_dir / "evolab_mapped.json", "r") as f:
    evolab_data = json.load(f)

# Convert eaId in evolab_data to int to match CSV
for player in evolab_data["data"]:
    player["eaId"] = int(player["eaId"])

# List of accelerateType columns
accel_columns = [
    "accelerateType_EXPLOSIVE",
    "accelerateType_MOSTLY_EXPLOSIVE",
    "accelerateType_CONTROLLED_EXPLOSIVE",
    "accelerateType_CONTROLLED",
    "accelerateType_CONTROLLED_LENGTHY",
    "accelerateType_MOSTLY_LENGTHY",
    "accelerateType_LENGTHY"
]

# Build mapping: eaId -> list of metaRatings per archetype
ratings_by_eaId = {}

for idx, row in predicted_df.iterrows():
    eaId = row["eaId"]
    archetype = row["archetype"]
    meta_rating = row["predicted_meta_rating"]
    chemstyle = row["chemstyle"]

    if pd.isna(meta_rating):
        continue

    # Recover accelerateType
    accelerateType = None
    for accel_col in accel_columns:
        if accel_col in row and row[accel_col] == 1:
            accelerateType = accel_col.replace("accelerateType_", "")
            break

    if chemstyle == "none":
        ratings_by_eaId.setdefault(eaId, []).append({
            "archetype": archetype,
            "metaR_0chem": round(meta_rating, 2),
            "metaR_3chem": None,
            "bestChemStyle": "None",
            "accelerateType_chem": accelerateType
        })
    else:
        existing = next((r for r in ratings_by_eaId.get(eaId, []) if r["archetype"] == archetype), None)
        if existing:
            if existing["metaR_3chem"] is None or meta_rating > existing["metaR_3chem"]:
                existing["metaR_3chem"] = round(meta_rating, 2)
                existing["bestChemStyle"] = chemstyle

# Inject ratings into the JSON structure
for player in evolab_data["data"]:
    player["metaRatings"] = ratings_by_eaId.get(player["eaId"], [])

# Save output
with open(data_dir / "evolab_meta.json", "w") as outfile:
    json.dump(evolab_data, outfile, indent=2)