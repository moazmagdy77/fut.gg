import pandas as pd
import json
from pathlib import Path

# Define data directory
data_dir = Path(__file__).resolve().parents[2] / "data"

# Reload data after kernel reset
predicted_df = pd.read_csv(data_dir / "predicted_metaratings.csv")

with open(data_dir / "evolab_mapped.json", "r") as f:
    evolab_data = json.load(f)

# Build mapping: eaId -> list of metaRatings per archetype
ratings_by_eaId = {}
for _, row in predicted_df.iterrows():
    eaId = row["eaId"]
    archetype = row["archetype"]
    rating = row["predicted_meta_rating"]
    if pd.notnull(rating):
        ratings_by_eaId.setdefault(eaId, []).append({
            "archetype": archetype,
            "metaRating": rating
        })

# Inject ratings into the JSON structure
for player in evolab_data["data"]:
    player["metaRatings"] = ratings_by_eaId.get(player["eaId"], [])

with open(data_dir / "evolab_meta.json", "w") as outfile:
    json.dump(evolab_data, outfile, indent=2)