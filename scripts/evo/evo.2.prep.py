import json
import pandas as pd
from pathlib import Path

# Define data directory
data_dir = Path(__file__).resolve().parents[2] / "data"

# Load the JSON file
file_path = data_dir / "evolab_mapped.json"
with open(file_path, "r") as f:
    data = json.load(f)

# Define fields to remove
fields_to_remove = ["overall", "foot", "rolesPlus", "rolesPlusPlus", "positions", "metaRatings"]

# Define one-hot encoding fields and their possible values
accelerate_types = [
    "EXPLOSIVE", "MOSTLY_EXPLOSIVE", "CONTROLLED_EXPLOSIVE", "CONTROLLED",
    "CONTROLLED_LENGTHY", "MOSTLY_LENGTHY", "LENGTHY"
]
bodytypes = [
    "Lean Medium", "Average Medium", "Stocky Medium", "Lean Tall", "Average Tall",
    "Stocky Tall", "Lean Short", "Average Short", "Stocky Short", "Unique"
]

# Define playstyles values
playstyles_list = [
    "Acrobatic", "Aerial", "Anticipate", "Block", "Bruiser", "Chip Shot", "Dead Ball",
    "Finesse Shot", "First Touch", "Flair", "Incisive Pass", "Intercept", "Jockey",
    "Long Ball Pass", "Long Throw", "Low Driven Shot", "Pinged Pass", "Power Header",
    "Power Shot", "Press Proven", "Quick Step", "Rapid", "Relentless", "Slide Tackle",
    "Technical", "Tiki Taka", "Trickster", "Trivela", "Whipped Pass"
]

processed_data = []

for player in data["data"]:
    # Remove unwanted fields
    for field in fields_to_remove:
        player.pop(field, None)

    # Replace rolesPlusPlusArchetypes with archetype
    player["archetype"] = player.pop("rolesPlusPlusArchetypes", [])

    # One-hot encode accelerateType and bodytype
    for acc_type in accelerate_types:
        player[f"accelerateType_{acc_type}"] = int(player.get("accelerateType") == acc_type)
    for body_type in bodytypes:
        player[f"bodytype_{body_type}"] = int(player.get("bodytype") == body_type)

    # Remove original fields after encoding
    player.pop("accelerateType", None)
    player.pop("bodytype", None)

    # Encode playstyles and playstylesPlus
    playstyle_encoding = {ps: 0 for ps in playstyles_list}
    for ps in player.get("playstyles", []):
        if ps in playstyle_encoding:
            playstyle_encoding[ps] = 1
    for ps in player.get("playstylesPlus", []):
        if ps in playstyle_encoding:
            playstyle_encoding[ps] = 2
    player.update(playstyle_encoding)

    # Remove original playstyles fields
    player.pop("playstyles", None)
    player.pop("playstylesPlus", None)

    processed_data.append(player)

# Normalize the data to create one row per eaId-archetype combination, skipping empty archetypes
expanded_data = []
for player in processed_data:
    archetypes = player.pop("archetype", [])
    if not archetypes:
        continue
    for archetype in archetypes:
        new_player = player.copy()
        new_player["archetype"] = archetype
        expanded_data.append(new_player)

# Convert to DataFrame for inspection
df_processed = pd.DataFrame(expanded_data)

# Explicitly reorder columns as specified
desired_order = [
    "eaId", "commonName", "archetype", "height", "weight",
    "skillMoves", "weakFoot", "attributeAcceleration", "attributeSprintSpeed",
    "attributePositioning", "attributeFinishing", "attributeShotPower",
    "attributeLongShots", "attributeVolleys", "attributePenalties",
    "attributeVision", "attributeCrossing", "attributeFkAccuracy", 
    "attributeShortPassing", "attributeLongPassing", "attributeCurve",
    "attributeAgility", "attributeBalance", "attributeReactions",
    "attributeBallControl", "attributeDribbling", "attributeComposure",
    "attributeInterceptions", "attributeHeadingAccuracy", "attributeDefensiveAwareness", 
    "attributeStandingTackle", "attributeSlidingTackle", 
    "attributeJumping", "attributeStamina", "attributeStrength", "attributeAggression",
    "Finesse Shot", "Chip Shot", "Power Shot", "Dead Ball", "Power Header", "Low Driven Shot",
    "Incisive Pass", "Pinged Pass", "Long Ball Pass", "Tiki Taka", "Whipped Pass",
    "Jockey", "Block", "Intercept", "Anticipate", "Slide Tackle", "Bruiser",
    "Technical", "Rapid", "Flair", "First Touch", "Trickster", "Press Proven",
    "Quick Step",  "Relentless", "Trivela", "Acrobatic", "Long Throw", "Aerial",
    "accelerateType_EXPLOSIVE", "accelerateType_MOSTLY_EXPLOSIVE",
    "accelerateType_CONTROLLED_EXPLOSIVE", "accelerateType_CONTROLLED",
    "accelerateType_CONTROLLED_LENGTHY", "accelerateType_MOSTLY_LENGTHY", "accelerateType_LENGTHY",    
    "bodytype_Lean Medium", "bodytype_Average Medium", "bodytype_Stocky Medium", 
    "bodytype_Lean Tall", "bodytype_Average Tall", "bodytype_Stocky Tall",
    "bodytype_Lean Short", "bodytype_Average Short", "bodytype_Stocky Short",  "bodytype_Unique"
]
# Filter to keep only columns that actually exist in the DataFrame
df_processed = df_processed[[col for col in desired_order if col in df_processed.columns]]

df_processed.to_csv(data_dir / "prediction_ready.csv", index=False)