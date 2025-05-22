import json
import pandas as pd
from pathlib import Path

# Define data directory
data_dir = Path(__file__).resolve().parents[1] / "data"

# Load the JSON file
file_path = data_dir / "evolab_mapped.json"
with open(file_path, "r") as f:
    data = json.load(f)

# Define fields to remove
fields_to_remove = ["overall", "foot", "rolesPlus", "rolesPlusPlus", "metaRatings"]

# Define one-hot encoding fields and their possible values
accelerate_types = [
    "EXPLOSIVE", "MOSTLY_EXPLOSIVE", "CONTROLLED_EXPLOSIVE", "CONTROLLED",
    "CONTROLLED_LENGTHY", "MOSTLY_LENGTHY", "LENGTHY"
]
bodytypes = [
    "Lean Medium", "Average Medium", "Stocky Medium", "Lean Tall", "Average Tall",
    "Stocky Tall", "Lean Short", "Average Short", "Stocky Short", "Unique"
]

playstyles_list = [
    "Acrobatic", "Aerial", "Anticipate", "Block", "Bruiser", "Chip Shot", "Dead Ball",
    "Finesse Shot", "First Touch", "Flair", "Incisive Pass", "Intercept", "Jockey",
    "Long Ball Pass", "Long Throw", "Low Driven Shot", "Pinged Pass", "Power Header",
    "Power Shot", "Press Proven", "Quick Step", "Rapid", "Relentless", "Slide Tackle",
    "Technical", "Tiki Taka", "Trickster", "Trivela", "Whipped Pass"
]

# Hardcoded position to archetypes map
position_to_archetypes = {
    "GK": ["goalkeeper", "sweeper_keeper"],
    "RB": ["fullback", "falseback", "wingback", "attacking_wingback"],
    "LB": ["fullback", "falseback", "wingback", "attacking_wingback"],
    "CB": ["defender", "stopper", "ball_playing_defender"],
    "CDM": ["holding", "centre_half", "deep_lying_playmaker", "wide_half"],
    "CM": ["box_to_box", "holding", "deep_lying_playmaker", "playmaker", "half_winger"],
    "CAM": ["playmaker", "shadow_striker", "half_winger", "classic_ten"],
    "RM": ["winger", "wide_midfielder", "wide_playmaker", "inside_forward"],
    "LM": ["winger", "wide_midfielder", "wide_playmaker", "inside_forward"],
    "RW": ["winger", "inside_forward", "wide_playmaker"],
    "LW": ["winger", "inside_forward", "wide_playmaker"],
    "ST": ["advanced_forward", "poacher", "false_nine", "target_forward"]
}

processed_data = []

for player in data["data"]:
    # Remove unwanted fields
    for field in fields_to_remove:
        player.pop(field, None)

    # One-hot encode accelerateType and bodytype
    for acc_type in accelerate_types:
        player[f"accelerateType_{acc_type}"] = int(player.get("accelerateType") == acc_type)
    for body_type in bodytypes:
        player[f"bodytype_{body_type}"] = int(player.get("bodytype") == body_type)

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

    player.pop("playstyles", None)
    player.pop("playstylesPlus", None)

    processed_data.append(player)

import copy
with open(data_dir / "maps.json", "r") as f:
    maps = json.load(f)
chemstyle_boosts = {str(c["id"]): c for c in maps["chemistryStylesBoosts"]}

expanded_data = []
for player in processed_data:
    positions = player.pop("positions", [])
    if not positions:
        continue
    for position in positions:
        archetypes = position_to_archetypes.get(position, [])
        for archetype in archetypes:
            for chemstyle_id, chemstyle in chemstyle_boosts.items():
                boosted_player = copy.deepcopy(player)
                # Apply boost
                for attr, boost in chemstyle["threeChemistryModifiers"].items():
                    if attr in boosted_player:
                        boosted_player[attr] = min(boosted_player[attr] + boost, 99)
                boosted_player["position"] = position
                boosted_player["archetype"] = archetype
                boosted_player["chemstyle"] = chemstyle["name"]
                expanded_data.append(boosted_player)

for player in processed_data:
    positions = player.get("positions", [])
    if not positions:
        continue
    for position in positions:
        archetypes = position_to_archetypes.get(position, [])
        for archetype in archetypes:
            chem0_player = copy.deepcopy(player)
            chem0_player["position"] = position
            chem0_player["archetype"] = archetype
            chem0_player["chemstyle"] = "None"
            expanded_data.append(chem0_player)

# Convert to DataFrame
df_processed = pd.DataFrame(expanded_data)

# Desired column order
desired_order = [
    "eaId", "commonName", "position", "archetype", "chemstyle", "height", "weight",
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

# Reorder columns if they exist
df_processed = df_processed[[col for col in desired_order if col in df_processed.columns]]

df_processed.drop_duplicates(subset=["eaId", "position", "archetype", "chemstyle"], inplace=True)

# Save
df_processed.to_csv(data_dir / "prediction_ready.csv", index=False)