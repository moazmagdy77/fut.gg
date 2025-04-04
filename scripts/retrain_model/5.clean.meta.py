import json
import pandas as pd
from pathlib import Path

# Set data directory path
data_dir = Path(__file__).resolve().parents[2] / "data"

# --- STEP 1: Load & clean original players data ---
with open(data_dir / "players_with_meta.json") as f:
    players = json.load(f)

# Filter out goalkeepers
players = [
    player for player in players
    if 'GK' not in (player.get("alternativePositionIdsMapped") or []) and player.get("positionMapped") != 'GK'
]

# Remove unwanted fields
fields_to_remove = [
    "playerType", "id", "evolutionId", "basePlayerEaId", "cosmeticEvolutionId", "partialEvolutionId",
    "foot", "position", "alternativePositionIds", "playstyles", "playstylesPlus", "rolesPlus",
    "rolesPlusPlus", "url", "bodytypeCode", "isRealFace", "facePace", "faceShooting",
    "facePassing", "faceDribbling", "faceDefending", "facePhysicality", "gkFaceDiving",
    "gkFaceHandling", "gkFaceKicking", "gkFaceReflexes", "gkFaceSpeed", "gkFacePositioning",
    "isUserEvolutions", "isEvoLabPlayerItem", "shirtNumber", "totalFaceStats", "attributeGkDiving",
    "attributeGkHandling", "attributeGkKicking", "attributeGkReflexes", "attributeGkPositioning", "totalIgs"
]

for player in players:
    # Keep only metaRating with chemistry = 0
    if isinstance(player.get("metaRatings"), list):
        filtered = []
        for meta in player["metaRatings"]:
            for rating in meta.get("ratings", []):
                if rating.get("chemistry") == 0:
                    filtered.append({
                        "archetype": meta["archetype"],
                        "metaRating": rating["metaRating"]
                    })
                    break
        player["metaRatings"] = filtered
    # Rename bodytypeCode to bodytype (already covered above)
    if "bodytypeCodeMapped" in player:
        player["bodytype"] = player.pop("bodytypeCodeMapped")

    # Remove unwanted fields
    for field in fields_to_remove:
        player.pop(field, None)

# --- STEP 2: Build flat dataset from cleaned players ---
attribute_cols = [k for k in players[0].keys() if k.startswith("attribute")]
categorical_cols = ["accelerateType", "bodytype"]
numeric_cols = ["height", "weight", "skillMoves", "weakFoot"]

# Gather all playstyles
all_playstyles = set()
for player in players:
    all_playstyles.update(player.get("playstylesMapped", []))
    all_playstyles.update(player.get("playstylesPlusMapped", []))
all_playstyles = sorted(list(all_playstyles))

def encode_playstyles(player):
    base = player.get("playstylesMapped", [])
    plus = player.get("playstylesPlusMapped", [])
    return {style: 2 if style in plus else 1 if style in base else 0 for style in all_playstyles}

rows = []
dropped_rows = 0

for player in players:
    if player.get("weight") is None or player.get("height") is None:
        dropped_rows += 1
        continue

    base_row = {
        "height": player.get("height"),
        "weight": player.get("weight"),
        "skillMoves": player.get("skillMoves"),
        "weakFoot": player.get("weakFoot"),
        "accelerateType": player.get("accelerateType"),
        "bodytype": player.get("bodytype")
    }

    for attr in attribute_cols:
        base_row[attr] = player.get(attr)

    base_row.update(encode_playstyles(player))

    for rating in player.get("metaRatings", []):
        row = base_row.copy()
        row["archetype"] = rating["archetype"]
        row["metaRating"] = rating["metaRating"]
        rows.append(row)

print(f"❌ Dropped rows due to missing height/weight: {dropped_rows}")

df = pd.DataFrame(rows)

# One-hot encode categoricals
df = pd.get_dummies(df, columns=categorical_cols, drop_first=False, dtype=int)

# Drop remaining NaNs
before = len(df)
df.dropna(inplace=True)
after = len(df)
print(f"❌ Dropped rows with NaN after encoding: {before - after}")

# Desired column order
ordered_columns = [
    "archetype", "metaRating", "height", "weight",
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
    "bodytype_Lean Short", "bodytype_Average Short", "bodytype_Stocky Short", "bodytype_Unique"
]

# Keep only columns that exist in the DataFrame to avoid KeyError
final_columns = [col for col in ordered_columns if col in df.columns]
missing_columns = [col for col in ordered_columns if col not in df.columns]
if missing_columns:
    print("⚠️ Missing columns that will not be included:", missing_columns)

# Reorder DataFrame
df = df[final_columns]

df.to_csv(data_dir / "training_dataset.csv", index=False)
print("✅ Final dataset saved as training_dataset.csv")