import json
import pandas as pd
from pathlib import Path

# Set data directory path
data_dir = Path(__file__).resolve().parents[1] / "data"

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

roles_archetypes = [
    "goalkeeper", "sweeper_keeper", "fullback", "falseback", "wingback", "attacking_wingback",
    "defender", "stopper", "ball_playing_defender", "holding", "centre_half", "deep_lying_playmaker",
    "wide_half", "box_to_box", "playmaker", "half_winger", "winger", "wide_midfielder",
    "wide_playmaker", "inside_forward", "shadow_striker", "classic_ten", "advanced_forward",
    "poacher", "false_nine", "target_forward"
]

role_mapping = {
    "GK Goalkeeper": "goalkeeper",
    "GK Sweeper Keeper": "sweeper_keeper",
    "RB Fullback": "fullback",
    "RB Falseback": "falseback",
    "RB Wingback": "wingback",
    "RB Attacking Wingback": "attacking_wingback",
    "LB Fullback": "fullback",
    "LB Falseback": "falseback",
    "LB Wingback": "wingback",
    "LB Attacking Wingback": "attacking_wingback",
    "CB Defender": "defender",
    "CB Stopper": "stopper",
    "CB Ball-Playing Defender": "ball_playing_defender",
    "CDM Holding": "holding",
    "CDM Centre-Half": "centre_half",
    "CDM Deep-Lying Playmaker": "deep_lying_playmaker",
    "CDM Wide Half": "wide_half",
    "CM Box to Box": "box_to_box",
    "CM Holding": "holding",
    "CM Deep-Lying Playmaker": "deep_lying_playmaker",
    "CM Playmaker": "playmaker",
    "CM Half-Winger": "half_winger",
    "RM Winger": "winger",
    "RM Wide Midfielder": "wide_midfielder",
    "RM Wide Playmaker": "wide_playmaker",
    "RM Inside Forward": "inside_forward",
    "LM Winger": "winger",
    "LM Wide Midfielder": "wide_midfielder",
    "LM Wide Playmaker": "wide_playmaker",
    "LM Inside Forward": "inside_forward",
    "CAM Playmaker": "playmaker",
    "CAM Shadow Striker": "shadow_striker",
    "CAM Half Winger": "half_winger",
    "CAM Classic 10": "classic_ten",
    "RW Winger": "winger",
    "RW Inside Forward": "inside_forward",
    "RW Wide Playmaker": "wide_playmaker",
    "LW Winger": "winger",
    "LW Inside Forward": "inside_forward",
    "LW Wide Playmaker": "wide_playmaker",
    "ST Advanced Forward": "advanced_forward",
    "ST Poacher": "poacher",
    "ST False 9": "false_nine",
    "ST Target Forward": "target_forward"
}

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

    role_cols = {v: 0 for v in roles_archetypes}
    for r in player.get("rolesPlusMapped", []):
        arch = role_mapping.get(r)
        if arch:
            role_cols[arch] = max(role_cols[arch], 1)
    for r in player.get("rolesPlusPlusMapped", []):
        arch = role_mapping.get(r)
        if arch:
            role_cols[arch] = 2
    base_row.update(role_cols)

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
    "goalkeeper", "sweeper_keeper", "fullback", "falseback", "wingback", "attacking_wingback",
    "defender", "stopper", "ball_playing_defender", "holding", "centre_half", "deep_lying_playmaker",
    "wide_half", "box_to_box", "playmaker", "half_winger", "winger", "wide_midfielder",
    "wide_playmaker", "inside_forward", "shadow_striker", "classic_ten", "advanced_forward",
    "poacher", "false_nine", "target_forward",
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