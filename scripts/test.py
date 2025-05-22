import json
from pathlib import Path

# File paths
INPUT_FILE = Path(__file__).resolve().parent / '../data'/"evolab.json"
MAPS_FILE = Path(__file__).resolve().parent / '../data'/"maps.json"
OUTPUT_FILE = Path(__file__).resolve().parent / '../data'/"output.json"

# Fields to remove
FIELDS_TO_REMOVE = [
    "id", "pathHash", "evolutionPath", "numberOfEvolutions", "evolutionId", "playerType", "game", "id", 
    "evolutionId", "cosmeticEvolutionId", "partialEvolutionId", "basePlayerEaId", "basePlayerSlug",
    "gender", "slug", "firstName", "lastName", "nickname", "searchableName", "dateOfBirth", "attackingWorkrate",
    "defensiveWorkrate", "nationEaId", "leagueEaId", "clubEaId", "uniqueClubEaId", "uniqueClubSlug",
    "rarityEaId", "raritySquadId", "guid", "accelerateType", "accelerateTypes", "hasDynamic", "url",
    "renderOnlyAsHtml", "isRealFace", "isHidden", "previousVersionsIds", "imagePath", "simpleCardImagePath",
    "futggCardImagePath", "cardImagePath", "shareImagePath", "socialImagePath", 
    "facePace", "faceShooting", "facePassing", "faceDribbling", "faceDefending",
    "facePhysicality", "targetFacePace", "targetFaceShooting", "targetFacePassing",
    "targetFaceDribbling", "targetFaceDefending", "targetFacePhysicality", "gkFaceDiving",
    "gkFaceHandling", "gkFaceKicking", "gkFaceReflexes", "gkFaceSpeed", "gkFacePositioning",
    "isOnMarket", "isUntradeable", "sbcSetEaId", "sbcChallengeEaId", "objectiveGroupEaId",
    "objectiveGroupObjectiveEaId", "objectiveCampaignLevelId", "campaignProps", "contentTypeId",
    "numberOfEvolutions", "blurbText", "smallBlurbText", "upgrades", "hasPrice", "trackerId",
    "liveHubTrackerId", "isUserEvolutions", "isEvoLabPlayerItem", "totalIgsBoost",
    "evolutionIgsBoost", "playerScore", "coinCost", "pointCost", "shirtNumber", "onLoanFromClubEaId",
    "premiumSeasonPassLevel", "standardSeasonPassLevel", "metarankStr", "ggRating", "ggRatingPos",
    "isSbcItem", "isObjectiveItem", "totalFaceStats", "totalIgs", "wasUpgraded", "totalTrainingTime",
    "maxTimeToStart", "totalIgsBoost", "evolutionIgsBoost"
]

def calculate_acceleration_type(accel, agility, strength, height):
    if not all(isinstance(x, (int, float)) for x in [accel, agility, strength, height]):
        return "CONTROLLED"
    if (agility - strength) >= 20 and accel >= 80 and height <= 175 and agility >= 80:
        return "EXPLOSIVE"
    elif (agility - strength) >= 12 and accel >= 80 and height <= 182 and agility >= 70:
        return "MOSTLY_EXPLOSIVE"
    elif (agility - strength) >= 4 and accel >= 70 and height <= 182 and agility >= 65:
        return "CONTROLLED_EXPLOSIVE"
    elif (strength - agility) >= 20 and strength >= 80 and height >= 188 and accel >= 55:
        return "LENGTHY"
    elif (strength - agility) >= 12 and strength >= 75 and height >= 183 and accel >= 55:
        return "MOSTLY_LENGTHY"
    elif (strength - agility) >= 4 and strength >= 65 and height >= 181 and accel >= 40:
        return "CONTROLLED_LENGTHY"
    return "CONTROLLED"

# Load input data
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Load maps
with open(MAPS_FILE, 'r', encoding='utf-8') as f:
    maps = json.load(f)

# Build chemistry boost map
chem_boost_map = {
    x["name"].lower(): x["threeChemistryModifiers"]
    for x in maps.get("ChemistryStylesBoosts", [])
}

# Process each player
for item in data.get("data", []):
    player = item.get("playerItemDefinition", {})

    # Clean fields
    for field in FIELDS_TO_REMOVE:
        player.pop(field, None)

    # Extract base attributes
    acc = player.get("attributeAcceleration", 0)
    agi = player.get("attributeAgility", 0)
    strg = player.get("attributeStrength", 0)
    height = player.get("height", 0)

    # Parse ggRatingStr
    gg_str = player.get("ggRatingStr", "")
    gg_entries = gg_str.strip("|").split("||")
    role_scores = {}
    for entry in gg_entries:
        if not entry:
            continue
        try:
            chem_id, role_id, score_str = entry.split(":")
            score = float(score_str)
        except ValueError:
            continue
        if role_id not in role_scores or score > role_scores[role_id]["ggMeta"]:
            role_scores[role_id] = {
                "role": maps["role"].get(role_id, f"UnknownRole({role_id})"),
                "ggMeta": round(score, 2),
                "ggChemStyle": maps["ggChemistryStyleNames"].get(chem_id, f"UnknownChem({chem_id})")
            }

    # Compose metaRatings
    metaRatings = []
    for role_id, entry in role_scores.items():
        boosts = chem_boost_map.get(entry["ggChemStyle"].lower(), {})
        accel_type = calculate_acceleration_type(
            acc + boosts.get("attributeAcceleration", 0),
            agi + boosts.get("attributeAgility", 0),
            strg + boosts.get("attributeStrength", 0),
            height
        )
        metaRatings.append({
            "role": entry["role"],
            "ggMeta": entry["ggMeta"],
            "ggChemStyle": entry["ggChemStyle"],
            "ggAccelType": accel_type
        })

    player["metaRatings"] = metaRatings
    player.pop("ggRatingStr", None)

# Save output
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)