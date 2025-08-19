import json
from pathlib import Path

# Define data directory
data_dir = Path(__file__).resolve().parents[1] / "data"

# Load files
evolab_path = data_dir / "evolab.json"
maps_path = data_dir / "maps.json"

with open(evolab_path, "r", encoding="utf-8") as f:
    evolab = json.load(f)

with open(maps_path, "r", encoding="utf-8") as f:
    maps = json.load(f)

# Fields to map and their corresponding mapping dicts
fields_to_map = {
    "position": maps["position"],
    "alternativePositionIds": maps["alternativePositionIds"],
    "foot": maps["foot"],
    "bodytypeCode": maps["bodytypeCode"],
    "playstyles": maps["playstyles"],
    "playstylesPlus": maps["playstylesPlus"],
    "rolesPlus": maps["rolesPlus"],
    "rolesPlusPlus": maps["rolesPlusPlus"]
}

# Function to map a value using a mapping dict
def map_value(value, mapping):
    if isinstance(value, list):
        return [mapping.get(str(v), v) for v in value]
    else:
        return mapping.get(str(value), value)

# Process each player item definition
for item in evolab["data"]:
    player_def = item.get("playerItemDefinition", {})
    
    # Remove unneeded fields
    fields_to_remove = [
        "playerType", "game", "id", "evolutionId", "cosmeticEvolutionId", "partialEvolutionId", "basePlayerEaId",
        "basePlayerSlug", "gender", "slug", "firstName", "lastName", "nickname", "searchableName", "dateOfBirth",
        "attackingWorkrate", "defensiveWorkrate", "nationEaId", "leagueEaId", "clubEaId", "uniqueClubEaId",
        "uniqueClubSlug", "rarityEaId", "raritySquadId", "guid", "accelerateTypes", "hasDynamic", "url", 
        "renderOnlyAsHtml", "isRealFace", "createdAt", "isHidden", "previousVersionsIds", "imagePath",
        "simpleCardImagePath", "futggCardImagePath", "cardImagePath", "shareImagePath", "socialImagePath",
        "attributeGkDiving", "attributeGkHandling", "attributeGkKicking", "attributeGkReflexes", "attributeGkPositioning",
        "facePace", "faceShooting", "facePassing", "faceDribbling", "faceDefending", "facePhysicality", 
        "targetFacePace", "targetFaceShooting", "targetFacePassing", "targetFaceDribbling", "targetFaceDefending", 
        "targetFacePhysicality", "gkFaceDiving", "gkFaceHandling", "gkFaceKicking", "gkFaceReflexes", "gkFaceSpeed", 
        "gkFacePositioning", "isOnMarket", "isUntradeable", "sbcSetEaId", "sbcChallengeEaId", "objectiveGroupEaId", 
        "objectiveGroupObjectiveEaId", "objectiveCampaignLevelId", "campaignProps", "contentTypeId", "numberOfEvolutions", 
        "blurbText", "smallBlurbText", "upgrades", "hasPrice", "trackerId", "liveHubTrackerId", "isUserEvolutions", 
        "isEvoLabPlayerItem", "playerScore", "coinCost", "pointCost", "shirtNumber", "onLoanFromClubEaId", 
        "isSbcItem", "isObjectiveItem", "totalFaceStats", "totalIgs"
    ]
    for field in fields_to_remove:
        player_def.pop(field, None)

    # Add archetypes for rolesPlusPlus before mapping
    if "rolesPlusPlus" in player_def:
        player_def["rolesPlusPlusArchetypes"] = list({
            maps["rolesPlusPlusArchetype"].get(str(v), None)
            for v in player_def["rolesPlusPlus"]
            if maps["rolesPlusPlusArchetype"].get(str(v), None) is not None
        })

    for field, mapping in fields_to_map.items():
        if field in player_def:
            player_def[field] = map_value(player_def[field], mapping)

    # Rename bodytypeCode to bodytype immediately after mapping
    if "bodytypeCode" in player_def:
        player_def["bodytype"] = player_def.pop("bodytypeCode")

    # Combine position and alternativePositionIds into positions and remove them
    position = player_def.pop("position", None)
    alt_positions = player_def.pop("alternativePositionIds", [])
    player_def["positions"] = list({pos for pos in ([position] if position else []) + alt_positions})

    # Calculate accelerateType based on attributes before clearing item
    accel = player_def.get("attributeAcceleration", 0)
    strength = player_def.get("attributeStrength", 0)
    agility = player_def.get("attributeAgility", 0)
    height = player_def.get("height", 0)

    accelerateType = "CONTROLLED"
    if (agility - strength) >= 20 and accel >= 80 and height <= 175:
        accelerateType = "EXPLOSIVE"
    elif (agility - strength) >= 12 and accel >= 80 and height <= 182:
        accelerateType = "MOSTLY_EXPLOSIVE"
    elif (agility - strength) >= 4 and accel >= 70 and height <= 182:
        accelerateType = "CONTROLLED_EXPLOSIVE"
    elif (strength - agility) >= 20 and strength >= 80 and height >= 188:
        accelerateType = "LENGTHY"
    elif (strength - agility) >= 12 and strength >= 75 and height >= 183:
        accelerateType = "MOSTLY_LENGTHY"
    elif (strength - agility) >= 4 and strength >= 65 and height >= 181:
        accelerateType = "CONTROLLED_LENGTHY"

    player_def["accelerateType"] = accelerateType

    # Replace item content with just the mapped playerItemDefinition
    item.clear()
    item.update(player_def)

    # Add empty metaRatings field
    item["metaRatings"] = []

# Save the updated result to a new JSON file
output_path = data_dir / "evolab_mapped.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(evolab, f, ensure_ascii=False, indent=2)

output_path