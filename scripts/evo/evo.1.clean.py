import json
from pathlib import Path

# Define data directory
data_dir = Path(__file__).resolve().parents[2] / "data"

# Load files
evolab_path = data_dir / "evolab.json"
maps_path = data_dir / "maps.json"

with open(evolab_path, "r", encoding="utf-8") as f:
    evolab = json.load(f)

with open(maps_path, "r", encoding="utf-8") as f:
    maps = json.load(f)

# Load club_players.json
club_players_path = data_dir / "club_players.json"
with open(club_players_path, "r", encoding="utf-8") as f:
    club_players_raw = json.load(f)

# Create a lookup dict using eaId as key
club_players_lookup = {
    p["data"]["eaId"]: p["data"].get("accelerateType")
    for p in club_players_raw
    if "eaId" in p["data"]
}

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

    # Replace item content with just the mapped playerItemDefinition
    item.clear()
    item.update(player_def)

    # Rename bodytypeCode to bodytype
    if "bodytypeCode" in item:
        item["bodytype"] = item.pop("bodytypeCode")

    # Combine position and alternativePositionIds into positions
    position = item.pop("position", None)
    alt_positions = item.pop("alternativePositionIds", [])
    item["positions"] = list({pos for pos in ([position] if position else []) + alt_positions})

    # Add empty metaRatings field
    item["metaRatings"] = []

# Fill in missing accelerateType fields
for item in evolab["data"]:
    ea_id = item.get("eaId")
    if item.get("accelerateType") is None and ea_id in club_players_lookup:
        item["accelerateType"] = club_players_lookup[ea_id]

# Save the updated result to a new JSON file
output_path = data_dir / "evolab_mapped.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(evolab, f, ensure_ascii=False, indent=2)

output_path