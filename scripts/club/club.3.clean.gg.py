import json
from pathlib import Path

# Define data directory
data_dir = Path(__file__).resolve().parents[2] / "data"

# Load data
player_data_path = data_dir / "club_players.json"
maps_path = data_dir / "maps.json"
output_path = data_dir / "enhanced_club_players.json"

with player_data_path.open() as f:
    raw_players = json.load(f)

with maps_path.open() as f:
    maps = json.load(f)

# Fields to delete from player data
fields_to_remove = [
    "game", "slug", "basePlayerSlug", "gender", "searchableName", "dateOfBirth", "attackingWorkrate",
    "defensiveWorkrate", "nationEaId", "leagueEaId", "clubEaId", "uniqueClubEaId",
    "uniqueClubSlug", "rarityEaId", "raritySquadId", "guid", "accelerateTypes", "hasDynamic",
    "renderOnlyAsHtml", "createdAt", "isHidden", "previousVersionsIds", "imagePath",
    "simpleCardImagePath", "futggCardImagePath", "cardImagePath", "shareImagePath",
    "socialImagePath", "targetFacePace", "targetFaceShooting", "targetFacePassing",
    "targetFaceDribbling", "targetFaceDefending", "targetFacePhysicality", "isOnMarket",
    "isUntradeable", "sbcSetEaId", "sbcChallengeEaId", "objectiveGroupEaId",
    "objectiveGroupObjectiveEaId", "objectiveCampaignLevelId", "campaignProps", "contentTypeId",
    "numberOfEvolutions", "blurbText", "smallBlurbText", "upgrades", "hasPrice", "trackerId",
    "liveHubTrackerId", "playerScore", "coinCost", "pointCost", "onLoanFromClubEaId",
    "isSbcItem", "isObjectiveItem", "firstName", "lastName"
]

# Function to map numeric values using provided maps
def apply_mappings(player, maps):
    new_fields = {}
    for key, mapping in maps.items():
        if key in player:
            value = player[key]
            if isinstance(value, list):
                new_fields[key + "Mapped"] = [mapping.get(str(v), v) for v in value]
            else:
                new_fields[key + "Mapped"] = mapping.get(str(value), value)
    return new_fields

# Process players
cleaned_players = []

for player_wrapper in raw_players:
    player = player_wrapper["data"]

    # Skip if rolesPlusPlus is missing or empty
    if not player.get("rolesPlusPlus"):
        continue

    mapped = apply_mappings(player, maps)

    # Add unique archetypes from rolesPlusPlus
    roles_ids = player.get("rolesPlusPlus", [])
    archetypes = list(set(
        maps["rolesPlusPlusArchetype"].get(str(rid))
        for rid in roles_ids
        if str(rid) in maps["rolesPlusPlusArchetype"]
    ))

    # Skip if no valid archetypes found after mapping
    if not archetypes:
        continue

    player["archetype"] = archetypes

    for field in fields_to_remove:
        player.pop(field, None)

    player.update(mapped)
    cleaned_players.append(player)

# Save cleaned data
with output_path.open("w") as f:
    json.dump(cleaned_players, f, indent=2)

output_path.name