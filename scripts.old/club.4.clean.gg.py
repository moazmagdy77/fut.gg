import json
from pathlib import Path

# Define data directory
data_dir = Path(__file__).resolve().parents[1] / "data"

# Load data
player_data_path = data_dir / "club_players.json"
maps_path = data_dir / "maps.json"
output_path = data_dir / "enhanced_club_players.json"

with player_data_path.open() as f:
    raw_players = json.load(f)

metarank_path = data_dir / "club_metarank.json"
with metarank_path.open() as f:
    metarank_data = json.load(f)
metarank_lookup = {entry["data"]["eaId"]: entry["data"]["scores"] for entry in metarank_data}

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

for i, player_wrapper in enumerate(raw_players):
    if "data" not in player_wrapper:
        print(f"Skipping index {i}, no 'data' key: {player_wrapper}")
        continue

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

    ea_id = player.get("eaId")
    if ea_id in metarank_lookup:
        raw_scores = metarank_lookup[ea_id]
        # Apply role and chemStyle mapping from maps.json
        role_map = maps.get("role", {})
        chem_map = maps.get("chemistryStyle", {})

        def map_score(score):
            return {
                "role": role_map.get(str(score["role"]), score["role"]),
                "chemistryStyle": chem_map.get(str(score["chemistryStyle"]), score["chemistryStyle"]),
                "score": score["score"],
                "rank": score["rank"],
                "isPlus": score["isPlus"],
                "isPlusPlus": score["isPlusPlus"]
            }

        player["ggMetaRatings"] = [map_score(s) for s in raw_scores]

    for field in fields_to_remove:
        player.pop(field, None)

    player.update(mapped)
    cleaned_players.append(player)

# Save cleaned data
with output_path.open("w") as f:
    json.dump(cleaned_players, f, indent=2)

output_path.name