import json
from pathlib import Path

# Define data directory
data_dir = Path(__file__).resolve().parents[2] / "data"

# Load the data from the files
club_players_path = data_dir / "club_players_with_meta.json"
evolab_meta_path = data_dir / "evolab_meta.json"

with club_players_path.open() as f:
    club_players = json.load(f)

with evolab_meta_path.open() as f:
    evolab_meta = json.load(f)["data"]

# Create a dictionary mapping eaId to player data for club players
club_players_by_eaid = {player["eaId"]: player for player in club_players}

# Index evolab_meta by eaId for quick lookup
evolab_meta_by_eaid = {player["eaId"]: player for player in evolab_meta}

# Prepare the combined list
combined_players = []
used_eaids = set()

# Add all unique evolab_meta players first
for player in evolab_meta:
    combined_players.append(player)
    player["evolution"] = True
    used_eaids.add(player["eaId"])

    # Remove rolesPlusPlusArchetypes if present
    player.pop("rolesPlusPlusArchetypes", None)

    # Add missing GK attributes with value 0 if they don't exist
    gk_attributes = [
        "attributeGkDiving",
        "attributeGkHandling",
        "attributeGkKicking",
        "attributeGkPositioning",
        "attributeGkReflexes"
    ]
    for attr in gk_attributes:
        if attr not in player:
            player[attr] = 1

# Add only the club players whose eaId is not already in evolab_meta
for player in club_players:
    eaid = player["eaId"]
    if eaid not in used_eaids:
        fields_to_remove = {
            "playerType", "id", "evolutionId", "cosmeticEvolutionId", "partialEvolutionId", "basePlayerEaId", "nickname",
            "foot", "position", "alternativePositionIds", "playstyles", "playstylesPlus", "rolesPlus", "rolesPlusPlus",
            "url", "bodytypeCode", "isRealFace", "facePace", "faceShooting", "facePassing", "faceDribbling", "faceDefending",
            "facePhysicality", "gkFaceDiving", "gkFaceHandling", "gkFaceKicking", "gkFaceReflexes", "gkFaceSpeed",
            "gkFacePositioning", "isUserEvolutions", "isEvoLabPlayerItem", "shirtNumber", "totalFaceStats", "totalIgs",
            "archetype"
        }
        for field in fields_to_remove:
            player.pop(field, None)

        # Combine positionMapped and alternativePositionIdsMapped into 'positions'
        positions = []
        if "positionMapped" in player:
            positions.append(player["positionMapped"])
        if "alternativePositionIdsMapped" in player:
            positions.extend(player["alternativePositionIdsMapped"])
        player["positions"] = positions
        player.pop("positionMapped", None)
        player.pop("alternativePositionIdsMapped", None)

        # Rename specific mapped fields to match evolab_meta format
        field_renames = {
            "bodytypeCodeMapped": "bodytype",
            "footMapped": "foot",
            "playstylesMapped": "playstyles",
            "playstylesPlusMapped": "playstylesPlus",
            "rolesPlusMapped": "rolesPlus",
            "rolesPlusPlusMapped": "rolesPlusPlus"
        }
        for old_field, new_field in field_renames.items():
            if old_field in player:
                player[new_field] = player.pop(old_field)

        player["evolution"] = False
        combined_players.append(player)

# Output combined list
output_path = data_dir / "final.json"
with open(output_path, "w") as f:
    json.dump(combined_players, f, indent=2, sort_keys=True)

output_path