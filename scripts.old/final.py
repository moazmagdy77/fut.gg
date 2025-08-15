import json
from pathlib import Path
from collections import defaultdict, OrderedDict

# Define data directory
data_dir = Path(__file__).resolve().parents[1] / "data"

# Load maps.json
maps_path = data_dir / "maps.json"
with open(maps_path) as f:
    maps = json.load(f)

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

# Prepare reverse mapping: text value -> ID
roles_text_to_id = {v: k for k, v in maps["rolesPlusPlus"].items()}

# Add all unique evolab_meta players first
for player in evolab_meta:
    combined_players.append(player)
    player["evolution"] = True

    player_roles_plus = player.get("rolesPlus", [])
    player_roles_plusplus = player.get("rolesPlusPlus", [])
    player_roles_archetypes = set()
    player_rolesplusplus_archetypes = set()

    for role_text in player_roles_plus:
        role_id = roles_text_to_id.get(role_text)
        if role_id:
            mapped_archetype = maps["rolesPlusPlusArchetype"].get(role_id)
            if mapped_archetype:
                player_roles_archetypes.add(mapped_archetype)

    for role_text in player_roles_plusplus:
        role_id = roles_text_to_id.get(role_text)
        if role_id:
            mapped_archetype = maps["rolesPlusPlusArchetype"].get(role_id)
            if mapped_archetype:
                player_rolesplusplus_archetypes.add(mapped_archetype)

    for meta_rating in player.get("esMetaRatings", []):  # keep for checking role flags
        archetype = meta_rating.get("archetype")
        meta_rating["hasRolePlus"] = archetype in player_roles_archetypes
        meta_rating["hasRolePlusPlus"] = archetype in player_rolesplusplus_archetypes

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
            "gkFacePositioning", "isUserEvolutions", "isEvoLabPlayerItem", "shirtNumber", "premiumSeasonPassLevel", "standardSeasonPassLevel",
            "metarankStr", "totalFaceStats", "totalIgs", "archetype", "evolutionIgsBoost", "totalIgsBoost"
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
            "playstylesMapped": "PS",
            "playstylesPlusMapped": "PS+",
            "rolesPlusMapped": "roles+",
            "rolesPlusPlusMapped": "roles++"
        }
        #combine positionMapped and alternativePositionIdsMapped into 'positions'
        for old_field, new_field in field_renames.items():
            if old_field in player:
                player[new_field] = player.pop(old_field)

        position_to_roles = maps["positionToRole"]
        role_to_archetype = maps["roleToArchetype"]
        chemstyle_boosts = {c["name"]: c for c in maps["ChemistryStylesBoosts"]}

        valid_roles = []
        for pos in positions:
            valid_roles.extend(position_to_roles.get(pos, []))

        gg_meta = player.get("ggMetaRatings", [])
        gg_by_role = defaultdict(list)
        for item in gg_meta:
            gg_by_role[item["role"]].append(item)

        # Keep only the highest ggMeta per role
        gg_by_role = {
            role: sorted(entries, key=lambda x: x.get("score", 0), reverse=True)[0]
            for role, entries in gg_by_role.items()
        }

        normalized_meta = []
        for role in sorted(set(valid_roles)):
            archetype = role_to_archetype.get(role)
            es_entry = next((x for x in player.get("esMetaRatings", []) if x.get("archetype") == archetype), {})

            gg_entry = gg_by_role.get(role, {})
            chemstyle_name = gg_entry.get("chemistryStyle")
            chemstyle_boost = {}
            if chemstyle_name:
                chemstyle_boost = next((c["threeChemistryModifiers"] for c in maps["ChemistryStylesBoosts"] if c["name"] == chemstyle_name), {})

            # Apply chemstyle boost
            accel = player.get("attributeAcceleration", 0) + chemstyle_boost.get("attributeAcceleration", 0)
            agility = player.get("attributeAgility", 0) + chemstyle_boost.get("attributeAgility", 0)
            strength = player.get("attributeStrength", 0) + chemstyle_boost.get("attributeStrength", 0)
            height = player.get("height", 0)

            # Compute acceleration type
            if (agility - strength) >= 20 and accel >= 80 and height <= 175:
                accel_type = "EXPLOSIVE"
            elif (agility - strength) >= 12 and accel >= 80 and height <= 182:
                accel_type = "MOSTLY_EXPLOSIVE"
            elif (agility - strength) >= 4 and accel >= 70 and height <= 182:
                accel_type = "CONTROLLED_EXPLOSIVE"
            elif (strength - agility) >= 20 and strength >= 80 and height >= 188:
                accel_type = "LENGTHY"
            elif (strength - agility) >= 12 and strength >= 75 and height >= 183:
                accel_type = "MOSTLY_LENGTHY"
            elif (strength - agility) >= 4 and strength >= 65 and height >= 181:
                accel_type = "CONTROLLED_LENGTHY"
            else:
                accel_type = "CONTROLLED"

            normalized_meta.append(OrderedDict([
                ("role", role),
                ("esMetaSub", es_entry.get("metaR_0chem")),
                ("esMetaChem", es_entry.get("metaR_3chem")),
                ("esChemS", es_entry.get("bestChemStyle")),
                ("accelTypeEsChem", es_entry.get("accelerateType_chem")),
                ("ggMeta", gg_entry.get("score")),
                ("ggRank", gg_entry.get("rank")),
                ("ggChemS", chemstyle_name),
                ("accelTypeGgChem", accel_type)
            ]))

        player["finalMetaRatings"] = normalized_meta


        player.pop("esMetaRatings", None)
        player.pop("ggMetaRatings", None)

        player_roles_plus = player.get("rolesPlus", [])
        player_roles_plusplus = player.get("rolesPlusPlus", [])
        player_roles_archetypes = set()
        player_rolesplusplus_archetypes = set()

        for role_text in player_roles_plus:
            role_id = roles_text_to_id.get(role_text)
            if role_id:
                mapped_archetype = maps["rolesPlusPlusArchetype"].get(role_id)
                if mapped_archetype:
                    player_roles_archetypes.add(mapped_archetype)

        for role_text in player_roles_plusplus:
            role_id = roles_text_to_id.get(role_text)
            if role_id:
                mapped_archetype = maps["rolesPlusPlusArchetype"].get(role_id)
                if mapped_archetype:
                    player_rolesplusplus_archetypes.add(mapped_archetype)

        for meta_rating in player.get("finalMetaRatings", []):
            meta_rating.pop("hasRolePlus", None)
            meta_rating.pop("hasRolePlusPlus", None)

        player["evolution"] = False
        combined_players.append(player)

# Output combined list
output_path = data_dir / "final.json"
with open(output_path, "w") as f:
    json.dump(combined_players, f, indent=2)

output_path