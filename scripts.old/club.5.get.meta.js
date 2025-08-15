const fs = require('fs');
const path = require('path');
const axios = require('axios');

const dataDir = path.resolve(__dirname, '..', 'data');

const players = JSON.parse(fs.readFileSync(path.join(dataDir, 'enhanced_club_players.json'), 'utf-8'));
const maps = JSON.parse(fs.readFileSync(path.join(dataDir, 'maps.json'), 'utf-8'));
const chemStyleMap = maps.chemStyles;

const MAX_CONCURRENT = 10;
const DELAY_BETWEEN_BATCHES_MS = 1000;
const MAX_RETRIES = 2;
const HARD_TIMEOUT_MS = 20000;

const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

const fetchMetaRatingsWithRetry = async (player, archetype) => {
  const url = `https://api.easysbc.io/squad-builder/meta-ratings?archetypeId=${archetype}&resourceId=${player.eaId}`;
  console.log(`📦 Fetching meta rating for player ${player.commonName} - Archetype: ${archetype}`);

  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      const result = await Promise.race([
        axios.get(url),
        new Promise((_, reject) => setTimeout(() => reject(new Error('⏰ Hard timeout exceeded')), HARD_TIMEOUT_MS))
      ]);
      console.log(`📬 Received data for ${player.commonName} - ${archetype}:`, result.data);
      return result.data;
    } catch (err) {
      if (attempt < MAX_RETRIES) {
        console.warn(`⚠️ Retry ${attempt + 1} for player ${player.commonName} - Archetype: ${archetype}`);
        await delay(1000);
      } else {
        console.error(`❌ Failed for ${player.commonName} with archetype ${archetype} after ${MAX_RETRIES + 1} attempts:`, err.message);
      }
    }
  }
  return [];
};

const positionToArchetypes = {
  GK: ['goalkeeper', 'sweeper_keeper'],
  RB: ['fullback', 'falseback', 'wingback', 'attacking_wingback'],
  LB: ['fullback', 'falseback', 'wingback', 'attacking_wingback'],
  CB: ['defender', 'stopper', 'ball_playing_defender'],
  CDM: ['holding', 'centre_half', 'deep_lying_playmaker', 'wide_half'],
  CM: ['box_to_box', 'holding', 'deep_lying_playmaker', 'playmaker', 'half_winger'],
  CAM: ['playmaker', 'shadow_striker', 'half_winger', 'classic_ten'],
  RM: ['winger', 'wide_midfielder', 'wide_playmaker', 'inside_forward'],
  LM: ['winger', 'wide_midfielder', 'wide_playmaker', 'inside_forward'],
  RW: ['winger', 'inside_forward', 'wide_playmaker'],
  LW: ['winger', 'inside_forward', 'wide_playmaker'],
  ST: ['advanced_forward', 'poacher', 'false_nine', 'target_forward']
};

(async () => {
  for (let i = 0; i < players.length; i += MAX_CONCURRENT) {
    const batch = players.slice(i, i + MAX_CONCURRENT);

    const results = await Promise.allSettled(
      batch.map(async (player, index) => {
        player.esMetaRatings = [];

        if (!player.eaId) return player;

        const knownArchetypes = new Set(player.archetype || []);

        // Gather additional archetypes based on playable positions
        const playablePositions = [];
        if (player.positionMapped) playablePositions.push(player.positionMapped);
        if (player.alternativePositionIdsMapped && Array.isArray(player.alternativePositionIdsMapped)) {
          playablePositions.push(...player.alternativePositionIdsMapped);
        }

        const extraArchetypes = new Set();
        for (const pos of playablePositions) {
          if (positionToArchetypes[pos]) {
            positionToArchetypes[pos].forEach(arch => extraArchetypes.add(arch));
          }
        }

        const allArchetypes = new Set([...knownArchetypes, ...extraArchetypes]);
        console.log(`✨ Processing ${player.commonName} with archetypes:`, [...allArchetypes]);
        
        await Promise.all([...allArchetypes].map(async (archetype) => {
          const ratings = await fetchMetaRatingsWithRetry(player, archetype);
          console.log(`🔍 Raw ratings for ${player.commonName} - Archetype: ${archetype}:`, ratings);
          if (!Array.isArray(ratings) || ratings.length === 0) {
            console.warn(`⚠️ No ratings received for ${player.commonName} - Archetype: ${archetype}`);
            return;
          }

          const ratingAt0Chem = ratings.find(r => r.chemistry === 0);
          console.log(`➡️ Rating at 0 Chem for ${player.commonName} (${archetype}):`, ratingAt0Chem);
          const metaR_0chem = ratingAt0Chem ? ratingAt0Chem.metaRating : null;

          const ratingsAt3Chem = ratings.filter(r => r.chemistry === 3);
          const bestChemRating = ratingsAt3Chem.find(r => r.isBestChemstyleAtChem === true);
          console.log(`➡️ Best Chem Rating for ${player.commonName} (${archetype}):`, bestChemRating);
          const metaR_3chem = bestChemRating ? bestChemRating.metaRating : null;
          const chemstyleId = bestChemRating ? bestChemRating.chemstyleId : null;

          const bestChemStyleName = chemstyleId && chemStyleMap[chemstyleId] ? chemStyleMap[chemstyleId] : null;
          const chemstyleBoost = chemstyleId && maps.chemistryStylesBoosts.find(c => c.id === Number(chemstyleId));
          console.log(`🧪 Chemstyle boost for ${player.commonName} (${archetype}) [ID: ${chemstyleId}]:`, chemstyleBoost);

          if (
            metaR_0chem === null ||
            metaR_3chem === null ||
            bestChemStyleName === null ||
            Number.isNaN(metaR_0chem) ||
            Number.isNaN(metaR_3chem)
          ) {
            console.warn(`⛔ Skipping ${player.commonName} (${archetype}) due to missing or NaN key data.`, {
              metaR_0chem,
              metaR_3chem,
              bestChemStyleName
            });
            return;
          }

          let accel = player.attributeAcceleration;
          let agility = player.attributeAgility;
          let strength = player.attributeStrength;
          const height = player.height;

          if (chemstyleBoost && chemstyleBoost.threeChemistryModifiers) {
            accel = Math.min(99, accel + (chemstyleBoost.threeChemistryModifiers.attributeAcceleration || 0));
            agility = Math.min(99, agility + (chemstyleBoost.threeChemistryModifiers.attributeAgility || 0));
            strength = Math.min(99, strength + (chemstyleBoost.threeChemistryModifiers.attributeStrength || 0));
          }

          let accelerateType_chem = "CONTROLLED";
          if ((agility - strength) >= 20 && accel >= 80 && height <= 175) {
            accelerateType_chem = "EXPLOSIVE";
          } else if ((agility - strength) >= 12 && accel >= 80 && height <= 182) {
            accelerateType_chem = "MOSTLY_EXPLOSIVE";
          } else if ((agility - strength) >= 4 && accel >= 70 && height <= 182) {
            accelerateType_chem = "CONTROLLED_EXPLOSIVE";
          } else if ((strength - agility) >= 20 && strength >= 80 && height >= 188) {
            accelerateType_chem = "LENGTHY";
          } else if ((strength - agility) >= 12 && strength >= 75 && height >= 183) {
            accelerateType_chem = "MOSTLY_LENGTHY";
          } else if ((strength - agility) >= 4 && strength >= 65 && height >= 181) {
            accelerateType_chem = "CONTROLLED_LENGTHY";
          }

          console.log(`➡️ Pushing meta:`, {
            archetype,
            metaR_0chem,
            metaR_3chem,
            bestChemStyle: bestChemStyleName,
            accelerateType_chem
          });
          player.esMetaRatings.push({
            archetype,
            metaR_0chem,
            metaR_3chem,
            bestChemStyle: bestChemStyleName,
            accelerateType_chem
          });

          await delay(250);
        }));
        console.log(`✅ Final metaRatings for ${player.commonName}:`, player.esMetaRatings);
        players[i + index] = player;
        return player;
      })
    );

    results.forEach((res, idx) => {
      if (res.status === 'fulfilled' && res.value) {
        players[i + idx] = res.value;
        console.log(`📝 Updating player ${res.value.commonName}`);
      }
    });

    await delay(DELAY_BETWEEN_BATCHES_MS);
    console.log(`✅ Completed batch ${i / MAX_CONCURRENT + 1}/${Math.ceil(players.length / MAX_CONCURRENT)}`);
  }

  fs.writeFileSync(path.join(dataDir, 'club_players_with_meta.json'), JSON.stringify(players, null, 2));
  console.log('✅ Finished saving club_players_with_meta.json');
})();