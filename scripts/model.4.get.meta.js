const fs = require('fs');
const path = require('path');
const axios = require('axios');

const dataDir = path.resolve(__dirname, '..', 'data');

const players = JSON.parse(fs.readFileSync(path.join(dataDir, 'enhanced_players.json'), 'utf-8'));
const maps = JSON.parse(fs.readFileSync(path.join(dataDir, 'maps.json'), 'utf-8'));
const chemStyleMap = maps.chemStyles;

let MAX_CONCURRENT = 5;
const DELAY_BETWEEN_BATCHES_MS = 1000;
const MAX_RETRIES = 3;
const HARD_TIMEOUT_MS = 20000;

const MIN_CONCURRENT = 3;
const MAX_CONCURRENT_LIMIT = 10;

const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

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

const fetchMetaRatingsWithRetry = async (player, archetype) => {
  const url = `https://api.easysbc.io/squad-builder/meta-ratings?archetypeId=${archetype}&resourceId=${player.eaId}`;
  console.log(`📦 Fetching meta rating for player ${player.commonName} - Archetype: ${archetype}`);

  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      const response = await Promise.race([
        axios.get(url),
        new Promise((_, reject) => setTimeout(() => reject(new Error('⏰ Hard timeout exceeded')), HARD_TIMEOUT_MS))
      ]);

      const ratings = Array.isArray(response.data) ? response.data : [];
      const mappedRatings = ratings.map(r => ({
        ...r,
        chemStyle: chemStyleMap[r.chemstyleId?.toString()] || r.chemstyleId
      }));

      return {
        archetype,
        ratings: mappedRatings
      };
    } catch (err) {
      if (attempt < MAX_RETRIES) {
        const backoffTime = 1000 * (2 ** attempt); // 1s, 2s, 4s
        console.warn(`⚠️ Retry ${attempt + 1} for ${player.commonName} - Archetype: ${archetype} after ${backoffTime / 1000}s`);
        await delay(backoffTime);
      } else {
        console.error(`❌ Failed for ${player.commonName} with archetype ${archetype} after ${MAX_RETRIES + 1} tries:`, err.message);
        return {
          archetype,
          ratings: []
        };
      }
    }
  }
};

const safeSave = async (players, filePath) => {
  try {
    await fs.promises.writeFile(filePath, JSON.stringify(players, null, 2));
    console.log(`💾 Progress saved to ${filePath}`);
  } catch (err) {
    console.error(`❌ Failed to save progress: ${err.message}`);
  }
};

(async () => {
  const outputFile = path.join(dataDir, 'players_with_meta.json');

  for (let i = 0; i < players.length; i += MAX_CONCURRENT) {
    const batch = players.slice(i, i + MAX_CONCURRENT);

    console.log(`🚀 Starting batch ${(i / MAX_CONCURRENT) + 1} with concurrency: ${MAX_CONCURRENT}`);
    const batchStart = Date.now();

    await Promise.allSettled(batch.map(async (player) => {
      player.metaRatings = [];

      if (!player.eaId) return;

      const playablePositions = [];
      if (player.positionMapped) playablePositions.push(player.positionMapped);
      if (Array.isArray(player.alternativePositionIdsMapped)) {
        playablePositions.push(...player.alternativePositionIdsMapped);
      }

      const allArchetypes = new Set();
      for (const pos of playablePositions) {
        if (positionToArchetypes[pos]) {
          positionToArchetypes[pos].forEach(arch => allArchetypes.add(arch));
        }
      }

      if (allArchetypes.size === 0) {
        console.warn(`⚠️ No archetypes found for ${player.commonName}`);
        return;
      }

      console.log(`✨ Processing ${player.commonName} with archetypes:`, [...allArchetypes]);

      for (const archetype of allArchetypes) {
        const ratingData = await fetchMetaRatingsWithRetry(player, archetype);

        if (!ratingData.ratings.length) {
          console.warn(`⚠️ No ratings found for ${player.commonName} - Archetype: ${archetype}`);
        }

        player.metaRatings.push(ratingData);
        await delay(200);
      }

      console.log(`📥 Finished ${player.commonName}`);
    }));

    const batchEnd = Date.now();
    const batchDuration = (batchEnd - batchStart) / 1000;
    console.log(`⏱ Batch ${(i / MAX_CONCURRENT) + 1} finished in ${batchDuration.toFixed(2)} seconds`);

    // Save after each batch
    await safeSave(players, outputFile);

    // Adjust concurrency
    if (batchDuration < 20 && MAX_CONCURRENT < MAX_CONCURRENT_LIMIT) {
      MAX_CONCURRENT++;
      console.log(`⬆️ Increasing concurrency to ${MAX_CONCURRENT}`);
    } else if (batchDuration > 40 && MAX_CONCURRENT > MIN_CONCURRENT) {
      MAX_CONCURRENT--;
      console.log(`⬇️ Decreasing concurrency to ${MAX_CONCURRENT}`);
    }

    console.log(`✅ Completed batch ${(i / MAX_CONCURRENT)}/${Math.ceil(players.length / MAX_CONCURRENT)}`);
    await delay(DELAY_BETWEEN_BATCHES_MS);
  }

  console.log(`🎉 Finished processing all players`);
  process.exit(0);
})();