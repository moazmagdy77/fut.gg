const fs = require('fs');
const path = require('path');
const axios = require('axios');

const dataDir = path.resolve(__dirname, '..', '..', 'data');

const players = JSON.parse(fs.readFileSync(path.join(dataDir, 'enhanced_club_players.json'), 'utf-8'));
const maps = JSON.parse(fs.readFileSync(path.join(dataDir, 'maps.json'), 'utf-8'));
const chemStyleMap = maps.chemStyles;

const MAX_CONCURRENT = 5;
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

(async () => {
  for (let i = 0; i < players.length; i += MAX_CONCURRENT) {
    const batch = players.slice(i, i + MAX_CONCURRENT);

    const results = await Promise.allSettled(
      batch.map(async (player) => {
        player.metaRatings = [];

        if (!player.archetype || !player.eaId) return;

        for (const archetype of player.archetype) {
          const ratings = await fetchMetaRatingsWithRetry(player, archetype);

          const mappedRatings = ratings
            .filter(r => r.chemistry === 0)
            .map(r => ({
              metaRating: r.metaRating
            }));

          if (!mappedRatings.length) {
            console.warn(`⚠️ No ratings found for ${player.commonName} - Archetype: ${archetype}`);
          }

          player.metaRatings.push({
            archetype,
            ...mappedRatings[0]
          });

          await delay(250);
        }
      })
    );

    await delay(DELAY_BETWEEN_BATCHES_MS);
    console.log(`✅ Completed batch ${i / MAX_CONCURRENT + 1}/${Math.ceil(players.length / MAX_CONCURRENT)}`);
  }

  fs.writeFileSync(path.join(dataDir, 'club_players_with_meta.json'), JSON.stringify(players, null, 2));
  console.log('✅ Finished saving club_players_with_meta.json');
})();