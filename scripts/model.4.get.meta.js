const fs = require('fs');
const path = require('path');
const axios = require('axios');

const dataDir = path.resolve(__dirname, '..', 'data');

const players = JSON.parse(fs.readFileSync(path.join(dataDir, 'enhanced_players.json'), 'utf-8'));
const maps = JSON.parse(fs.readFileSync(path.join(dataDir, 'maps.json'), 'utf-8'));
const chemStyleMap = maps.chemStyles;

const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

(async () => {
  const dataDir = path.resolve(__dirname, '..', '..', 'data');
  const players = JSON.parse(fs.readFileSync(path.join(dataDir, 'enhanced_players.json'), 'utf-8'));
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
        const response = await Promise.race([
          axios.get(url),
          new Promise((_, reject) => setTimeout(() => reject(new Error('⏰ Hard timeout exceeded')), HARD_TIMEOUT_MS))
        ]);

        const ratings = Array.isArray(response.data) ? response.data : [];
        const mappedRatings = ratings.map(r => ({
          ...r,
          chemStyle: chemStyleMap[r.chemstyleId.toString()] || r.chemstyleId
        }));

        return {
          archetype,
          ratings: mappedRatings
        };
      } catch (err) {
        if (attempt < MAX_RETRIES) {
          console.warn(`⚠️ Retry ${attempt + 1} for ${player.commonName} - Archetype: ${archetype}`);
          await delay(1000);
        } else {
          console.error(`❌ Failed for ${player.commonName} with archetype ${archetype}:`, err.message);
          return {
            archetype,
            ratings: []
          };
        }
      }
    }
  };

  for (let i = 0; i < players.length; i += MAX_CONCURRENT) {
    const batch = players.slice(i, i + MAX_CONCURRENT);
    console.time(`⏱ Batch ${i / MAX_CONCURRENT + 1}`);

    await Promise.allSettled(batch.map(async player => {
      player.metaRatings = [];

      if (!player.archetype || !player.eaId) return;

      for (const archetype of player.archetype) {
        const ratingData = await fetchMetaRatingsWithRetry(player, archetype);

        if (!ratingData.ratings.length) {
          console.warn(`⚠️ No ratings found for ${player.commonName} - Archetype: ${archetype}`);
        }

        player.metaRatings.push(ratingData);
        await delay(250);
      }

      console.log(`📥 Finished ${player.commonName}`);
    }));

    console.timeEnd(`⏱ Batch ${i / MAX_CONCURRENT + 1}`);
    console.log(`✅ Completed batch ${i / MAX_CONCURRENT + 1}/${Math.ceil(players.length / MAX_CONCURRENT)}`);
    await delay(DELAY_BETWEEN_BATCHES_MS);
  }

  try {
    await Promise.race([
      new Promise((resolve, reject) => {
        fs.writeFile(path.join(dataDir, 'players_with_meta.json'), JSON.stringify(players, null, 2), (err) => {
          if (err) return reject(err);
          return resolve();
        });
      }),
      new Promise((_, reject) => setTimeout(() => reject(new Error('❌ File write timed out')), 30000))
    ]);
    console.log(`💾 Successfully wrote output file.`);
  } catch (err) {
    console.error(`❌ Failed to write file: ${err.message}`);
  }

  process.exit(0);
})();