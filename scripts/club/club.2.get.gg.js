const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

const fetchWithRetry = async (browser, id, retries = 2) => {
  const apiUrl = `https://www.fut.gg/api/fut/player-item-definitions/25/${id}/`;
  console.log(`📦 Fetching data for player ID ${id}`);

  for (let attempt = 0; attempt <= retries; attempt++) {
    const page = await browser.newPage();
    await page.setUserAgent(
      'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    );
    page.setDefaultNavigationTimeout(60000);

    try {
      const result = await Promise.race([
        page.goto(apiUrl, { waitUntil: 'domcontentloaded' }).then(async () => {
          const content = await page.evaluate(() => document.body.innerText);
          return JSON.parse(content);
        }),
        new Promise((_, reject) => setTimeout(() => reject(new Error('⏰ Hard timeout exceeded')), 20000))
      ]);

      await page.close();
      return result;
    } catch (err) {
      await page.close();
      if (attempt < retries) {
        console.warn(`⚠️ Retry ${attempt + 1} for player ID ${id}`);
        await new Promise(r => setTimeout(r, 1000));
      } else {
        console.error(`❌ Failed for player ID ${id} after ${retries + 1} attempts: ${err.message}`);
      }
    }
  }

  return null;
};

(async () => {
  const dataDir = path.resolve(__dirname, '..', '..', 'data');
  const playerIds = JSON.parse(fs.readFileSync(path.join(dataDir, 'club_ids.json')));
  const MAX_CONCURRENT = 5;
  const DELAY_BETWEEN_BATCHES_MS = 1000;

  const browser = await puppeteer.launch({
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
  });

  const playerData = [];

  for (let i = 0; i < playerIds.length; i += MAX_CONCURRENT) {
    const batch = playerIds.slice(i, i + MAX_CONCURRENT);

    const results = await Promise.allSettled(
      batch.map(async (id) => {
        const json = await fetchWithRetry(browser, id);
        if (json) playerData.push(json);
      })
    );

    await new Promise(resolve => setTimeout(resolve, DELAY_BETWEEN_BATCHES_MS));
    console.log(`✅ Completed batch ${i / MAX_CONCURRENT + 1}/${Math.ceil(playerIds.length / MAX_CONCURRENT)}`);
  }

  fs.writeFileSync(path.join(dataDir, 'club_players.json'), JSON.stringify(playerData, null, 2));
  await browser.close();
  console.log(`✅ Saved data for ${playerData.length} players to club_players.json`);
  process.exit(0);
})();