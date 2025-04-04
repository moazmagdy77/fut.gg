const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

(async () => {
  const dataDir = path.resolve(__dirname, '..', '..', 'data');
  const playerIds = JSON.parse(fs.readFileSync(path.join(dataDir, 'player_ids.json')));

  const browser = await puppeteer.launch({
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
  });
  const page = await browser.newPage();

  await page.setUserAgent(
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
  );

  const playerData = [];

  for (const id of playerIds) {
    const apiUrl = `https://www.fut.gg/api/fut/player-item-definitions/25/${id}/`;
    console.log(`📦 Fetching data for player ID ${id}`);

    try {
      await page.goto(apiUrl, { waitUntil: 'networkidle2' });
      const content = await page.evaluate(() => document.body.innerText);
      const json = JSON.parse(content);
      playerData.push(json);
    } catch (err) {
      console.error(`❌ Failed for player ID ${id}: ${err.message}`);
    }

    // Delay between requests to avoid rate-limiting
    await new Promise(resolve => setTimeout(resolve, 500));
  }

  fs.writeFileSync(path.join(dataDir, 'player_data.json'), JSON.stringify(playerData, null, 2));
  await browser.close();
  console.log(`✅ Saved data for ${playerData.length} players to player_data.json`);
  process.exit(0);
})();