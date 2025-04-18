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

  let browser;
  const launchBrowser = async () => {
    return puppeteer.launch({
      headless: true,
      args: ['--no-sandbox', '--disable-setuid-sandbox'],
    });
  };
  browser = await launchBrowser();

  const playerData = [];

  for (let i = 0; i < playerIds.length; i += MAX_CONCURRENT) {
    const batch = playerIds.slice(i, i + MAX_CONCURRENT);
    
    if (i > 0 && i % (MAX_CONCURRENT * 50) === 0) {
      console.log(`🔁 Restarting browser to free resources...`);
      try {
        await browser.close();
      } catch (err) {
        console.warn('⚠️ Error closing browser during restart:', err.message);
      }
      browser = await launchBrowser();
    }

    console.time(`⏱ Batch ${i / MAX_CONCURRENT + 1}`);

    const results = await Promise.allSettled(
      batch.map(async (id) => {
        const json = await fetchWithRetry(browser, id);
        console.log(`📥 Finished ID ${id}`);
        if (json) playerData.push(json);
      })
    );
    console.timeEnd(`⏱ Batch ${i / MAX_CONCURRENT + 1}`);

    await new Promise(resolve => setTimeout(resolve, DELAY_BETWEEN_BATCHES_MS));
    console.log(`✅ Completed batch ${i / MAX_CONCURRENT + 1}/${Math.ceil(playerIds.length / MAX_CONCURRENT)}`);
  }

  console.log(`💾 Writing output to file...`);
  try {
    await Promise.race([
      new Promise((resolve, reject) => {
        fs.writeFile(path.join(dataDir, 'club_players.json'), JSON.stringify(playerData, null, 2), (err) => {
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
  
  try {
    await Promise.race([
      browser.close(),
      new Promise((_, reject) => setTimeout(() => reject(new Error('❌ Browser.close() timed out')), 20000))
    ]);
  } catch (err) {
    console.error(err.message);
  }
  console.log(`✅ Saved data for ${playerData.length} players to club_players.json`);
  process.exit(0);
})();