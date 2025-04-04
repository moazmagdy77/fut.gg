const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

(async () => {
  const browser = await puppeteer.launch({ headless: true });
  const page = await browser.newPage();

  const TOP_PLAYER_PAGES = 73;
  const NEW_PLAYER_PAGES = 20;

  const dataDir = path.resolve(__dirname, '..', '..', 'data');
  const progressFile = path.join(dataDir, 'scraping_progress.json');
  const outputFile = path.join(dataDir, 'player_ids.json');

  let allPlayerIds = new Set();
  let progress = {};

  if (fs.existsSync(outputFile)) {
    allPlayerIds = new Set(JSON.parse(fs.readFileSync(outputFile)));
  }

  if (fs.existsSync(progressFile)) {
    progress = JSON.parse(fs.readFileSync(progressFile));
  }

  const pageConfigs = [
    { baseUrl: 'https://www.fut.gg/players/?role_plus_plus_count__gte=1&page=', pages: TOP_PLAYER_PAGES, label: 'top' },
    { baseUrl: 'https://www.fut.gg/players/new/?page=', pages: NEW_PLAYER_PAGES, label: 'new' }
  ];

  for (const { baseUrl, pages, label } of pageConfigs) {
    const startPage = progress[label] || 1;

    for (let i = startPage; i <= pages; i++) {
      const url = `${baseUrl}${i}`;

      let success = false;
      let retries = 1;

      while (!success && retries >= 0) {
        console.log(`🔄 Visiting ${url}`);
        try {
          await page.goto(url, {
            waitUntil: 'networkidle2',
            timeout: 60000
          });
          success = true;
        } catch (err) {
          console.warn(`⚠️ Timeout on ${url}. Retrying in 2 minutes... (${retries} retries left)`);
          if (retries === 0) throw err;
          retries--;
          await new Promise(res => setTimeout(res, 2 * 60 * 1000));
        }
      }

      const cardLinks = await page.$$eval('a.fc-card-container', cards =>
        cards.map(card => card.href)
      );

      const playerIds = cardLinks.map(url => {
        const parts = url.split('/');
        return parts[parts.length - 2].replace('25-', '');
      });

      playerIds.forEach(id => allPlayerIds.add(id));

      fs.writeFileSync(outputFile, JSON.stringify([...allPlayerIds], null, 2));
      progress[label] = i + 1;
      fs.writeFileSync(progressFile, JSON.stringify(progress, null, 2));

      await new Promise(res => setTimeout(res, 5000));
    }
  }

  await browser.close();
  console.log(`✅ Finished. Total unique player IDs: ${allPlayerIds.size}`);
  process.exit(0);
})();