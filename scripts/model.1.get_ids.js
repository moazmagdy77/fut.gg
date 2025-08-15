// model.1.get_ids.js --> The Robust Fetcher

const puppeteer = require('puppeteer-extra');
const StealthPlugin = require('puppeteer-extra-plugin-stealth');
const fs = require('fs');
const path = require('path');

// Apply the stealth plugin to make the browser harder to detect
puppeteer.use(StealthPlugin());

// --- Configuration ---
const TOP_PLAYER_PAGES = 100; // Total pages you want to scrape
const BATCH_SIZE = 25; // How many pages to scrape before restarting the browser
const MAX_RETRIES = 3; // Max attempts for each page

const dataDir = path.resolve(__dirname, '..', 'data');
const outputDir = path.join(dataDir, 'raw', 'r_objects');

// --- Main Function ---
(async () => {
  let browser;

  // Create the output directory if it doesn't exist
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  const baseUrl = 'https://www.fut.gg/players/?page=';

  for (let i = 1; i <= TOP_PLAYER_PAGES; i++) {
    // --- Browser Restart Logic ---
    // Relaunch the browser at the start and after each batch
    if ((i - 1) % BATCH_SIZE === 0) {
      if (browser) {
        console.log(`\n🔄 Restarting browser after batch of ${BATCH_SIZE} pages...`);
        await browser.close();
      }
      console.log('🚀 Launching new browser instance...');
      browser = await puppeteer.launch({ headless: true });
    }

    const outputFilePath = path.join(outputDir, `page_${i}.json`);

    if (fs.existsSync(outputFilePath)) {
      console.log(`🟡 Page ${i}/${TOP_PLAYER_PAGES} already downloaded. Skipping.`);
      continue;
    }

    let success = false;
    for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
      let page = null;
      try {
        console.log(`🔄 Visiting page ${i}/${TOP_PLAYER_PAGES} at ${baseUrl}${i} (Attempt ${attempt}/${MAX_RETRIES})`);
        page = await browser.newPage();
        await page.goto(`${baseUrl}${i}`, { timeout: 90000 });

        console.log('... waiting for player cards to render...');
        await page.waitForSelector('a[href*="/players/"]', { timeout: 60000 });
        console.log('... cards rendered.');
        
        await new Promise(res => setTimeout(res, 3000));

        console.log('... extracting data from global state...');
        const R_object = await page.evaluate(() => window.$R);

        if (R_object) {
          fs.writeFileSync(outputFilePath, JSON.stringify(R_object, null, 2));
          console.log(`✅ Success! Saved data for page ${i}`);
          success = true;
          break; // Exit the retry loop on success
        } else {
          throw new Error('Could not find window.$R object on the page.');
        }
      } catch (err) {
        console.warn(`⚠️ Attempt ${attempt} failed for page ${i}: ${err.message}`);
        if (attempt === MAX_RETRIES) {
          console.error(`❌ All attempts failed for page ${i}. Skipping.`);
          const screenshotPath = `error_screenshot_page_${i}.png`;
          if (page) await page.screenshot({ path: screenshotPath, fullPage: true });
          console.log(`📸 Screenshot saved to ${screenshotPath}`);
        } else {
            await new Promise(res => setTimeout(res, 5000)); // Wait before retrying
        }
      } finally {
        if (page) await page.close();
      }
    }
     await new Promise(res => setTimeout(res, 1000)); // Polite delay between pages
  }

  // --- Final Shutdown Logic ---
  if (browser) {
    try {
      console.log('🚪 Closing final browser instance...');
      await Promise.race([
        browser.close(),
        new Promise((_, reject) => setTimeout(() => reject(new Error('Browser close timed out')), 15000))
      ]);
      console.log('✅ Browser closed successfully.');
    } catch (e) {
      console.warn(`⚠️ Could not close browser gracefully: ${e.message}.`);
    }
  }
  
  console.log(`🎉 Finished fetching raw data objects.`);
  process.exit(0); // Force exit to prevent hanging
})();