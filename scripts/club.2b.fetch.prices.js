// club.2b.fetch.prices.js
const fs = require('fs').promises;
const path = require('path');
const puppeteer = require('puppeteer');

// --- Configuration ---
const TRADEABLE_IDS_FILE = '../data/tradeable_ids.json';
const PRICES_DIR = '../data/raw/prices';
const PRICES_URL_TEMPLATE = (eaId) => `https://www.fut.gg/api/fut/player-prices/26/${eaId}/`;

// --- Helpers ---
const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

async function ensureDirExists(dirPath) {
    try {
        await fs.mkdir(dirPath, { recursive: true });
    } catch (error) {
        if (error.code !== 'EEXIST') throw error;
    }
}

async function fetchWithPuppeteer(browser, url) {
    const page = await browser.newPage();
    await page.setUserAgent('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36');
    try {
        await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 15000 });
        const content = await page.evaluate(() => document.body.innerText);
        return JSON.parse(content);
    } catch (err) {
        console.warn(`⚠️ Failed to fetch ${url}: ${err.message}`);
        return null;
    } finally {
        await page.close();
    }
}

// --- Main ---
(async () => {
    console.log("💰 Starting Price Fetching Script...");
    
    const scriptDir = __dirname;
    const idsPath = path.resolve(scriptDir, TRADEABLE_IDS_FILE);
    const pricesDirAbs = path.resolve(scriptDir, PRICES_DIR);
    
    await ensureDirExists(pricesDirAbs);

    let tradeableIds = [];
    try {
        const raw = await fs.readFile(idsPath, 'utf-8');
        tradeableIds = JSON.parse(raw);
    } catch (e) {
        console.log("ℹ️ No tradeable_ids.json found or empty. Skipping price fetch.");
        return;
    }

    if (tradeableIds.length === 0) {
        console.log("ℹ️ 0 Tradeable players found. Exiting.");
        return;
    }

    console.log(`ℹ️ Fetching prices for ${tradeableIds.length} tradeable players...`);
    
    const browser = await puppeteer.launch({ headless: true, args: ['--no-sandbox'] });
    
    for (let i = 0; i < tradeableIds.length; i++) {
        const eaId = tradeableIds[i];
        const url = PRICES_URL_TEMPLATE(eaId);
        
        const json = await fetchWithPuppeteer(browser, url);
        
        if (json && json.data && json.data.currentPrice) {
            const priceData = {
                price: json.data.currentPrice.price,
                isExtinct: json.data.currentPrice.isExtinct
            };
            await fs.writeFile(path.join(pricesDirAbs, `${eaId}.json`), JSON.stringify(priceData, null, 2));
            console.log(`✅ [${i+1}/${tradeableIds.length}] Saved price for ${eaId}: ${priceData.price}`);
        } else {
            console.log(`❌ [${i+1}/${tradeableIds.length}] Failed/No price for ${eaId}`);
        }
        
        // Small delay to be polite
        await delay(500); 
    }

    await browser.close();
    console.log("🎉 Price fetching complete.");
})();