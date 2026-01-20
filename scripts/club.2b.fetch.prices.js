// club.2b.fetch.prices.js (Turbo Edition)
// High-Performance Parallel Price Fetcher
// Optimizations: Concurrency 25, Stealth Plugin, Persistent Browser Contexts

const fs = require('fs').promises;
const path = require('path');

// Use puppeteer-extra with stealth (Critical for Fut.gg API access)
const puppeteer = require('puppeteer-extra');
const StealthPlugin = require('puppeteer-extra-plugin-stealth');
puppeteer.use(StealthPlugin());

// --- Configuration ---
const CONCURRENCY = 25; // user-preferred concurrency level
const RETRIES = 2;
const DELAY_MIN = 50;
const DELAY_MAX = 200;

const TRADEABLE_IDS_FILE = '../data/tradeable_ids.json';
const PRICES_DIR = '../data/raw/prices';
const PRICES_URL_TEMPLATE = (eaId) => `https://www.fut.gg/api/fut/player-prices/26/${eaId}/`;

// --- Helpers ---
const sleep = (ms) => new Promise(r => setTimeout(r, ms));
const randInt = (min, max) => Math.floor(Math.random() * (max - min + 1)) + min;

async function ensureDir(dir) {
    await fs.mkdir(dir, { recursive: true }).catch(e => { if (e.code !== 'EEXIST') throw e; });
}

async function fileExists(filePath) {
    try { await fs.access(filePath); return true; } catch { return false; }
}

// Robust Context Creator
async function createSafeContext(browser) {
    if (typeof browser.createBrowserContext === 'function') {
        return await browser.createBrowserContext();
    }
    if (typeof browser.createIncognitoBrowserContext === 'function') {
        return await browser.createIncognitoBrowserContext();
    }
    return browser.defaultBrowserContext();
}

// --- Fetching Logic ---
async function fetchPrice(eaId, browser, pricesDir) {
    let context = null;
    let page = null;
    const url = PRICES_URL_TEMPLATE(eaId);

    for (let i = 0; i <= RETRIES; i++) {
        try {
            if (i > 0) await sleep(500 * i);

            // Create lightweight context
            context = await createSafeContext(browser);
            page = await context.newPage();

            // Block resources for speed (we only need the JSON text)
            await page.setRequestInterception(true);
            page.on('request', r => ['image','media','font','stylesheet'].includes(r.resourceType()) ? r.abort() : r.continue());

            await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 10000 });
            
            // Extract body text (JSON)
            const content = await page.evaluate(() => document.body.innerText);
            if (!content) throw new Error("Empty body");
            
            const json = JSON.parse(content);
            
            // Close pages immediately
            await page.close();
            await context.close();
            page = null; context = null;

            // Process & Save
            if (json && json.data && json.data.currentPrice) {
                const priceData = {
                    price: json.data.currentPrice.price,
                    isExtinct: json.data.currentPrice.isExtinct
                };
                await fs.writeFile(path.join(pricesDir, `${eaId}.json`), JSON.stringify(priceData, null, 2));
                return { id: eaId, status: 'success', price: priceData.price };
            } else {
                return { id: eaId, status: 'no_price_data' };
            }

        } catch (e) {
            // Cleanup on error
            if (page && !page.isClosed()) await page.close().catch(() => {});
            if (context) await context.close().catch(() => {});

            if (i === RETRIES) {
                return { id: eaId, status: 'error', error: e.message };
            }
        }
    }
}

// --- Main Runner ---
(async () => {
    console.log("💰 Starting TURBO Price Fetcher...");
    const start = Date.now();
    const root = __dirname;
    const pricesDirAbs = path.resolve(root, PRICES_DIR);
    const idsPath = path.resolve(root, TRADEABLE_IDS_FILE);

    // Setup
    await ensureDir(pricesDirAbs);

    // Load IDs
    let tradeableIds = [];
    try {
        const raw = await fs.readFile(idsPath, 'utf-8');
        tradeableIds = JSON.parse(raw);
    } catch (e) {
        console.log("ℹ️ No tradeable_ids.json found. Skipping.");
        return;
    }

    if (tradeableIds.length === 0) {
        console.log("ℹ️ 0 Tradeable players found. Exiting.");
        return;
    }

    // Filter Existing Prices (Optional: Remove if you always want fresh prices)
    // For prices, you usually want FRESH data, so we might NOT skip existing files.
    // However, if you are recovering from a crash, skipping is helpful.
    // Uncomment below to skip existing:
    /*
    const todo = [];
    for (const id of tradeableIds) {
        if (!await fileExists(path.join(pricesDirAbs, `${id}.json`))) {
            todo.push(id);
        }
    }
    */
    const todo = [...tradeableIds]; // Fetch ALL prices fresh

    console.log(`📋 Fetching prices for ${todo.length} players...`);

    // Launch Browser
    console.log("🕸️ Launching Browser...");
    const browser = await puppeteer.launch({
        headless: "new",
        defaultViewport: null,
        args: ['--no-sandbox', '--disable-setuid-sandbox']
    });

    // Worker Pool
    let processed = 0;
    const total = todo.length;
    
    async function worker() {
        while (todo.length > 0) {
            const id = todo.shift();
            await fetchPrice(id, browser, pricesDirAbs);
            
            processed++;
            if (processed % 50 === 0 || processed === total) {
                const pct = ((processed / total) * 100).toFixed(1);
                const elapsed = (Date.now() - start) / 1000;
                const rate = (processed / elapsed).toFixed(2);
                console.log(`⚡ ${processed}/${total} (${pct}%) | Rate: ${rate} p/s | Elapsed: ${elapsed.toFixed(0)}s`);
            }
            await sleep(randInt(DELAY_MIN, DELAY_MAX));
        }
    }

    console.log(`🔥 Igniting ${CONCURRENCY} concurrent workers...`);
    const workers = Array(CONCURRENCY).fill(null).map(() => worker());
    await Promise.all(workers);

    await browser.close();
    console.log(`🎉 Prices fetched in ${((Date.now() - start)/1000).toFixed(2)}s.`);
})();