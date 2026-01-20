// model.3.fetch.data.js (Turbo Edition)
// High-Performance Parallel Fetcher
// Optimizations: Concurrency=25, Async I/O (Axios + Puppeteer parallel), Reduced Delays

const fs = require('fs').promises;
const path = require('path');
const axios = require('axios');

const puppeteer = require('puppeteer-extra');
const StealthPlugin = require('puppeteer-extra-plugin-stealth');
puppeteer.use(StealthPlugin());

// --- Configuration ---
const CONCURRENCY = 25; // Aggressive concurrency for your 32GB RAM
const RETRIES = 2;
// Tightened delays (we rely on stealth and concurrency for distribution)
const DELAY_MIN = 50;   
const DELAY_MAX = 200; 

const CLUB_IDS_FILE = '../data/player_ids.json';
const MAPS_FILE = '../data/maps.json';
const GG_DATA_DIR = '../data/raw/ggData';
const ES_META_DIR = '../data/raw/esMeta';
const GG_META_DIR = '../data/raw/ggMeta';

// Templates
const URL_GG_DETAILS = (id) => `https://www.fut.gg/api/fut/player-item-definitions/26/${id}/`;
const URL_GG_META = (id) => `https://www.fut.gg/api/fut/metarank/player/${id}/`;
const URL_ES_META = (id, role) => `https://api.easysbc.io/players/${id}?player-role-id=${role}&expanded=false`;

// --- Helpers ---
const sleep = (ms) => new Promise(r => setTimeout(r, ms));
const randInt = (min, max) => Math.floor(Math.random() * (max - min + 1)) + min;

async function ensureDir(dir) {
    await fs.mkdir(dir, { recursive: true }).catch(e => { if (e.code !== 'EEXIST') throw e; });
}

async function fileExists(filePath) {
    try { await fs.access(filePath); return true; } catch { return false; }
}

async function saveData(filePath, data) {
    const tmp = `${filePath}.tmp`;
    await fs.writeFile(tmp, JSON.stringify(data, null, 2));
    await fs.rename(tmp, filePath);
}

// Robust Context Creator (Handles API differences)
async function createSafeContext(browser) {
    if (typeof browser.createBrowserContext === 'function') {
        return await browser.createBrowserContext();
    }
    if (typeof browser.createIncognitoBrowserContext === 'function') {
        return await browser.createIncognitoBrowserContext();
    }
    return browser.defaultBrowserContext();
}

async function checkSkippability(id, dirs) {
    const [d1, d2, d3] = await Promise.all([
        fileExists(path.join(dirs.ggData, `${id}_ggData.json`)),
        fileExists(path.join(dirs.esMeta, `${id}_esMeta.json`)),
        fileExists(path.join(dirs.ggMeta, `${id}_ggMeta.json`))
    ]);
    return d1 && d2 && d3;
}

// --- Fetching Logic ---

async function fetchPuppeteer(page, url, type) {
    for (let i = 0; i <= RETRIES; i++) {
        try {
            if (i > 0) await sleep(500 * i);
            await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 15000 });
            const text = await page.evaluate(() => document.body.innerText);
            if (!text) throw new Error("Empty body");
            return JSON.parse(text);
        } catch (e) {
            if (i === RETRIES) {
                // Suppress logspam for expected occasional failures
                // console.warn(`⚠️ [${type}] Failed ${url}: ${e.message}`);
                return null;
            }
        }
    }
}

async function fetchEasySBC(eaId, roleId) {
    const url = URL_ES_META(eaId, roleId);
    for (let i = 0; i <= RETRIES; i++) {
        try {
            const res = await axios.get(url, { timeout: 8000 });
            return res.data;
        } catch (e) {
            if (i === RETRIES) return null;
            await sleep(200 * (i + 1));
        }
    }
}

// --- Worker ---
async function processPlayer(eaId, browser, maps, dirs) {
    const eaIdStr = String(eaId);
    let context = null;
    let page = null;
    
    try {
        context = await createSafeContext(browser);
        page = await context.newPage();
        
        await page.setRequestInterception(true);
        page.on('request', r => ['image','media','font','stylesheet'].includes(r.resourceType()) ? r.abort() : r.continue());

        // 1. Fetch Details (Must happen first to get Roles)
        const details = await fetchPuppeteer(page, URL_GG_DETAILS(eaId), "Details");
        if (!details || !details.data) return { id: eaId, status: 'fail_gg_details' };

        // 2. Parallel Execution: Run GG Meta (Puppeteer) AND EasySBC (Axios) at the same time
        // This hides the latency of the Axios calls behind the Puppeteer page load
        
        // Task A: Puppeteer fetch for Meta
        const taskGGMeta = fetchPuppeteer(page, URL_GG_META(eaId), "MetaRank");

        // Task B: Calculate Roles and fetch EasySBC (Axios)
        const taskESMeta = (async () => {
            const pDef = details.data;
            const positions = [pDef.position, ...(pDef.alternativePositionIds || [])];
            const rolesToFetch = new Set();
            
            positions.forEach(pos => {
                const roles = maps.positionIdToEsRoleIds[String(pos)];
                if (roles) roles.forEach(r => rolesToFetch.add(r));
            });

            const esResults = [];
            if (rolesToFetch.size > 0) {
                // Fire these requests nearly simultaneously
                const promises = Array.from(rolesToFetch).map(async (rId) => {
                    const data = await fetchEasySBC(eaId, rId);
                    if (data) return { roleId: rId, data };
                    return null;
                });
                const results = await Promise.all(promises);
                esResults.push(...results.filter(r => r !== null));
            }
            return esResults;
        })();

        // Wait for both
        const [meta, esData] = await Promise.all([taskGGMeta, taskESMeta]);

        // Cleanup Puppeteer immediately
        await page.close();
        await context.close();
        page = null; context = null;

        // 3. Save All Data
        await saveData(path.join(dirs.ggData, `${eaIdStr}_ggData.json`), details);
        if (meta) await saveData(path.join(dirs.ggMeta, `${eaIdStr}_ggMeta.json`), meta);
        if (esData && esData.length > 0) await saveData(path.join(dirs.esMeta, `${eaIdStr}_esMeta.json`), esData);

        return { id: eaId, status: 'success' };

    } catch (e) {
        return { id: eaId, status: 'error' };
    } finally {
        if (page && !page.isClosed()) await page.close().catch(() => {});
        if (context && context !== browser.defaultBrowserContext()) await context.close().catch(() => {});
    }
}

// --- Main Runner ---
(async () => {
    console.log("🚀 Starting TURBO Data Fetcher...");
    const start = Date.now();

    const root = __dirname;
    const dirs = {
        ggData: path.resolve(root, GG_DATA_DIR),
        esMeta: path.resolve(root, ES_META_DIR),
        ggMeta: path.resolve(root, GG_META_DIR)
    };
    
    await Promise.all(Object.values(dirs).map(ensureDir));
    const ids = JSON.parse(await fs.readFile(path.resolve(root, CLUB_IDS_FILE)));
    const maps = JSON.parse(await fs.readFile(path.resolve(root, MAPS_FILE)));

    // Fast Filtering
    const todo = [];
    const chunk = 250; // Larger chunk for filesystem check
    for (let i = 0; i < ids.length; i+=chunk) {
        const batch = ids.slice(i, i+chunk);
        const results = await Promise.all(batch.map(async id => ({ 
            id, skip: await checkSkippability(id, dirs) 
        })));
        todo.push(...results.filter(r => !r.skip).map(r => r.id));
    }

    console.log(`📋 Fetching ${todo.length} players. (Skipped ${ids.length - todo.length})`);
    if (todo.length === 0) return;

    console.log("🕸️ Launching Browser...");
    const browser = await puppeteer.launch({
        headless: "new",
        defaultViewport: null,
        args: ['--no-sandbox', '--disable-setuid-sandbox']
    });

    let processed = 0;
    const total = todo.length;
    
    async function worker() {
        while (todo.length > 0) {
            const id = todo.shift();
            await processPlayer(id, browser, maps, dirs);
            
            processed++;
            if (processed % 50 === 0) {
                const pct = ((processed / total) * 100).toFixed(1);
                const elapsed = (Date.now() - start) / 1000;
                const rate = (processed / elapsed).toFixed(2); // Players per second
                console.log(`⚡ ${processed}/${total} (${pct}%) | Rate: ${rate} p/s | Elapsed: ${elapsed.toFixed(0)}s`);
            }
            
            // Minimal delay to allow CPU context switching
            await sleep(randInt(DELAY_MIN, DELAY_MAX));
        }
    }

    console.log(`🔥 Igniting ${CONCURRENCY} concurrent workers...`);
    const workers = Array(CONCURRENCY).fill(null).map(() => worker());
    await Promise.all(workers);

    await browser.close();
    console.log(`✅ DONE in ${((Date.now() - start)/1000/60).toFixed(2)} minutes.`);
})();