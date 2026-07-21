// shared.fetch.data.js (Turbo v2)
// High-performance fetcher for fut.gg (ggData + ggMeta) and EasySBC (esMeta).
//
// Speed model / bot-detection:
//   - fut.gg is behind Cloudflare, so it MUST be hit from a real (stealth) browser.
//     Each worker navigates to fut.gg ONCE to clear Cloudflare, then pulls the JSON
//     APIs with in-page fetch() (page.evaluate). That request originates inside the
//     fut.gg page — same-origin, carries cf_clearance + real fingerprint — so it passes
//     bot-detection exactly like the site's own XHRs, while skipping the heavy
//     per-request page navigation the old version did.
//   - Pages/contexts are created ONCE per worker and reused across all its players.
//   - EasySBC has no Cloudflare, so esMeta uses axios directly (no browser).

const fs = require('fs').promises;
const path = require('path');
const axios = require('axios');

const puppeteer = require('puppeteer-extra');
const StealthPlugin = require('puppeteer-extra-plugin-stealth');
const { resolveChromePath } = require('./browser');
puppeteer.use(StealthPlugin());

// --- Process Arguments ---
// Example: node shared.fetch.data.js --file ../data/club_ids.json --mode club
const args = process.argv.slice(2);
let INPUT_FILE = '../data/club_ids.json';
let MODE = 'club';
for (let i = 0; i < args.length; i++) {
    if (args[i] === '--file' || args[i] === '--inputFile') { INPUT_FILE = args[++i]; }
    else if (args[i] === '--mode') { MODE = args[++i]; }
}

// --- Configuration (env-tunable) ---
// Sweet spot from benchmarking: ~4.7 p/s at 100% success across conc 8–12; fut.gg
// rate-limits well above ~12 (conc 25 → ~60% failures). Env-tunable if you want to probe higher.
const CONCURRENCY = Math.max(1, Number(process.env.CONCURRENCY || 8));
const PRIME_CONCURRENCY = Math.max(1, Number(process.env.PRIME_CONCURRENCY || 4)); // cap simultaneous Cloudflare primes
const RETRIES = 2;
const DELAY_MIN = Number(process.env.DELAY_MIN || 20);
const DELAY_MAX = Number(process.env.DELAY_MAX || 120);

const MAPS_FILE = '../data/maps.json';
const FUTGG_ORIGIN = 'https://www.fut.gg/';

const URL_GG_DETAILS = (id) => `https://www.fut.gg/api/fut/player-item-definitions/26/${id}/`;
const URL_GG_META = (id) => `https://www.fut.gg/api/fut/metarank/player/${id}/`;
const URL_ES_META = (id, role) => `https://api.easysbc.io/players/${id}?player-role-id=${role}&expanded=false`;

// --- Helpers ---
const sleep = (ms) => new Promise(r => setTimeout(r, ms));
const randInt = (min, max) => Math.floor(Math.random() * (max - min + 1)) + min;

async function ensureDir(dir) { await fs.mkdir(dir, { recursive: true }).catch(e => { if (e.code !== 'EEXIST') throw e; }); }
async function fileExists(filePath) { try { await fs.access(filePath); return true; } catch { return false; } }
async function saveData(filePath, data) {
    const tmp = `${filePath}.tmp`;
    await fs.writeFile(tmp, JSON.stringify(data, null, 2));
    await fs.rename(tmp, filePath);
}

async function createSafeContext(browser) {
    if (typeof browser.createBrowserContext === 'function') return await browser.createBrowserContext();
    if (typeof browser.createIncognitoBrowserContext === 'function') return await browser.createIncognitoBrowserContext();
    return browser.defaultBrowserContext();
}

// Bounded concurrency for Cloudflare priming navigations (avoids a startup burst).
function makeSemaphore(max) {
    let active = 0; const q = [];
    return {
        acquire: () => new Promise((res) => { const go = () => { if (active < max) { active++; res(); } else q.push(go); }; go(); }),
        release: () => { active--; if (q.length) q.shift()(); },
    };
}

async function checkSkippability(id, mode) {
    const root = __dirname;
    const trip = async (base) => {
        const [a, b, c] = await Promise.all([
            fileExists(path.join(root, `../data/raw/${base}/ggData`, `${id}_ggData.json`)),
            fileExists(path.join(root, `../data/raw/${base}/esMeta`, `${id}_esMeta.json`)),
            fileExists(path.join(root, `../data/raw/${base}/ggMeta`, `${id}_ggMeta.json`)),
        ]);
        return a && b && c;
    };
    if (mode === 'model') return trip('training');
    return (await trip('club - main')) || (await trip('club - rest'));
}

// --- EasySBC (no Cloudflare → axios) ---
async function fetchEasySBC(eaId, roleId) {
    const url = URL_ES_META(eaId, roleId);
    for (let i = 0; i <= RETRIES; i++) {
        try { const res = await axios.get(url, { timeout: 8000 }); return res.data; }
        catch (e) { if (i === RETRIES) return null; await sleep(200 * (i + 1)); }
    }
}

// --- fut.gg via in-page fetch (Cloudflare-safe) ---
async function primeCloudflare(page, url) {
    // Clear Cloudflare via a LIGHT fut.gg URL (a JSON API endpoint) rather than the heavy
    // homepage — fast, and reliable even with many simultaneous contexts.
    try { await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 30000 }); return true; }
    catch { return false; }
}

async function newFetchPage(context) {
    const page = await context.newPage();
    await page.setRequestInterception(true);
    page.on('request', (r) => {
        // Block heavy sub-resources on navigations; always allow xhr/fetch/document/script.
        const t = r.resourceType();
        (['image', 'media', 'font', 'stylesheet'].includes(t) ? r.abort() : r.continue()).catch(() => {});
    });
    return page;
}

function blocked(obj) {
    return !obj || obj.__err || obj.__status === 403 || obj.__status === 429 || obj.__status === 503 || obj.__status === 'nonjson';
}

// Fetch details + metarank in a single in-page round-trip, in parallel.
async function fetchGG(page, id) {
    return await page.evaluate(async (dUrl, mUrl) => {
        const getJson = async (u) => {
            try {
                const r = await fetch(u, { headers: { accept: 'application/json' }, credentials: 'include' });
                if (!r.ok) return { __status: r.status };
                const t = await r.text();
                try { return JSON.parse(t); } catch { return { __status: 'nonjson' }; }
            } catch (e) { return { __err: String((e && e.message) || e) }; }
        };
        const [details, meta] = await Promise.all([getJson(dUrl), getJson(mUrl)]);
        return { details, meta };
    }, URL_GG_DETAILS(id), URL_GG_META(id));
}

async function processPlayer(eaId, page, reprime, maps) {
    const eaIdStr = String(eaId);

    // 1. fut.gg details + meta (in-page), refreshing Cloudflare on a challenge.
    let gg = null;
    for (let attempt = 0; attempt <= RETRIES; attempt++) {
        gg = await fetchGG(page, eaId).catch(() => null);
        if (gg && gg.details && gg.details.data) break;
        if (gg && blocked(gg.details)) { await reprime(); await sleep(400 * (attempt + 1)); continue; }
        break; // genuine miss (e.g. 404) — don't hammer
    }
    const details = gg && gg.details;
    if (!details || !details.data) return { id: eaId, status: 'fail_gg_details' };
    const meta = (gg.meta && !blocked(gg.meta)) ? gg.meta : null;

    // 2. EasySBC esMeta (axios), roles derived from positions.
    const pDef = details.data;
    const rolesToFetch = new Set();
    [pDef.position, ...(pDef.alternativePositionIds || [])].forEach((pos) => {
        const roles = maps.positionIdToEsRoleIds[String(pos)];
        if (roles) roles.forEach((r) => rolesToFetch.add(r));
    });
    let esData = [];
    if (rolesToFetch.size > 0) {
        const results = await Promise.all(Array.from(rolesToFetch).map(async (rId) => {
            const data = await fetchEasySBC(eaId, rId);
            return data ? { roleId: rId, data } : null;
        }));
        esData = results.filter(Boolean);
    }

    // 3. Save (subfolder by mode + overall). Dirs are pre-created in the runner.
    let subfolder = 'training';
    if (MODE === 'club') subfolder = (details.data.overall >= 75) ? 'club - main' : 'club - rest';
    const base = path.resolve(__dirname, `../data/raw/${subfolder}`);
    await saveData(path.join(base, 'ggData', `${eaIdStr}_ggData.json`), details);
    if (meta) await saveData(path.join(base, 'ggMeta', `${eaIdStr}_ggMeta.json`), meta);
    if (esData.length > 0) await saveData(path.join(base, 'esMeta', `${eaIdStr}_esMeta.json`), esData);
    return { id: eaId, status: 'success' };
}

// --- Main Runner ---
(async () => {
    const logPrefix = MODE === 'club' ? '(Club Edition)' : '(Model Retraining)';
    console.log(`🚀 Starting TURBO v2 Data Fetcher ${logPrefix}...`);
    const start = Date.now();
    const root = __dirname;

    const baseRawDirs = [];
    for (const sub of ['club - main', 'club - rest', 'training']) {
        for (const kind of ['ggData', 'ggMeta', 'esMeta']) baseRawDirs.push(path.resolve(root, `../data/raw/${sub}/${kind}`));
    }
    await Promise.all(baseRawDirs.map(ensureDir));

    const idsPath = path.resolve(root, INPUT_FILE);
    const mapsPath = path.resolve(root, MAPS_FILE);

    let ids, maps;
    try { ids = JSON.parse(await fs.readFile(idsPath)); }
    catch (e) { console.error(`❌ Could not load ${INPUT_FILE}: ${e.message}`); return; }
    try { maps = JSON.parse(await fs.readFile(mapsPath)); }
    catch (e) { console.error(`❌ Could not load maps.json: ${e.message}`); return; }

    // Fast filtering (skip already-fetched).
    const todo = [];
    const chunk = 250;
    console.log(`🔎 Filtering ${ids.length} existing players...`);
    for (let i = 0; i < ids.length; i += chunk) {
        const batch = ids.slice(i, i + chunk);
        const results = await Promise.all(batch.map(async (id) => ({ id, skip: await checkSkippability(id, MODE) })));
        todo.push(...results.filter((r) => !r.skip).map((r) => r.id));
    }
    console.log(`📋 Fetching ${todo.length} new players. (Skipped ${ids.length - todo.length})`);
    if (todo.length === 0) { console.log('✅ Nothing to fetch. Exiting.'); return; }

    console.log('🕸️ Launching Browser...');
    const browser = await puppeteer.launch({
        headless: 'new',
        executablePath: resolveChromePath(puppeteer),
        defaultViewport: null,
        args: ['--no-sandbox', '--disable-setuid-sandbox'],
    });

    const primeSem = makeSemaphore(PRIME_CONCURRENCY);
    const primeUrl = URL_GG_DETAILS(todo[0]); // light JSON endpoint used only to clear Cloudflare

    let processed = 0, succeeded = 0;
    const total = todo.length;
    const logProgress = () => {
        if (processed % 50 === 0 || processed === total) {
            const pct = ((processed / total) * 100).toFixed(1);
            const elapsed = (Date.now() - start) / 1000;
            const rate = (processed / elapsed).toFixed(2);
            console.log(`⚡ ${processed}/${total} (${pct}%) | ok ${succeeded} | ${rate} p/s | ${elapsed.toFixed(0)}s`);
        }
    };

    // Each worker gets its OWN context (own cf_clearance), so fut.gg sees many light
    // sessions rather than one bursty one. Priming is bounded by primeSem to avoid a
    // startup burst; then the page is reused for all of that worker's players.
    async function worker() {
        let ctx = await createSafeContext(browser);
        let page = await newFetchPage(ctx);
        let repriming = null;
        const reprime = () => (repriming ||= (async () => {
            await primeSem.acquire();
            try { await primeCloudflare(page, primeUrl); } finally { primeSem.release(); }
        })().finally(() => { repriming = null; }));

        await reprime(); // initial bounded prime

        while (true) {
            const id = todo.shift();
            if (id === undefined) break;
            try {
                const res = await processPlayer(id, page, reprime, maps);
                if (res.status === 'success') succeeded++;
            } catch (e) {
                // context/page crashed — rebuild and continue (this id refetches on a later run).
                try { await ctx.close().catch(() => {}); } catch (_) {}
                ctx = await createSafeContext(browser).catch(() => null);
                page = ctx ? await newFetchPage(ctx).catch(() => null) : null;
                if (!page) break;
            }
            processed++;
            logProgress();
            await sleep(randInt(DELAY_MIN, DELAY_MAX));
        }
        if (ctx) await ctx.close().catch(() => {});
    }

    const nWorkers = Math.min(CONCURRENCY, total);
    console.log(`🔥 Igniting ${nWorkers} workers (per-worker clearance, ${PRIME_CONCURRENCY}-way priming)...`);
    await Promise.all(Array.from({ length: nWorkers }, () => worker()));

    await browser.close();
    console.log(`✅ DONE: ${succeeded}/${total} succeeded in ${((Date.now() - start) / 1000 / 60).toFixed(2)} min.`);
})();
