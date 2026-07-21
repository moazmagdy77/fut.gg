// model.1.get_ids.js (Fixed: Robust Wait Strategy)
// Fixes "No IDs found" by waiting for valid Regex matches, not just generic links.

'use strict';

const puppeteer = require('puppeteer-extra');
const StealthPlugin = require('puppeteer-extra-plugin-stealth');
const fs = require('fs');
const fsp = fs.promises;
const path = require('path');

puppeteer.use(StealthPlugin());
const { resolveChromePath } = require('./browser');

// --- Configuration ---
const TOP_PLAYER_PAGES = Number(process.env.TOTAL_PAGES || 334);
const BATCH_SIZE       = Number(process.env.BATCH_SIZE   || 25);
const CONCURRENCY      = Math.max(1, Number(process.env.CONCURRENCY || 5));
const MAX_RETRIES      = Math.max(1, Number(process.env.MAX_RETRIES  || 5)); // Increased retries slightly
const HEADLESS         = process.env.HEADLESS === 'false' ? false : true; 

const BASE_URL = 'https://www.fut.gg/players/?page=';
// The ID regex we validated: 26-{id} at the end of the string
const ID_REGEX = /26-(\d+)\/?$/;

const DATA_DIR   = path.resolve(__dirname, '..', 'data');
const TEMP_DIR   = path.join(DATA_DIR, 'raw', 'temp_ids');
const FINAL_FILE = path.join(DATA_DIR, 'player_ids.json');

// TIMEOUTS
const NAV_TIMEOUT_MS = 60_000;
const WAIT_READY_MS  = 30_000;

// Polite delay
const POLITE_DELAY_MIN_MS = 500;
const POLITE_DELAY_MAX_MS = 1500;

// --- Helpers ---
const sleep = (ms) => new Promise(res => setTimeout(res, ms));
const randInt = (min, max) => Math.floor(Math.random() * (max - min + 1)) + min;
const backoff = (attempt) => (3_000 * Math.pow(2, attempt - 1)) + randInt(0, 1_000);

const ensureDir = async (dir) => {
  if (!fs.existsSync(dir)) await fsp.mkdir(dir, { recursive: true });
};

const chunk = (arr, size) => {
  const out = [];
  for (let i = 0; i < arr.length; i += size) out.push(arr.slice(i, i + size));
  return out;
};

const tempPathFor = (i) => path.join(TEMP_DIR, `ids_page_${i}.json`);

const writeJsonAtomic = async (filePath, obj) => {
  const tmp = `${filePath}.tmp`;
  await fsp.writeFile(tmp, JSON.stringify(obj, null, 2), 'utf8');
  await fsp.rename(tmp, filePath);
};

// --- Browser Setup ---
async function createIsolatedContext(browser) {
  if (typeof browser.createBrowserContext === 'function') return await browser.createBrowserContext();
  if (typeof browser.createIncognitoBrowserContext === 'function') return await browser.createIncognitoBrowserContext();
  return browser.defaultBrowserContext();
}

async function newConfiguredPage(context) {
  const page = await context.newPage();
  await page.setViewport({ width: 1366, height: 768 });
  
  // Aggressive blocking for speed
  await page.setRequestInterception(true);
  page.on('request', req => {
    const type = req.resourceType();
    if (['image', 'media', 'font', 'stylesheet', 'other'].includes(type)) req.abort();
    else req.continue();
  });
  return page;
}

// --- Core Scraper ---
async function fetchOnePage(i, page) {
  const url = `${BASE_URL}${i}`;
  const outFile = tempPathFor(i);

  for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    try {
      // 1. Navigate
      await page.goto(url, { timeout: NAV_TIMEOUT_MS, waitUntil: 'domcontentloaded' });
      
      // 2. THE FIX: Wait for DATA, not just selectors.
      // This waits until the page has at least 5 links that match our specific ID pattern.
      // It completely ignores the header "Players" link.
      await page.waitForFunction((regexStr) => {
          const re = new RegExp(regexStr);
          const anchors = Array.from(document.querySelectorAll('a[href*="/players/"]'));
          // Count how many links match the "26-{id}" pattern
          const validMatches = anchors.filter(a => re.test(a.href)).length;
          // Only return true if we found a reasonable number (e.g., > 5) implies grid loaded
          return validMatches > 5; 
      }, { timeout: WAIT_READY_MS }, ID_REGEX.source);

      // 3. Extract (Now guaranteed to have data)
      const extractedIds = await page.evaluate((regexStr) => {
        const regex = new RegExp(regexStr); 
        const links = Array.from(document.querySelectorAll('a[href*="/players/"]'));
        
        return links
          .map(el => {
            const match = el.getAttribute('href').match(regex);
            return match ? parseInt(match[1], 10) : null;
          })
          .filter(id => id !== null);
      }, ID_REGEX.source);

      if (!extractedIds || extractedIds.length === 0) {
        throw new Error('Logic error: waitForFunction passed but extraction failed.');
      }
      
      const uniqueLocal = [...new Set(extractedIds)];
      
      await writeJsonAtomic(outFile, uniqueLocal);
      console.log(`✅ [p${i}] Got ${uniqueLocal.length} IDs`);
      return true;

    } catch (err) {
      console.warn(`⚠️ [p${i}] Attempt ${attempt} failed: ${err.message}`);
      if (attempt < MAX_RETRIES) await sleep(backoff(attempt));
    }
  }
  console.error(`❌ [p${i}] Failed after ${MAX_RETRIES} attempts.`);
  return false;
}

// --- Worker & Orchestration ---
async function workerLoop(tasks, browser) {
  const context = await createIsolatedContext(browser);
  let page = await newConfiguredPage(context); // one reused page per worker
  try {
    while (true) {
      const i = tasks.getNext();
      if (i === null) break;
      if (fs.existsSync(tempPathFor(i))) continue;
      try {
        await fetchOnePage(i, page);
      } catch (e) {
        // page crashed — recreate in the same context and continue
        try { await page.close().catch(() => {}); } catch (_) {}
        page = await newConfiguredPage(context);
      }
      await sleep(randInt(POLITE_DELAY_MIN_MS, POLITE_DELAY_MAX_MS));
    }
  } finally {
    try { await page.close().catch(() => {}); } catch (_) {}
    if (context.close) await context.close().catch(() => {});
  }
}

function makeTaskDispenser(items) {
  let idx = 0;
  return { getNext: () => (idx < items.length ? items[idx++] : null) };
}

// --- Final Merger ---
async function mergeResults() {
  console.log('\n🔗 Merging temporary files...');
  const allIds = new Set();
  
  // Ensure the directory exists before reading
  if (!fs.existsSync(TEMP_DIR)) {
      console.log("⚠️ Temp directory not found (no new pages fetched).");
      return;
  }

  const files = fs.readdirSync(TEMP_DIR).filter(f => f.startsWith('ids_page_') && f.endsWith('.json'));

  if (files.length === 0) {
    console.log('ℹ️ No new data files found to merge.');
    return;
  }

  for (const file of files) {
    try {
      const data = JSON.parse(fs.readFileSync(path.join(TEMP_DIR, file)));
      data.forEach(id => allIds.add(id));
    } catch (e) {
      console.warn(`⚠️ Corrupt file skipped: ${file}`);
    }
  }

  // Load existing IDs if we want to append (Optional, but safer)
  if (fs.existsSync(FINAL_FILE)) {
      try {
          const existing = JSON.parse(fs.readFileSync(FINAL_FILE));
          existing.forEach(id => allIds.add(id));
      } catch(e) {}
  }

  const sortedIds = Array.from(allIds).sort((a, b) => a - b);
  await fsp.writeFile(FINAL_FILE, JSON.stringify(sortedIds, null, 2));
  
  console.log(`\n🎉 SUCCESS: Total unique IDs: ${sortedIds.length}`);
  console.log(`📂 Saved to: ${FINAL_FILE}`);
}

// --- Main ---
(async () => {
  console.log("🚀 Starting Unified ID Scraper (single browser, page reuse)...");
  await ensureDir(TEMP_DIR);

  const allPages = Array.from({ length: TOP_PLAYER_PAGES }, (_, k) => k + 1);
  const todo = allPages.filter(i => !fs.existsSync(tempPathFor(i)));

  if (todo.length > 0) {
    // Launch Chrome ONCE and reuse it (+ one page per worker). The old version
    // relaunched the browser and recreated a page per 25-page batch — pure overhead.
    const browser = await puppeteer.launch({
        headless: HEADLESS,
        executablePath: resolveChromePath(puppeteer),
        defaultViewport: null,
        args: ['--no-sandbox', '--disable-setuid-sandbox'],
    });
    try {
      const nWorkers = Math.min(CONCURRENCY, todo.length);
      console.log(`📦 Scraping ${todo.length} pages with ${nWorkers} workers...`);
      const dispenser = makeTaskDispenser(todo);
      const workers = Array(nWorkers).fill(null).map(() => workerLoop(dispenser, browser));
      await Promise.all(workers);
    } finally {
      if (browser) await browser.close().catch(() => {});
    }
  } else {
    console.log('✅ All pages already cached — merging.');
  }

  await mergeResults();
})();