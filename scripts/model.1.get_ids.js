// model.1.get_ids.parallel.js --> Parallel Robust Fetcher (Puppeteer + Stealth)
// Requires: puppeteer-extra, puppeteer-extra-plugin-stealth
// Usage: CONCURRENCY=4 node model.1.get_ids.parallel.js

'use strict';

const puppeteer = require('puppeteer-extra');
const StealthPlugin = require('puppeteer-extra-plugin-stealth');
const fs = require('fs');
const fsp = fs.promises;
const path = require('path');
const os = require('os');

puppeteer.use(StealthPlugin());

// --- Configuration (env overrides allowed) ---
const TOP_PLAYER_PAGES = Number(process.env.TOTAL_PAGES || 334); // total pages to scrape
const BATCH_SIZE       = Number(process.env.BATCH_SIZE   || 25); // pages per browser lifecycle
const CONCURRENCY      = Math.max(1, Number(process.env.CONCURRENCY || 4)); // parallel workers per batch
const MAX_RETRIES      = Math.max(1, Number(process.env.MAX_RETRIES  || 3));
const HEADLESS         = process.env.HEADLESS === 'false' ? false : true; // set HEADLESS=false to watch
const BLOCK_RESOURCES  = process.env.BLOCK_RESOURCES === 'false' ? false : true; // block heavy assets for speed

const baseUrl = 'https://www.fut.gg/players/?page=';

const dataDir   = path.resolve(__dirname, '..', 'data');
const outputDir = path.join(dataDir, 'raw', 'r_objects');

const SELECTOR_READY = 'a[href*="/players/"]'; // wait for cards presence
const NAV_TIMEOUT_MS = 90_000;
const WAIT_READY_MS  = 60_000;

// polite delay (randomized) between page tasks per worker
const POLITE_DELAY_MIN_MS = 600;
const POLITE_DELAY_MAX_MS = 1800;

// --- Helpers ---
const sleep = (ms) => new Promise(res => setTimeout(res, ms));
const randInt = (min, max) => Math.floor(Math.random() * (max - min + 1)) + min;
const backoff = (attempt) => (5_000 * Math.pow(2, attempt - 1)) + randInt(0, 1_000);

const ensureDir = async (dir) => {
  if (!fs.existsSync(dir)) await fsp.mkdir(dir, { recursive: true });
};

const chunk = (arr, size) => {
  const out = [];
  for (let i = 0; i < arr.length; i += size) out.push(arr.slice(i, i + size));
  return out;
};

const outputPathFor = (i) => path.join(outputDir, `page_${i}.json`);

// Write atomically (tmp -> rename) to avoid partial files on crash
const writeJsonAtomic = async (filePath, obj) => {
  const tmp = `${filePath}.tmp`;
  await fsp.writeFile(tmp, JSON.stringify(obj, null, 2), 'utf8');
  await fsp.rename(tmp, filePath);
};

// Create and configure a fresh page
async function newConfiguredPage(context, browserUserAgent) {
  const page = await context.newPage();

  // Use a UA without "Headless" if present. Stealth handles most bits, this just helps.
  const ua = (browserUserAgent || '').replace(/Headless/i, '');
  if (ua) await page.setUserAgent(ua);

  await page.setViewport({ width: 1366, height: 768 });

  if (BLOCK_RESOURCES) {
    await page.setRequestInterception(true);
    page.on('request', req => {
      const type = req.resourceType();
      // Allow only what's likely needed for app logic
      if (type === 'image' || type === 'media' || type === 'font') {
        req.abort();
      } else {
        req.continue();
      }
    });
  }

  return page;
}

// Core fetch with retries
async function fetchOnePage(i, context, browserUserAgent) {
  const url = `${baseUrl}${i}`;
  const outputFilePath = outputPathFor(i);

  for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    let page = null;
    try {
      console.log(`🔄 [p${i}] Visiting ${url} (Attempt ${attempt}/${MAX_RETRIES})`);
      page = await newConfiguredPage(context, browserUserAgent);

      // Go early; we'll still await a specific selector.
      await page.goto(url, { timeout: NAV_TIMEOUT_MS, waitUntil: 'domcontentloaded' });

      // Wait for an element that reliably indicates the content is there
      await page.waitForSelector(SELECTOR_READY, { timeout: WAIT_READY_MS });
      await sleep(3_000); // small settle time for client-side global state

      const R_object = await page.evaluate(() => window.$R);

      if (!R_object) {
        throw new Error('window.$R not found on the page.');
      }

      await writeJsonAtomic(outputFilePath, R_object);
      console.log(`✅ [p${i}] Saved ${path.basename(outputFilePath)}`);
      return; // success
    } catch (err) {
      console.warn(`⚠️ [p${i}] Attempt ${attempt} failed: ${err && err.message ? err.message : err}`);
      if (attempt === MAX_RETRIES) {
        const screenshotPath = path.join(process.cwd(), `error_screenshot_page_${i}.png`);
        try {
          if (page) await page.screenshot({ path: screenshotPath, fullPage: true });
          console.log(`📸 [p${i}] Screenshot saved to ${screenshotPath}`);
        } catch (e) {
          console.warn(`⚠️ [p${i}] Could not capture screenshot: ${e.message}`);
        }
      } else {
        const wait = backoff(attempt);
        console.log(`⏳ [p${i}] Backing off for ${wait}ms before retry...`);
        await sleep(wait);
      }
    } finally {
      if (page) {
        try { await page.close(); } catch (_) {}
      }
    }
  }

  console.error(`❌ [p${i}] All attempts failed. Skipping.`);
}

// Worker that pulls tasks from a shared index
async function workerLoop(workerId, tasks, browser) {
  const context = await browser.createIncognitoBrowserContext();
  const browserUserAgent = await browser.userAgent();

  let next = 0;
  // Sneaky trick: each worker advances an atomic cursor via closure
  // We'll pass a function that returns next index each time.
  const getNext = tasks.getNext;

  try {
    for (;;) {
      const i = getNext();
      if (i === null) break;

      const out = outputPathFor(i);
      if (fs.existsSync(out)) {
        console.log(`🟡 [p${i}] Already downloaded. Skipping.`);
      } else {
        await fetchOnePage(i, context, browserUserAgent);
        // Polite per-task delay (randomized) to stagger request timing
        await sleep(randInt(POLITE_DELAY_MIN_MS, POLITE_DELAY_MAX_MS));
      }
    }
  } finally {
    try { await context.close(); } catch (_) {}
  }
}

// Build a simple task dispenser for a batch
function makeTaskDispenser(pageNumbers) {
  let idx = 0;
  return {
    getNext: () => {
      if (idx >= pageNumbers.length) return null;
      return pageNumbers[idx++];
    }
  };
}

// --- Main ---
(async () => {
  await ensureDir(outputDir);

  const allPages = Array.from({ length: TOP_PLAYER_PAGES }, (_, k) => k + 1);
  const batches = chunk(allPages, BATCH_SIZE);

  let browser = null;

  for (let b = 0; b < batches.length; b++) {
    const batch = batches[b];

    // Filter out pages we already have to avoid launching a browser for nothing
    const todo = batch.filter(i => !fs.existsSync(outputPathFor(i)));
    if (todo.length === 0) {
      console.log(`🧹 Batch ${b + 1}/${batches.length}: nothing to do (all ${batch.length} pages exist).`);
      continue;
    }

    // Launch a fresh browser for this batch (memory & fingerprint reset)
    if (browser) {
      console.log(`\n🔄 Restarting browser for batch ${b + 1}...`);
      try { await browser.close(); } catch (_) {}
    }

    console.log(`🚀 Launching new browser instance for batch ${b + 1}/${batches.length} (pages ${batch[0]}–${batch[batch.length - 1]})...`);
    browser = await puppeteer.launch({
      headless: HEADLESS,
      // TIP: if you run in containers/CI, enable the two flags below:
      // args: ['--no-sandbox', '--disable-setuid-sandbox'],
      defaultViewport: null
    });

    // Spin up workers for this batch
    const dispenser = makeTaskDispenser(todo);
    const workerCount = Math.min(CONCURRENCY, Math.max(1, todo.length));
    console.log(`🏃 Processing ${todo.length} page(s) with concurrency=${workerCount}...`);

    const workers = [];
    for (let w = 0; w < workerCount; w++) {
      workers.push(workerLoop(w + 1, dispenser, browser));
    }
    await Promise.all(workers);
    console.log(`✅ Finished batch ${b + 1}/${batches.length}.`);
  }

  // Final shutdown
  if (browser) {
    try {
      console.log('🚪 Closing final browser instance...');
      await Promise.race([
        browser.close(),
        new Promise((_, reject) => setTimeout(() => reject(new Error('Browser close timed out')), 15_000))
      ]);
      console.log('✅ Browser closed successfully.');
    } catch (e) {
      console.warn(`⚠️ Could not close browser gracefully: ${e.message}.`);
    }
  }

  console.log('🎉 Finished fetching raw data objects.');
  // Force exit to prevent hanging on stray handles
  process.exit(0);
})();
