// model.1.get_ids.js
// Collects the top player eaIds from fut.gg into data/player_ids.json.
//
// fut.gg exposes the players list as a JSON API — /api/fut/players/v2/26/?page=N —
// returning { data: [{ eaId, ... }], total, next }. We read eaIds straight from that
// JSON instead of rendering the React grid and scraping <a href> nodes: no DOM wait,
// no HTML parsing, far less latency. The API is Cloudflare-gated, so (like the other
// scrapers) each worker navigates to fut.gg ONCE to clear Cloudflare, then pulls pages
// with in-page fetch() from that primed, same-origin context.

'use strict';

const puppeteer = require('puppeteer-extra');
const StealthPlugin = require('puppeteer-extra-plugin-stealth');
const fs = require('fs');
const path = require('path');

puppeteer.use(StealthPlugin());
const { resolveChromePath } = require('./browser');
const { sleep, randInt, ensureDir, writeJsonAtomic, createIsolatedContext } = require('./scrape_utils');

// --- Configuration ---
const TOP_PLAYER_PAGES = Number(process.env.TOTAL_PAGES || 334); // fut.gg lists ~10k players (30/page)
const CONCURRENCY = Math.max(1, Number(process.env.CONCURRENCY || 5));
const MAX_RETRIES = Math.max(1, Number(process.env.MAX_RETRIES || 5));
const HEADLESS = process.env.HEADLESS === 'false' ? false : 'new';

const FUTGG_ORIGIN = 'https://www.fut.gg/';
const API_URL = (page) => `https://www.fut.gg/api/fut/players/v2/26/?page=${page}`;

const DATA_DIR = path.resolve(__dirname, '..', 'data');
const TEMP_DIR = path.join(DATA_DIR, 'raw', 'temp_ids');
const FINAL_FILE = path.join(DATA_DIR, 'player_ids.json');

const NAV_TIMEOUT_MS = 60_000;
const POLITE_DELAY_MIN_MS = 150;
const POLITE_DELAY_MAX_MS = 500;

const backoff = (attempt) => (2_000 * Math.pow(2, attempt - 1)) + randInt(0, 1_000);
const tempPathFor = (i) => path.join(TEMP_DIR, `ids_page_${i}.json`);

// --- Browser setup ---
async function newConfiguredPage(context) {
  const page = await context.newPage();
  await page.setViewport({ width: 1366, height: 768 });
  await page.setRequestInterception(true);
  page.on('request', (req) => {
    const type = req.resourceType();
    (['image', 'media', 'font', 'stylesheet', 'other'].includes(type) ? req.abort() : req.continue()).catch(() => {});
  });
  return page;
}

// --- Core: fetch one page's eaIds via in-page JSON fetch ---
async function fetchOnePage(i, page, reprime) {
  const outFile = tempPathFor(i);

  for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    try {
      const result = await page.evaluate(async (url) => {
        try {
          const r = await fetch(url, { headers: { accept: 'application/json' }, credentials: 'include' });
          if (!r.ok) return { __status: r.status };
          const t = await r.text();
          try { return JSON.parse(t); } catch { return { __status: 'nonjson' }; }
        } catch (e) { return { __err: String((e && e.message) || e) }; }
      }, API_URL(i));

      // Cloudflare / transient block → re-prime this worker's context and retry.
      if (!result || result.__err || typeof result.__status !== 'undefined') {
        throw new Error(`blocked (${result && (result.__status || result.__err)})`);
      }

      const data = Array.isArray(result.data) ? result.data : [];
      const ids = [...new Set(data.map((p) => p && p.eaId).filter((v) => Number.isFinite(v)))];
      await writeJsonAtomic(outFile, ids);
      console.log(ids.length ? `✅ [p${i}] ${ids.length} IDs` : `✅ [p${i}] empty (past end of list)`);
      return true;
    } catch (err) {
      console.warn(`⚠️ [p${i}] attempt ${attempt} failed: ${err.message}`);
      if (attempt < MAX_RETRIES) {
        await reprime();
        await sleep(backoff(attempt));
      }
    }
  }
  console.error(`❌ [p${i}] failed after ${MAX_RETRIES} attempts.`);
  return false;
}

// --- Worker: own context + primed page, reused across all its pages ---
async function workerLoop(tasks, browser) {
  let context = null;
  let page = null;
  try {
    context = await createIsolatedContext(browser);
    page = await newConfiguredPage(context);
    const reprime = async () => {
      try { await page.goto(FUTGG_ORIGIN, { waitUntil: 'domcontentloaded', timeout: NAV_TIMEOUT_MS }); }
      catch { /* next fetch attempt will surface the block again */ }
    };
    await reprime(); // clear Cloudflare once up front

    while (true) {
      const i = tasks.getNext();
      if (i === null) break;
      if (fs.existsSync(tempPathFor(i))) continue;
      try {
        await fetchOnePage(i, page, reprime);
      } catch (e) {
        // page/context crashed — rebuild in the same context and continue. If even the
        // rebuild fails (e.g. a CDP protocol timeout under heavy machine load), bow out
        // of this worker instead of crashing the run; its pages get retried next run.
        try { await page.close().catch(() => {}); } catch (_) {}
        page = await newConfiguredPage(context).catch(() => null);
        if (!page) break;
        await reprime();
      }
      await sleep(randInt(POLITE_DELAY_MIN_MS, POLITE_DELAY_MAX_MS));
    }
  } catch (e) {
    // A worker must never reject the top-level Promise.all — that would crash the run.
    console.warn(`⚠️ worker aborted: ${e.message}`);
  } finally {
    try { if (page) await page.close().catch(() => {}); } catch (_) {}
    try { if (context && context.close) await context.close().catch(() => {}); } catch (_) {}
  }
}

function makeTaskDispenser(items) {
  let idx = 0;
  return { getNext: () => (idx < items.length ? items[idx++] : null) };
}

// --- Merge temp pages → player_ids.json (union with any existing IDs) ---
async function mergeResults() {
  console.log('\n🔗 Merging page files...');
  if (!fs.existsSync(TEMP_DIR)) { console.log('⚠️ No temp dir — nothing fetched.'); return; }

  const files = fs.readdirSync(TEMP_DIR).filter((f) => f.startsWith('ids_page_') && f.endsWith('.json'));
  if (files.length === 0) { console.log('ℹ️ No page files to merge.'); return; }

  const allIds = new Set();
  for (const file of files) {
    try { JSON.parse(fs.readFileSync(path.join(TEMP_DIR, file))).forEach((id) => allIds.add(id)); }
    catch { console.warn(`⚠️ Corrupt file skipped: ${file}`); }
  }
  if (fs.existsSync(FINAL_FILE)) {
    try { JSON.parse(fs.readFileSync(FINAL_FILE)).forEach((id) => allIds.add(id)); } catch { /* ignore */ }
  }

  const sortedIds = Array.from(allIds).sort((a, b) => a - b);
  await writeJsonAtomic(FINAL_FILE, sortedIds);
  console.log(`\n🎉 SUCCESS: ${sortedIds.length} unique IDs → ${path.relative(process.cwd(), FINAL_FILE)}`);
}

// --- Main ---
(async () => {
  console.log('🚀 Starting ID scraper (JSON players API, single browser, page reuse)...');
  await ensureDir(TEMP_DIR);

  const allPages = Array.from({ length: TOP_PLAYER_PAGES }, (_, k) => k + 1);
  const todo = allPages.filter((i) => !fs.existsSync(tempPathFor(i)));

  if (todo.length > 0) {
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
      await Promise.all(Array.from({ length: nWorkers }, () => workerLoop(dispenser, browser)));
    } finally {
      await browser.close().catch(() => {});
    }
  } else {
    console.log('✅ All pages already cached — merging.');
  }

  await mergeResults();
})();
