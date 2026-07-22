// club.2b.fetch.prices.js — R2 CDN Edition
//
// fut.gg no longer serves live prices from /api/fut/player-prices/ (that endpoint
// now returns 403 / a Cloudflare challenge — this is what broke the old scraper).
// Instead the site loads ALL player prices from two static JSON files on its R2 CDN:
//
//   • https://r2.fut.gg/26/player-prices-index.v1.<hash>.json   (shared id index)
//   • https://r2.fut.gg/26/player-prices-<platform>-dyn.v1.<hash>.json  (live prices)
//
// They use a compact columnar layout: the index holds a delta-encoded, ascending
// list of eaIds (`id0` + `d[]`), and the dyn file holds parallel `p[]` (price) and
// `s[]` (status) arrays. ids[i] ↔ p[i] ↔ s[i]. Fetching those two files prices every
// player in ONE shot — replacing 2214 per-id requests (each of which used to spin up
// a fresh browser context). The files are Cloudflare-gated (plain HTTP gets 403), so
// we read them from a primed, stealthed browser context. Content hashes rotate, so we
// discover the current URLs live by loading a price-bearing page.
'use strict';

const fs = require('fs').promises;
const path = require('path');

const puppeteer = require('puppeteer-extra');
const StealthPlugin = require('puppeteer-extra-plugin-stealth');
const { resolveChromePath } = require('./browser');
const { ensureDir, writeJsonAtomic } = require('./scrape_utils');
const { discoverPriceUrls, buildPriceMap } = require('./futgg_prices');
puppeteer.use(StealthPlugin());

// --- Configuration ---
const PLATFORM = process.env.PLATFORM || 'ps5';
const DATA_DIR = path.resolve(__dirname, '..', 'data');
const TRADEABLE_IDS_FILE = path.join(DATA_DIR, 'tradeable_ids.json');
const PRICES_DIR = path.join(DATA_DIR, 'raw', 'prices');

// --- Helpers (ensureDir / writeJsonAtomic come from scrape_utils) ---
async function readJson(file, fallback) {
  try { return JSON.parse(await fs.readFile(file, 'utf-8')); }
  catch { return fallback; }
}

// R2 price-CDN discovery + decode live in futgg_prices.js (shared with
// fetch.fodder.prices.js).

// --- Main ---
(async () => {
  const start = Date.now();
  console.log('💰 Fetching prices from fut.gg R2 CDN price files...');
  await ensureDir(PRICES_DIR);

  const tradeableIds = await readJson(TRADEABLE_IDS_FILE, null);
  if (!Array.isArray(tradeableIds) || tradeableIds.length === 0) {
    console.log('ℹ️ No tradeable_ids.json (or empty). Nothing to price.');
    return;
  }
  console.log(`📋 ${tradeableIds.length} tradeable players to price.`);

  const browser = await puppeteer.launch({
    headless: 'new',
    executablePath: resolveChromePath(puppeteer),
    defaultViewport: null,
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
  });

  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1366, height: 768 });
    await page.setRequestInterception(true);
    page.on('request', (r) => {
      const t = r.resourceType();
      (['image', 'media', 'font', 'stylesheet'].includes(t) ? r.abort() : r.continue()).catch(() => {});
    });

    const urls = await discoverPriceUrls(page, PLATFORM);
    console.log(`🔗 index: ${urls.index.split('/').pop()}`);
    console.log(`🔗 dyn  : ${urls.dyn.split('/').pop()}`);

    const priceMap = await buildPriceMap(page, urls);
    console.log(`📦 Decoded ${priceMap.size} player prices from CDN.`);
    if (priceMap.size === 0) throw new Error('Decoded 0 prices — aborting so existing price files are not clobbered.');

    let found = 0;
    let missing = 0;
    let extinct = 0;
    // Bounded fan-out: 2000+ simultaneous writes can exhaust file descriptors.
    const WRITE_CHUNK = 64;
    for (let i = 0; i < tradeableIds.length; i += WRITE_CHUNK) {
      await Promise.all(tradeableIds.slice(i, i + WRITE_CHUNK).map(async (rawId) => {
        const hit = priceMap.get(Number(rawId));
        if (hit) found++; else missing++;
        const priceData = {
          price: hit ? hit.price : 0,
          isExtinct: hit ? hit.isExtinct : true,
        };
        if (priceData.isExtinct) extinct++;
        await writeJsonAtomic(path.join(PRICES_DIR, `${rawId}.json`), priceData);
      }));
    }

    const secs = ((Date.now() - start) / 1000).toFixed(2);
    console.log(`🎉 Priced ${found}/${tradeableIds.length} players (missing ${missing}, no-price/extinct ${extinct}) in ${secs}s.`);
  } finally {
    await browser.close().catch(() => {});
  }
})().catch((e) => { console.error('❌ Price fetch failed:', e.message); process.exit(1); });
