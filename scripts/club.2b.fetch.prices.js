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
const { sleep, ensureDir, writeJsonAtomic } = require('./scrape_utils');
puppeteer.use(StealthPlugin());

// --- Configuration ---
const PLATFORM = process.env.PLATFORM || 'ps5';
const DATA_DIR = path.resolve(__dirname, '..', 'data');
const TRADEABLE_IDS_FILE = path.join(DATA_DIR, 'tradeable_ids.json');
const PRICES_DIR = path.join(DATA_DIR, 'raw', 'prices');
const NAV_TIMEOUT_MS = 45_000;

// --- Helpers (sleep / ensureDir / writeJsonAtomic come from scrape_utils) ---
async function readJson(file, fallback) {
  try { return JSON.parse(await fs.readFile(file, 'utf-8')); }
  catch { return fallback; }
}

// The R2 file hashes rotate (dyn ~hourly), so we can't hardcode the URLs. Load a
// price-bearing page (the players grid shows a price on every card) and collect the
// R2 requests the site itself makes.
async function discoverPriceUrls(page) {
  const seen = new Set();
  const onResp = (r) => {
    const u = r.url();
    if (u.includes('r2.fut.gg') && u.includes('player-prices')) seen.add(u);
  };
  const pick = () => {
    const arr = [...seen];
    const index = arr.find((u) => u.includes('index'));
    const dyn = arr.find((u) => u.includes('dyn') && u.includes(`-${PLATFORM}-`)) || arr.find((u) => u.includes('dyn'));
    return { index, dyn };
  };
  page.on('response', onResp);
  try {
    // Prime Cloudflare on the homepage, then hit pages that render live prices. The
    // players grid is the most reliable trigger for the global R2 price files; a
    // known market player's page is the fallback.
    await page.goto('https://www.fut.gg/', { waitUntil: 'domcontentloaded', timeout: NAV_TIMEOUT_MS }).catch(() => {});
    const targets = ['https://www.fut.gg/players/'];
    try {
      const eaId = await page.evaluate(async () => {
        // a cheap, definitely-tradeable mid-gold — its page always loads a live price
        const r = await fetch('https://www.fut.gg/api/fut/players/v2/26/?page=1&market_players=true&overall__gte=84&overall__lte=87&sorts=current_price&price__gte=200', { headers: { accept: 'application/json' } });
        const j = await r.json();
        const hit = (j.data || []).find((p) => p && p.eaId);
        return hit ? hit.eaId : null;
      });
      if (eaId) targets.push(`https://www.fut.gg/players/26-${eaId}/`);
    } catch { /* players grid alone is usually enough */ }

    for (const t of targets) {
      await page.goto(t, { waitUntil: 'domcontentloaded', timeout: NAV_TIMEOUT_MS }).catch(() => {});
      for (let i = 0; i < 60; i++) {
        const u = pick();
        if (u.index && u.dyn) break;
        await sleep(250);
      }
      const u = pick();
      if (u.index && u.dyn) break;
    }
  } finally {
    page.off('response', onResp);
  }
  const urls = pick();
  if (!urls.index || !urls.dyn) {
    console.warn('   R2 price URLs seen:', [...seen]);
    throw new Error(`Could not discover R2 price URLs (index=${urls.index}, dyn=${urls.dyn}). fut.gg may have changed its price scheme.`);
  }
  return urls;
}

// Fetch both columnar files inside the (Cloudflare-cleared) browser context and
// decode them into a Map<eaId, {price, isExtinct, status}>.
async function buildPriceMap(page, urls) {
  const raw = await page.evaluate(async (u) => {
    const [idx, dyn] = await Promise.all([
      fetch(u.index).then((r) => r.json()),
      fetch(u.dyn).then((r) => r.json()),
    ]);
    return { id0: idx.id0, d: idx.d || [], p: dyn.p || [], s: dyn.s || [] };
  }, urls);

  if (!Number.isFinite(raw.id0) || raw.d.length + 1 !== raw.p.length) {
    // Not fatal, but warn: index/dyn are expected to be aligned (ids = d.length + 1).
    console.warn(`⚠️ Index/price length mismatch (ids=${raw.d.length + 1}, prices=${raw.p.length}). Proceeding with the shorter length.`);
  }

  const map = new Map();
  const setAt = (ea, i) => {
    const pr = raw.p[i];
    const price = (typeof pr === 'number' && pr > 0) ? pr : 0;
    map.set(ea, { price, isExtinct: price === 0, status: raw.s[i] ?? null });
  };
  let ea = raw.id0;
  setAt(ea, 0);
  for (let i = 0; i < raw.d.length; i++) { ea += raw.d[i]; setAt(ea, i + 1); }
  return map;
}

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

    const urls = await discoverPriceUrls(page);
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
