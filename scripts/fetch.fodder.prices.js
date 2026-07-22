// fetch.fodder.prices.js
// Fetches the current cheapest BIN per overall rating from fut.gg (for the Fodder
// Value view) and writes data/fodder_prices.json.
//   • Primary: the cheapest-by-rating overview API in ONE call (covers ~81–93) —
//     data[rating][0].price = cheapest card at that rating.
//   • Fallback for ratings the overview omits (94–99): the players search sorted by
//     current_price (cheapest first) gives the cheapest eaId, whose price we read from
//     the R2 price file (the search response itself carries no price).
// Cloudflare-gated, so everything runs from a primed stealth browser context.
'use strict';

const path = require('path');
const puppeteer = require('puppeteer-extra');
const StealthPlugin = require('puppeteer-extra-plugin-stealth');
const { resolveChromePath } = require('./browser');
const { sleep, writeJsonAtomic } = require('./scrape_utils');
const { discoverPriceUrls, buildPriceMap } = require('./futgg_prices');
puppeteer.use(StealthPlugin());

const PLATFORM = process.env.PLATFORM || 'ps5';
const OUT_FILE = path.resolve(__dirname, '..', 'data', 'fodder_prices.json');
const OVERVIEW_URL = `https://www.fut.gg/api/fut/market/cheapest-by-rating/v2/overview/?platform=${PLATFORM}`;
const NAV_TIMEOUT_MS = 45_000;

(async () => {
  console.log('🍞 Fetching cheapest-by-rating (fodder) prices from fut.gg...');
  const browser = await puppeteer.launch({
    headless: 'new',
    executablePath: resolveChromePath(puppeteer),
    defaultViewport: null,
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
  });
  try {
    const page = await browser.newPage();
    await page.setRequestInterception(true);
    page.on('request', (r) => {
      const t = r.resourceType();
      (['image', 'media', 'font', 'stylesheet'].includes(t) ? r.abort() : r.continue()).catch(() => {});
    });

    // Prime Cloudflare, then read the overview via same-origin in-page fetch.
    await page.goto('https://www.fut.gg/', { waitUntil: 'domcontentloaded', timeout: NAV_TIMEOUT_MS }).catch(() => {});
    await sleep(1000);

    const res = await page.evaluate(async (url) => {
      try {
        const r = await fetch(url, { headers: { accept: 'application/json' }, credentials: 'include' });
        if (!r.ok) return { __status: r.status };
        const j = await r.json();
        const data = j.data || {};
        const byRating = {};
        for (const [rating, arr] of Object.entries(data)) {
          if (Array.isArray(arr) && arr.length && typeof arr[0].price === 'number') {
            byRating[rating] = arr[0].price;
          }
        }
        return { byRating };
      } catch (e) { return { __err: String((e && e.message) || e) }; }
    }, OVERVIEW_URL);

    if (!res || res.__status || res.__err || !res.byRating || Object.keys(res.byRating).length === 0) {
      throw new Error(`No fodder prices returned (status=${res && res.__status}, err=${res && res.__err}).`);
    }

    const byRating = res.byRating;

    // Fill ratings the overview omits (typically 94–99) via the players search
    // (cheapest-first) + the R2 price file. Skips extinct (price 0) candidates.
    const missing = [];
    for (let r = 81; r <= 99; r++) if (!(String(r) in byRating)) missing.push(r);

    if (missing.length) {
      console.log(`   overview covered ${Object.keys(byRating).length} ratings; filling ${missing.join(', ')} via search + R2...`);
      let priceMap = null;
      try {
        const urls = await discoverPriceUrls(page, PLATFORM);
        priceMap = await buildPriceMap(page, urls);
      } catch (e) {
        console.warn(`   ⚠️ could not load R2 prices for the fallback: ${e.message}`);
      }
      if (priceMap) {
        for (const r of missing) {
          try {
            const eaIds = await page.evaluate(async (rating) => {
              const u = `https://www.fut.gg/api/fut/players/v2/26/?page=1&sorts=current_price&market_players=true&overall__gte=${rating}&overall__lte=${rating}&price__gte=200`;
              const resp = await fetch(u, { headers: { accept: 'application/json' } });
              if (!resp.ok) return [];
              const j = await resp.json();
              return (j.data || []).map((p) => p && p.eaId).filter(Boolean);
            }, r);
            // cheapest-first order; take the first candidate with a non-zero R2 price
            for (const eaId of eaIds) {
              const hit = priceMap.get(Number(eaId));
              if (hit && hit.price > 0) { byRating[String(r)] = hit.price; break; }
            }
          } catch (e) { /* leave this rating unpriced (no tradeable fodder there) */ }
          await sleep(400);
        }
      }
    }

    const out = { platform: PLATFORM, fetchedAt: new Date().toISOString(), byRating };
    await writeJsonAtomic(OUT_FILE, out);
    const ratings = Object.keys(out.byRating).map(Number).sort((a, b) => a - b);
    const filled = missing.filter((r) => String(r) in byRating);
    console.log(`✅ ${ratings.length} ratings (${ratings[0]}–${ratings[ratings.length - 1]})${filled.length ? `, incl. fallback ${filled.join(', ')}` : ''} → ${path.relative(process.cwd(), OUT_FILE)}`);
  } finally {
    await browser.close().catch(() => {});
  }
})().catch((e) => { console.error('❌ Fodder price fetch failed:', e.message); process.exit(1); });
