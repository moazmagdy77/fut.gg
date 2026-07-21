// fetch.fodder.prices.js
// Fetches the current cheapest BIN per overall rating from fut.gg (for the Fodder
// Value view) and writes data/fodder_prices.json. Uses the cheapest-by-rating
// overview API in ONE call: data[rating] is an ascending-by-price list, so
// data[rating][0].price is the cheapest card at that rating. Cloudflare-gated, so we
// read it from a primed stealth browser context (plain HTTP gets 403).
'use strict';

const path = require('path');
const puppeteer = require('puppeteer-extra');
const StealthPlugin = require('puppeteer-extra-plugin-stealth');
const { resolveChromePath } = require('./browser');
const { sleep, writeJsonAtomic } = require('./scrape_utils');
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

    const out = { platform: PLATFORM, fetchedAt: new Date().toISOString(), byRating: res.byRating };
    await writeJsonAtomic(OUT_FILE, out);
    const ratings = Object.keys(out.byRating).map(Number).sort((a, b) => a - b);
    console.log(`✅ ${ratings.length} ratings (${ratings[0]}–${ratings[ratings.length - 1]}) → ${path.relative(process.cwd(), OUT_FILE)}`);
  } finally {
    await browser.close().catch(() => {});
  }
})().catch((e) => { console.error('❌ Fodder price fetch failed:', e.message); process.exit(1); });
