// futgg_prices.js
// Shared helpers for fut.gg's R2 price CDN, used by club.2b.fetch.prices.js and
// fetch.fodder.prices.js. fut.gg serves every player's price as two static columnar
// JSON files on r2.fut.gg (a delta-encoded eaId index + parallel price/status arrays);
// the old /api/fut/player-prices endpoint now 403s. Content hashes rotate, so we
// discover the current URLs by loading a price-bearing page, then decode in-browser
// (the files are Cloudflare-gated — plain HTTP gets 403).
'use strict';

const { sleep } = require('./scrape_utils');

const NAV_TIMEOUT_MS = 45_000;

// Load a price-bearing page (the players grid renders a price on every card) and
// collect the two R2 price requests the site itself makes.
async function discoverPriceUrls(page, platform = 'ps5') {
  const seen = new Set();
  const onResp = (r) => {
    const u = r.url();
    if (u.includes('r2.fut.gg') && u.includes('player-prices')) seen.add(u);
  };
  const pick = () => {
    const arr = [...seen];
    const index = arr.find((u) => u.includes('index'));
    const dyn = arr.find((u) => u.includes('dyn') && u.includes(`-${platform}-`)) || arr.find((u) => u.includes('dyn'));
    return { index, dyn };
  };
  page.on('response', onResp);
  try {
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

// Fetch both columnar files in the (Cloudflare-cleared) browser context and decode
// them into a Map<eaId, {price, isExtinct, status}>. price 0 = no price / extinct.
async function buildPriceMap(page, urls) {
  const raw = await page.evaluate(async (u) => {
    const [idx, dyn] = await Promise.all([
      fetch(u.index).then((r) => r.json()),
      fetch(u.dyn).then((r) => r.json()),
    ]);
    return { id0: idx.id0, d: idx.d || [], p: dyn.p || [], s: dyn.s || [] };
  }, urls);

  if (!Number.isFinite(raw.id0) || raw.d.length + 1 !== raw.p.length) {
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

module.exports = { discoverPriceUrls, buildPriceMap };
