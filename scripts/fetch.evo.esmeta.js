// fetch.evo.esmeta.js
// Fetches EXACT evo esMeta from EasySBC and writes data/raw/evo_esmeta.json.
//
// Uses the per-evo detail endpoint, which returns the SAME metaRatings shape as the base
// esMeta API (per role: chemistry 0..3, chemstyleId, metaRating, isBestChemstyleAtChem):
//   1. GET /evolved-players?sort-meta-rating-3-chem   -> list of evos (resourceId + manualEvolvedPlayerId)
//   2. GET /players/{resourceId}?v2&type=manual-evolved-player&manualEvolvedPlayerId={mid}  -> metaRatings
// So club.3.clean can parse evo esMeta identically to non-evo (chem 0 -> sub, best chem 3 ->
// on-chem + esChemStyle). Keyed by resourceId (== fut.gg evolab eaId). ~1 + N requests.
//
// Auth (plain Bearer API, no Cloudflare): EASYSBC_TOKEN env -> .auth/easysbc.token -> profile session.
// Usage:  EASYSBC_TOKEN=... node fetch.evo.esmeta.js
'use strict';

const fs = require('fs').promises;
const { existsSync } = require('fs');
const path = require('path');
const axios = require('axios');
const puppeteer = require('puppeteer-extra');
const StealthPlugin = require('puppeteer-extra-plugin-stealth');
const { resolveChromePath } = require('./browser');
puppeteer.use(StealthPlugin());

const ROOT = path.resolve(__dirname, '..');
const PROFILE_DIR = path.join(ROOT, '.auth', 'browser-profile');
const TOKEN_FILE = path.join(ROOT, '.auth', 'easysbc.token');
const OUT_FILE = path.join(ROOT, 'data', 'raw', 'evo_esmeta.json');

const API = 'https://api-fc26.easysbc.io';
const HEADLESS = process.env.HEADLESS === 'false' ? false : true;
const DELAY_MS = 150;
const HEADERS_BASE = { Accept: 'application/json', Platform: 'Playstation', Origin: 'https://www.easysbc.io', Referer: 'https://www.easysbc.io/' };

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function getToken() {
  if (process.env.EASYSBC_TOKEN) return process.env.EASYSBC_TOKEN.trim();
  if (existsSync(TOKEN_FILE)) {
    const t = (await fs.readFile(TOKEN_FILE, 'utf-8')).trim();
    if (t) return t;
  }
  // Read the token from the logged-in EasySBC session in the persistent profile.
  const browser = await puppeteer.launch({
    headless: HEADLESS, executablePath: resolveChromePath(puppeteer),
    userDataDir: PROFILE_DIR, defaultViewport: null,
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
  });
  try {
    const page = await browser.newPage();

    // Primary: capture the Bearer token from the app's own authenticated API calls.
    // Robust regardless of where EasySBC stores it (localStorage / cookie / IndexedDB).
    let captured = null;
    page.on('request', (req) => {
      if (captured) return;
      try {
        const h = req.headers();
        const auth = h['authorization'] || h['Authorization'];
        if (auth && /bearer\s+eyJ/i.test(auth) && req.url().includes('easysbc.io')) {
          captured = auth.replace(/bearer\s+/i, '').trim();
        }
      } catch (_) { /* ignore */ }
    });

    for (const url of ['https://www.easysbc.io/players', 'https://www.easysbc.io/']) {
      try { await page.goto(url, { waitUntil: 'networkidle2', timeout: 45000 }); } catch (_) {}
      for (let i = 0; i < 25 && !captured; i++) await sleep(300); // wait for late XHRs
      if (captured) break;
    }
    if (captured) return captured;

    // Fallback: deep-scan localStorage + sessionStorage (recursing nested JSON) + cookies.
    return await page.evaluate(() => {
      const rx = /eyJ[\w-]+\.[\w-]+\.[\w-]+/;
      const find = (val, depth) => {
        if (val == null || depth > 5) return null;
        if (typeof val === 'string') { const m = val.match(rx); return m ? m[0] : null; }
        if (typeof val === 'object') { for (const k in val) { const r = find(val[k], depth + 1); if (r) return r; } }
        return null;
      };
      for (const store of [window.localStorage, window.sessionStorage]) {
        if (!store) continue;
        for (let i = 0; i < store.length; i++) {
          const raw = store.getItem(store.key(i));
          const direct = raw && raw.match(rx);
          if (direct) return direct[0];
          try { const r = find(JSON.parse(raw), 0); if (r) return r; } catch (_) {}
        }
      }
      const m = document.cookie.match(rx);
      return m ? m[0] : null;
    });
  } finally {
    await browser.close();
  }
}

async function apiGet(url, token) {
  const { data } = await axios.get(url, { headers: { ...HEADERS_BASE, Authorization: `Bearer ${token}` }, timeout: 20000 });
  return data;
}

(async () => {
  console.log('🧪 Fetching evo esMeta (per-evo detail, with chem styles) from EasySBC...');
  const token = await getToken();
  if (!token) {
    console.error('❌ No EasySBC token. Set EASYSBC_TOKEN, or run auth.login.js and log into easysbc.io.');
    process.exit(1);
  }

  const list = await apiGet(`${API}/evolved-players?sort-meta-rating-3-chem`, token);
  if (!Array.isArray(list)) { console.error('❌ Unexpected evolved-players response.'); process.exit(1); }
  console.log(`Evos to fetch: ${list.length}`);

  const out = {};
  let i = 0, ok = 0;
  for (const p of list) {
    const rid = p.resourceId, mid = p.manualEvolvedPlayerId;
    i += 1;
    if (rid == null || !mid) continue;
    try {
      const d = await apiGet(`${API}/players/${rid}?v2&type=manual-evolved-player&manualEvolvedPlayerId=${encodeURIComponent(mid)}`, token);
      out[String(rid)] = {
        name: p.name, assetId: p.assetId, bestPlayerRoleId: d.bestPlayerRoleId,
        metaRatings: Array.isArray(d.metaRatings) ? d.metaRatings : [],
      };
      ok += 1;
    } catch (e) {
      console.warn(`  ${p.name} (${rid}): ${e.response ? 'HTTP ' + e.response.status : e.message}`);
    }
    if (i % 10 === 0) console.log(`  ...${i}/${list.length}`);
    await sleep(DELAY_MS);
  }

  await fs.mkdir(path.dirname(OUT_FILE), { recursive: true });
  const tmp = OUT_FILE + '.tmp';
  await fs.writeFile(tmp, JSON.stringify(out, null, 2));
  await fs.rename(tmp, OUT_FILE);
  const rows = Object.values(out).reduce((s, e) => s + e.metaRatings.length, 0);
  console.log(`✅ ${ok}/${list.length} evos, ${rows} metaRating rows → ${path.relative(process.cwd(), OUT_FILE)}`);
})().catch((e) => { console.error('❌', e.message); process.exit(1); });
