// fetch.evolab.js
// Fetches the logged-in fut.gg Evo Lab ("my players") and writes data/evolab.json,
// replacing the manual "copy the XHR response" step. Reuses the persistent session
// created by auth.login.js. No credentials are stored in code — a fresh Supabase
// access token is read from the live browser session at runtime.
//
// Usage:  node fetch.evolab.js            (headless)
//         HEADLESS=false node fetch.evolab.js   (visible, for debugging)
'use strict';

const fs = require('fs').promises;
const path = require('path');
const puppeteer = require('puppeteer-extra');
const StealthPlugin = require('puppeteer-extra-plugin-stealth');
const { resolveChromePath } = require('./browser');
puppeteer.use(StealthPlugin());

const PROFILE_DIR = path.resolve(__dirname, '..', '.auth', 'browser-profile');
const OUT_FILE = path.resolve(__dirname, '..', 'data', 'evolab.json');
const EVOLAB_API = 'https://www.fut.gg/api/fut/evo-lab/';
const PAGE_URL = 'https://www.fut.gg/evo-lab/my-players/';
const HEADLESS = process.env.HEADLESS === 'false' ? false : true;

async function saveAtomic(file, data) {
  const tmp = `${file}.tmp`;
  await fs.writeFile(tmp, JSON.stringify(data, null, 2));
  await fs.rename(tmp, file);
}

(async () => {
  console.log('🧬 Fetching fut.gg Evo Lab via your saved session...');
  const browser = await puppeteer.launch({
    headless: HEADLESS,
    executablePath: resolveChromePath(puppeteer),
    userDataDir: PROFILE_DIR,
    defaultViewport: null,
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
  });

  try {
    const page = await browser.newPage();

    // 1) Capture the SPA's own authenticated call if it fires during navigation.
    let captured = null;
    page.on('response', async (res) => {
      try {
        if (res.url().startsWith(EVOLAB_API) && res.status() === 200) {
          captured = await res.json();
        }
      } catch (_) { /* ignore non-JSON / races */ }
    });

    await page.goto(PAGE_URL, { waitUntil: 'networkidle2', timeout: 60000 });
    await new Promise((r) => setTimeout(r, 1500));

    // 2) Fallback: call the API from the page context using the fresh Supabase
    //    access token the app keeps in localStorage (auto-refreshed on load).
    if (!captured || !captured.data) {
      captured = await page.evaluate(async (apiUrl) => {
        const key = Object.keys(localStorage).find((k) => /sb-.*-auth-token/.test(k));
        let token = null;
        if (key) {
          try { token = JSON.parse(localStorage.getItem(key)).access_token; } catch (_) {}
        }
        const res = await fetch(apiUrl, {
          headers: { accept: 'application/json', ...(token ? { authorization: `Bearer ${token}` } : {}) },
          credentials: 'include',
        });
        if (!res.ok) return { __error: res.status };
        return res.json();
      }, EVOLAB_API);
    }

    if (!captured || captured.__error || !captured.data) {
      const why = captured && captured.__error ? `HTTP ${captured.__error}` : 'no data / not logged in';
      console.error(`❌ Could not fetch Evo Lab (${why}).`);
      console.error('   Run  `node auth.login.js`  and log in to fut.gg first, then retry.');
      process.exitCode = 1;
      return;
    }

    await saveAtomic(OUT_FILE, captured);
    const n = Array.isArray(captured.data) ? captured.data.length : '?';
    console.log(`✅ Saved ${n} evo entries to ${path.relative(process.cwd(), OUT_FILE)}`);
  } finally {
    await browser.close();
  }
})();
