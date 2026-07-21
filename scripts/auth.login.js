// auth.login.js
// One-time interactive login. Opens a real (headful) browser with a PERSISTENT
// profile so your fut.gg / EasySBC sessions (cookies + Supabase token) survive for
// later headless fetches. Nothing is committed: the profile lives in .auth/ (gitignored),
// and no tokens are ever written into code.
//
// Usage:  node auth.login.js
'use strict';

const path = require('path');
const readline = require('readline');
const puppeteer = require('puppeteer-extra');
const StealthPlugin = require('puppeteer-extra-plugin-stealth');
const { resolveChromePath } = require('./browser');
puppeteer.use(StealthPlugin());

const PROFILE_DIR = path.resolve(__dirname, '..', '.auth', 'browser-profile');

(async () => {
  console.log('🔐 Opening a browser with a persistent profile at:', PROFILE_DIR);
  const browser = await puppeteer.launch({
    headless: false,
    executablePath: resolveChromePath(puppeteer),
    userDataDir: PROFILE_DIR,
    defaultViewport: null,
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
  });

  const page = (await browser.pages())[0] || (await browser.newPage());
  await page.goto('https://www.fut.gg/evo-lab/my-players/', { waitUntil: 'domcontentloaded' }).catch(() => {});
  const page2 = await browser.newPage();
  await page2.goto('https://www.easysbc.io/', { waitUntil: 'domcontentloaded' }).catch(() => {});

  console.log('\n👉 Log in to BOTH sites in the opened tabs:');
  console.log('   • fut.gg      (Evo Lab)');
  console.log('   • easysbc.io  (for evo esMeta)');
  console.log('   When logged in to both, come back here and press ENTER to save the session.\n');

  await new Promise((resolve) => {
    const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
    rl.question('Press ENTER once you are logged in... ', () => { rl.close(); resolve(); });
  });

  await browser.close();
  console.log('✅ Session saved to .auth/browser-profile — you can now run fetch.evolab.js headless.');
})();
