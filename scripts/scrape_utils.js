// scrape_utils.js
// Small helpers shared by the puppeteer scrapers so they aren't copy-pasted across
// files: timing, directory creation, atomic JSON writes, cross-version isolated
// browser contexts, bounded concurrency, and Cloudflare-block detection.
'use strict';

const fs = require('fs').promises;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const randInt = (min, max) => Math.floor(Math.random() * (max - min + 1)) + min;

async function ensureDir(dir) {
  await fs.mkdir(dir, { recursive: true }).catch((e) => { if (e.code !== 'EEXIST') throw e; });
}

async function fileExists(filePath) {
  try { await fs.access(filePath); return true; } catch { return false; }
}

// tmp + rename so a crash mid-write can't leave a half-written (invalid) JSON file
// that would break a downstream JSON.parse.
async function writeJsonAtomic(filePath, data) {
  const tmp = `${filePath}.tmp`;
  await fs.writeFile(tmp, JSON.stringify(data, null, 2));
  await fs.rename(tmp, filePath);
}

// Isolated browser context (own cookies / cf_clearance), tolerant of puppeteer
// version differences in the context API.
async function createIsolatedContext(browser) {
  if (typeof browser.createBrowserContext === 'function') return browser.createBrowserContext();
  if (typeof browser.createIncognitoBrowserContext === 'function') return browser.createIncognitoBrowserContext();
  return browser.defaultBrowserContext();
}

// Bounds simultaneous work (e.g. Cloudflare priming navigations) to avoid a burst.
function makeSemaphore(max) {
  let active = 0;
  const q = [];
  return {
    acquire: () => new Promise((res) => { const go = () => { if (active < max) { active++; res(); } else q.push(go); }; go(); }),
    release: () => { active--; if (q.length) q.shift()(); },
  };
}

// True if an in-page fetch result looks like a Cloudflare block / non-JSON body.
function blocked(obj) {
  return !obj || obj.__err || obj.__status === 403 || obj.__status === 429 || obj.__status === 503 || obj.__status === 'nonjson';
}

module.exports = { sleep, randInt, ensureDir, fileExists, writeJsonAtomic, createIsolatedContext, makeSemaphore, blocked };
