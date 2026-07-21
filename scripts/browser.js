// browser.js
// Cross-platform Chrome/Chromium resolver shared by every puppeteer script, so the repo
// runs on macOS / Windows / Linux without hardcoded paths.
//
// Resolution order:
//   1. PUPPETEER_EXECUTABLE_PATH env var (explicit override)
//   2. puppeteer's own downloaded Chrome (if `npx puppeteer browsers install chrome` was run)
//   3. a system-installed browser for the current OS
// Returns undefined if none is found (puppeteer then falls back to its default and prints
// its own install guidance).
'use strict';

const { existsSync } = require('fs');
const path = require('path');

function candidatesForPlatform() {
  const plt = process.platform;

  if (plt === 'darwin') {
    return [
      '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
      '/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary',
      '/Applications/Chromium.app/Contents/MacOS/Chromium',
      '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge',
    ];
  }

  if (plt === 'win32') {
    const pf = process.env['PROGRAMFILES'] || 'C:\\Program Files';
    const pf86 = process.env['PROGRAMFILES(X86)'] || 'C:\\Program Files (x86)';
    const local = process.env['LOCALAPPDATA'];
    return [
      path.join(pf, 'Google', 'Chrome', 'Application', 'chrome.exe'),
      path.join(pf86, 'Google', 'Chrome', 'Application', 'chrome.exe'),
      local && path.join(local, 'Google', 'Chrome', 'Application', 'chrome.exe'),
      path.join(pf, 'Google', 'Chrome SxS', 'Application', 'chrome.exe'), // Canary
      path.join(pf86, 'Microsoft', 'Edge', 'Application', 'msedge.exe'),
    ].filter(Boolean);
  }

  // linux / other
  return [
    '/usr/bin/google-chrome',
    '/usr/bin/google-chrome-stable',
    '/usr/bin/chromium',
    '/usr/bin/chromium-browser',
    '/snap/bin/chromium',
    '/usr/bin/microsoft-edge',
  ];
}

// `puppeteerInstance` (optional): pass your puppeteer-extra/puppeteer instance so we can
// prefer its bundled Chrome when present.
function resolveChromePath(puppeteerInstance) {
  const envPath = process.env.PUPPETEER_EXECUTABLE_PATH;
  if (envPath && existsSync(envPath)) return envPath;

  try {
    if (puppeteerInstance && typeof puppeteerInstance.executablePath === 'function') {
      const bundled = puppeteerInstance.executablePath();
      if (bundled && existsSync(bundled)) return bundled;
    }
  } catch (_) { /* bundled Chrome not installed */ }

  return candidatesForPlatform().find((p) => p && existsSync(p));
}

module.exports = { resolveChromePath };
