// Required modules
const fs = require('fs').promises; // Using promises version for async file operations
const path = require('path');
const axios = require('axios');
const puppeteer = require('puppeteer');

// --- Configuration ---
const CLUB_IDS_FILE = '../data/club_ids.json'; // Relative to script location
const MAPS_FILE = '../data/maps.json'; // Relative to script location for mappings

// Output Directories (relative to script location)
const GG_DATA_DIR = '../data/raw/ggData'; // For fut.gg player item definitions
const ES_META_DIR = '../data/raw/esMeta'; // For EasySBC meta ratings
const GG_META_DIR = '../data/raw/ggMeta'; // For fut.gg metarank

// API Endpoints
const FUTGG_PLAYER_DETAILS_URL_TEMPLATE = (eaId) => `https://www.fut.gg/api/fut/player-item-definitions/25/${eaId}/`;
const FUTGG_METARANK_URL_TEMPLATE = (eaId) => `https://www.fut.gg/api/fut/metarank/player/${eaId}/`;
const EASYSBC_META_URL_TEMPLATE = (eaId, esRoleId) => `https://api.easysbc.io/players/${eaId}?player-role-id=${esRoleId}&expanded=false`;

// Request Management
const MAX_CONCURRENT_PLAYERS_IN_BATCH = 100;
const DELAY_BETWEEN_BATCHES_MS = 2000;
const DELAY_BETWEEN_ARCHETYPE_CALLS_MS = 250;
const MAX_RETRIES_API = 2;
const API_TIMEOUT_MS = 25000;
const PUPPETEER_PAGE_TIMEOUT_MS = 30000;
const BROWSER_RESTART_INTERVAL_BATCHES = 1;

// Logging Configuration
const VERBOSE_LOGGING = false;

// --- Helper Functions ---

const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

async function ensureDirExists(dirPath) {
    try {
        await fs.mkdir(dirPath, { recursive: true });
    } catch (error) {
        if (error.code !== 'EEXIST') {
            console.error(`❌ Error creating directory ${dirPath}: ${error.message}\n${error.stack || ''}`);
            throw error;
        }
    }
}

async function saveData(filePath, data) {
    try {
        await fs.writeFile(filePath, JSON.stringify(data, null, 2), 'utf-8');
    } catch (error) {
        console.error(`❌ Error saving data to ${filePath}: ${error.message}\n${error.stack || ''}`);
        throw error;
    }
}

async function checkPlayerDataExists(eaIdStr, ggDataDirAbs, esMetaDirAbs, ggMetaDirAbs) {
    const ggDataPath = path.join(ggDataDirAbs, `${eaIdStr}_ggData.json`);
    const esMetaPath = path.join(esMetaDirAbs, `${eaIdStr}_esMeta.json`);
    const ggMetaPath = path.join(ggMetaDirAbs, `${eaIdStr}_ggMeta.json`);
    try {
        await Promise.all([
            fs.access(ggDataPath),
            fs.access(esMetaPath),
            fs.access(ggMetaPath)
        ]);
        return true;
    } catch (error) {
        return false;
    }
}

async function fetchFutGgWithPuppeteer(browser, url, identifier, dataType) {
    for (let attempt = 0; attempt <= MAX_RETRIES_API; attempt++) {
        let page;
        try {
            page = await browser.newPage();
            await page.setRequestInterception(true);
            page.on('request', (req) => {
                if (['image', 'stylesheet', 'font', 'media'].includes(req.resourceType())) {
                    req.abort();
                } else {
                    req.continue();
                }
            });
            await page.setUserAgent('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36');
            page.setDefaultNavigationTimeout(PUPPETEER_PAGE_TIMEOUT_MS);
            if (VERBOSE_LOGGING || attempt > 0) {
                console.log(`📦 [Fut.gg ${dataType}] Attempt ${attempt + 1}/${MAX_RETRIES_API + 1} for ID ${identifier}`);
            }
            await page.goto(url, { waitUntil: 'domcontentloaded' });
            const content = await page.evaluate(() => document.body.innerText);
            await page.close();
            if (!content) throw new Error("No content retrieved from page.");
            return JSON.parse(content);
        } catch (err) {
            if (page) await page.close().catch(e => console.warn(`⚠️ Error closing page for ID ${identifier}: ${e.message}`));
            if (attempt < MAX_RETRIES_API) {
                console.warn(`⚠️ [Fut.gg ${dataType}] Error for ID ${identifier} (Attempt ${attempt + 1}). Retrying... Error: ${err.message}`);
                await delay(1000 * (attempt + 1));
            } else {
                console.error(`❌ [Fut.gg ${dataType}] Failed for ID ${identifier} after ${MAX_RETRIES_API + 1} attempts. URL: ${url}. Error: ${err.message}`);
                return null;
            }
        }
    }
    return null;
}

async function fetchEasySBCWithAxios(eaId, esRoleId) {
    const url = EASYSBC_META_URL_TEMPLATE(eaId, esRoleId);
    for (let attempt = 0; attempt <= MAX_RETRIES_API; attempt++) {
        try {
            if (VERBOSE_LOGGING || attempt > 0) {
                 console.log(`📦 [EasySBC] Attempt ${attempt + 1}/${MAX_RETRIES_API + 1} for ID ${eaId}, Role ID: ${esRoleId}`);
            }
            const response = await axios.get(url, { timeout: API_TIMEOUT_MS });
            if (typeof response.data === 'object' && response.data !== null && !Array.isArray(response.data)) {
                return response.data;
            } else {
                console.warn(`⚠️ [EasySBC] Unexpected data format for ID ${eaId}, Role ID: ${esRoleId}. Expected object, got: ${typeof response.data}.`);
                return null;
            }
        } catch (err) {
            if (attempt < MAX_RETRIES_API) {
                console.warn(`⚠️ [EasySBC] Error for ID ${eaId}, Role ID: ${esRoleId} (Attempt ${attempt + 1}). Retrying... Error: ${err.message}`);
                await delay(1000 * (attempt + 1));
            } else {
                console.error(`❌ [EasySBC] Failed for ID ${eaId}, Role ID: ${esRoleId} after ${MAX_RETRIES_API + 1} attempts. Error: ${err.message}`);
                return null;
            }
        }
    }
    return null;
}

// --- Main Processing Function ---
(async () => {
    console.log("🚀 Starting API data fetching script...");
    const overallStartTime = Date.now();

    const scriptDir = __dirname;
    const clubIdsFilePath = path.resolve(scriptDir, CLUB_IDS_FILE);
    const mapsFilePath = path.resolve(scriptDir, MAPS_FILE);
    const ggDataDirAbs = path.resolve(scriptDir, GG_DATA_DIR);
    const esMetaDirAbs = path.resolve(scriptDir, ES_META_DIR);
    const ggMetaDirAbs = path.resolve(scriptDir, GG_META_DIR);

    let playerEaIds, positionIdToEsRoleIds;

    try {
        const rawIds = await fs.readFile(clubIdsFilePath, 'utf-8');
        playerEaIds = JSON.parse(rawIds);
        if (!Array.isArray(playerEaIds)) throw new Error("Club IDs file is not an array.");
        console.log(`ℹ️ Loaded ${playerEaIds.length} player EA IDs.`);
    } catch (error) {
        console.error(`❌ Fatal error reading ${CLUB_IDS_FILE}: ${error.message}\n${error.stack || ''}. Exiting.`);
        return;
    }

    // --- MODIFIED: Load mappings from maps.json ---
    try {
        const mapsRaw = await fs.readFile(mapsFilePath, 'utf-8');
        const mapsData = JSON.parse(mapsRaw);
        if (!mapsData.positionIdToEsRoleIds) {
            throw new Error("'positionIdToEsRoleIds' key not found in maps.json.");
        }
        positionIdToEsRoleIds = mapsData.positionIdToEsRoleIds;
        console.log("ℹ️ Successfully loaded 'positionIdToEsRoleIds' mapping from maps.json.");
    } catch (error) {
        console.error(`❌ Fatal error reading or parsing ${MAPS_FILE}: ${error.message}\n${error.stack || ''}. Exiting.`);
        return;
    }

    try {
        await Promise.all([
            ensureDirExists(ggDataDirAbs),
            ensureDirExists(esMetaDirAbs),
            ensureDirExists(ggMetaDirAbs)
        ]);
    } catch (error) {
        console.error(`❌ Fatal error creating output directories. Exiting.`);
        return;
    }

    let browser = await puppeteer.launch({ headless: true, args: ['--no-sandbox', '--disable-setuid-sandbox'] });
    let batchesProcessedSinceRestart = 0;
    let successfulPlayerProcessing = 0;
    let failedPlayerProcessing = 0;
    let skippedExistingCount = 0;
    let fetchedNewCount = 0;

    for (let i = 0; i < playerEaIds.length; i += MAX_CONCURRENT_PLAYERS_IN_BATCH) {
        const batchEaIds = playerEaIds.slice(i, i + MAX_CONCURRENT_PLAYERS_IN_BATCH);
        const batchNumber = Math.floor(i / MAX_CONCURRENT_PLAYERS_IN_BATCH) + 1;
        console.log(`\n🔄 Processing Batch ${batchNumber}/${Math.ceil(playerEaIds.length / MAX_CONCURRENT_PLAYERS_IN_BATCH)} (Players ${i + 1} to ${Math.min(i + MAX_CONCURRENT_PLAYERS_IN_BATCH, playerEaIds.length)})`);
        const batchStartTime = Date.now();

        if (batchesProcessedSinceRestart >= BROWSER_RESTART_INTERVAL_BATCHES && BROWSER_RESTART_INTERVAL_BATCHES > 0) {
            console.log("🔁 Restarting Puppeteer browser to free resources...");
            await browser.close().catch(e => console.warn("⚠️ Error closing browser during restart:", e.message));
            browser = await puppeteer.launch({ headless: true, args: ['--no-sandbox', '--disable-setuid-sandbox'] });
            batchesProcessedSinceRestart = 0;
            console.log("✅ Browser restarted.");
        }

        const playerPromises = batchEaIds.map(async (eaId) => {
            const eaIdStr = String(eaId);

            const allFilesExist = await checkPlayerDataExists(eaIdStr, ggDataDirAbs, esMetaDirAbs, ggMetaDirAbs);
            if (allFilesExist) {
                if (VERBOSE_LOGGING) console.log(`⏭️ [Cache] Data for player ID ${eaIdStr} already exists. Skipping fetch.`);
                return { eaId: eaIdStr, status: 'skipped_exists', success: true };
            }

            if (VERBOSE_LOGGING) console.log(`--- Fetching new data for EA ID: ${eaIdStr} ---`);
            let futGgDetailsData = null;
            let playerSucceeded = true;

            const [detailsResult, metarankResult] = await Promise.allSettled([
                fetchFutGgWithPuppeteer(browser, FUTGG_PLAYER_DETAILS_URL_TEMPLATE(eaId), eaId, "details"),
                fetchFutGgWithPuppeteer(browser, FUTGG_METARANK_URL_TEMPLATE(eaId), eaId, "metarank")
            ]);

            if (detailsResult.status === 'fulfilled' && detailsResult.value) {
                futGgDetailsData = detailsResult.value;
                try {
                    await saveData(path.join(ggDataDirAbs, `${eaIdStr}_ggData.json`), futGgDetailsData);
                } catch (saveError) { playerSucceeded = false; }
            } else {
                console.error(`❌ [Fut.gg details] Fetch failed or empty for ${eaIdStr}: ${detailsResult.reason?.message || 'No data returned'}`);
                playerSucceeded = false;
            }

            if (metarankResult.status === 'fulfilled' && metarankResult.value) {
                try {
                    await saveData(path.join(ggMetaDirAbs, `${eaIdStr}_ggMeta.json`), metarankResult.value);
                } catch (saveError) { playerSucceeded = false; }
            } else {
                console.error(`❌ [Fut.gg metarank] Fetch failed or empty for ${eaIdStr}: ${metarankResult.reason?.message || 'No data returned'}`);
                playerSucceeded = false;
            }

            if (playerSucceeded && futGgDetailsData && futGgDetailsData.data) {
                const playerDefinition = futGgDetailsData.data;
                const esRoleIdsToFetch = new Set();

                const allPositions = [playerDefinition.position, ...(playerDefinition.alternativePositionIds || [])];
                allPositions.forEach(posId => {
                    if (posId !== undefined && positionIdToEsRoleIds[String(posId)]) {
                        positionIdToEsRoleIds[String(posId)].forEach(roleId => esRoleIdsToFetch.add(roleId));
                    }
                });

                if (esRoleIdsToFetch.size > 0) {
                    const allEsApiResponses = [];
                    let esSbcFetchOverallSuccess = true;
                    for (const esRoleId of esRoleIdsToFetch) {
                        const esResponseObject = await fetchEasySBCWithAxios(eaId, esRoleId);
                        if (esResponseObject) {
                            allEsApiResponses.push({ roleId: esRoleId, data: esResponseObject });
                        } else {
                            esSbcFetchOverallSuccess = false;
                        }
                        if (esRoleIdsToFetch.size > 1) await delay(DELAY_BETWEEN_ARCHETYPE_CALLS_MS);
                    }

                    if (!esSbcFetchOverallSuccess) {
                        playerSucceeded = false;
                        console.warn(`⚠️ [EasySBC] One or more role ID fetches failed for ${eaIdStr}. Data might be incomplete.`);
                    }

                    if (allEsApiResponses.length > 0) {
                        try {
                            await saveData(path.join(esMetaDirAbs, `${eaIdStr}_esMeta.json`), allEsApiResponses);
                        } catch (saveError) { playerSucceeded = false; }
                    }
                }
            }
            return { eaId: eaIdStr, status: playerSucceeded ? 'fetched_success' : 'fetched_fail', success: playerSucceeded };
        });

        const batchResults = await Promise.allSettled(playerPromises);
        let currentBatchSuccess = 0;
        let currentBatchFail = 0;

        batchResults.forEach(result => {
            if (result.status === 'fulfilled') {
                if (result.value.success) {
                    successfulPlayerProcessing++;
                    currentBatchSuccess++;
                    if (result.value.status === 'skipped_exists') skippedExistingCount++;
                    else if (result.value.status === 'fetched_success') fetchedNewCount++;
                } else {
                    failedPlayerProcessing++;
                    currentBatchFail++;
                }
            } else {
                failedPlayerProcessing++;
                currentBatchFail++;
                console.error(`❌ Critical error processing a player promise (rejected): ${result.reason?.stack || result.reason}`);
            }
        });

        batchesProcessedSinceRestart++;
        const batchEndTime = Date.now();
        console.log(`⏱ Batch ${batchNumber} completed in ${(batchEndTime - batchStartTime) / 1000}s. Success: ${currentBatchSuccess}, Fail/Partial: ${currentBatchFail}.`);

        if (i + MAX_CONCURRENT_PLAYERS_IN_BATCH < playerEaIds.length) {
            await delay(DELAY_BETWEEN_BATCHES_MS);
        }
    }

    await browser.close().catch(e => console.warn("⚠️ Error closing main browser instance:", e.message));
    const overallEndTime = Date.now();
    console.log(`\n🎉 All player ID processing attempts completed in ${((overallEndTime - overallStartTime) / 1000 / 60).toFixed(2)} minutes.`);
    console.log(`📊 Total Player IDs in input: ${playerEaIds.length}`);
    console.log(`  Processed Successfully (fetched or already existing): ${successfulPlayerProcessing}`);
    console.log(`    - Skipped (data already existed): ${skippedExistingCount}`);
    console.log(`    - Newly Fetched & Saved: ${fetchedNewCount}`);
    console.log(`  Failed/Partially Failed during fetch: ${failedPlayerProcessing}`);
    console.log("Script finished.");
})();