// Required modules
const fs = require('fs').promises; // Using promises version for async file operations
const path = require('path');
const axios = require('axios');
const puppeteer = require('puppeteer');

// --- Configuration ---
const CLUB_IDS_FILE = '../data/club_ids.json'; // Relative to script location

// Output Directories (relative to script location)
const GG_DATA_DIR = '../data/raw/ggData'; // For fut.gg player item definitions
const ES_META_DIR = '../data/raw/esMeta'; // For EasySBC meta ratings
const GG_META_DIR = '../data/raw/ggMeta'; // For fut.gg metarank

// API Endpoints
const FUTGG_PLAYER_DETAILS_URL_TEMPLATE = (eaId) => `https://www.fut.gg/api/fut/player-item-definitions/25/${eaId}/`;
const FUTGG_METARANK_URL_TEMPLATE = (eaId) => `https://www.fut.gg/api/fut/metarank/player/${eaId}/`;
const EASYSBC_META_URL_TEMPLATE = (eaId, archetypeId) => `https://api.easysbc.io/squad-builder/meta-ratings?archetypeId=${archetypeId}&resourceId=${eaId}`;

// Request Management
const MAX_CONCURRENT_PLAYERS_IN_BATCH = 10;
const DELAY_BETWEEN_BATCHES_MS = 2000; // Delay after a batch is processed
const DELAY_BETWEEN_ARCHETYPE_CALLS_MS = 250; // Polite delay for EasySBC calls for the same player
const MAX_RETRIES_API = 2;
const API_TIMEOUT_MS = 25000; // Timeout for individual API calls (Axios)
const PUPPETEER_PAGE_TIMEOUT_MS = 30000; // Timeout for Puppeteer page navigation/actions
const BROWSER_RESTART_INTERVAL_BATCHES = 20; // Restart browser every X batches

// Logging Configuration
const VERBOSE_LOGGING = false; // Set to true for more detailed success logs

// Mapping for EasySBC Archetypes
const positionIdToArchetype = {
    "0": ["goalkeeper", "sweeper_keeper"],
    "3": ["fullback", "falseback", "wingback", "attacking_wingback"], // RB
    "5": ["ball_playing_defender", "defender", "stopper"],           // CB
    "7": ["fullback", "falseback", "wingback", "attacking_wingback"], // LB
    "10": ["holding", "deep_lying_playmaker", "wide_half", "centre_half"], // CDM
    "12": ["winger", "wide_midfielder", "wide_playmaker", "inside_forward"], // RM
    "14": ["holding", "deep_lying_playmaker", "playmaker", "half_winger", "box_to_box"], // CM
    "16": ["winger", "wide_midfielder", "wide_playmaker", "inside_forward"], // LM
    "18": ["playmaker", "half_winger", "classic_ten", "shadow_striker"], // CAM
    "23": ["inside_forward", "winger", "wide_playmaker"], // RW
    "25": ["advanced_forward", "target_forward", "poacher", "false_nine"], // ST
    "27": ["inside_forward", "winger", "wide_playmaker"]  // LW
};

// --- Helper Functions ---

/**
 * Creates a delay for a specified number of milliseconds.
 * @param {number} ms - The delay in milliseconds.
 */
const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

/**
 * Ensures a directory exists, creating it if necessary.
 * @param {string} dirPath - The path to the directory.
 */
async function ensureDirExists(dirPath) {
    try {
        await fs.mkdir(dirPath, { recursive: true });
    } catch (error) {
        if (error.code !== 'EEXIST') { // Ignore error if directory already exists
            console.error(`❌ Error creating directory ${dirPath}: ${error.message}\n${error.stack || ''}`);
            throw error; // Re-throw if it's a different error
        }
    }
}

/**
 * Saves data to a JSON file.
 * @param {string} filePath - The full path to the file.
 * @param {any} data - The data to save.
 */
async function saveData(filePath, data) {
    try {
        await fs.writeFile(filePath, JSON.stringify(data, null, 2), 'utf-8');
    } catch (error) {
        console.error(`❌ Error saving data to ${filePath}: ${error.message}\n${error.stack || ''}`);
        throw error;
    }
}

/**
 * Checks if all data files for a given player ID exist.
 * @param {string} eaIdStr - The player's EA ID as a string.
 * @param {string} ggDataDirAbs - Absolute path to fut.gg player item definitions directory.
 * @param {string} esMetaDirAbs - Absolute path to EasySBC meta ratings directory.
 * @param {string} ggMetaDirAbs - Absolute path to fut.gg metarank directory.
 * @returns {Promise<boolean>} True if all files exist, false otherwise.
 */
async function checkPlayerDataExists(eaIdStr, ggDataDirAbs, esMetaDirAbs, ggMetaDirAbs) {
    const ggDataPath = path.join(ggDataDirAbs, `${eaIdStr}_ggData.json`);
    const esMetaPath = path.join(esMetaDirAbs, `${eaIdStr}_esMeta.json`);
    const ggMetaPath = path.join(ggMetaDirAbs, `${eaIdStr}_ggMeta.json`);

    try {
        await fs.access(ggDataPath);
        await fs.access(esMetaPath);
        await fs.access(ggMetaPath);
        return true; // All files are accessible
    } catch (error) {
        // If any fs.access throws an error (commonly ENOENT), it means a file doesn't exist
        return false;
    }
}

/**
 * Deletes all raw data files for a given player ID.
 * @param {string} eaIdStr - The player's EA ID as a string.
 * @param {string} ggDataDirAbs - Absolute path to fut.gg player item definitions directory.
 * @param {string} esMetaDirAbs - Absolute path to EasySBC meta ratings directory.
 * @param {string} ggMetaDirAbs - Absolute path to fut.gg metarank directory.
 */
async function deletePlayerData(eaIdStr, ggDataDirAbs, esMetaDirAbs, ggMetaDirAbs) {
    const filesToDelete = [
        path.join(ggDataDirAbs, `${eaIdStr}_ggData.json`),
        path.join(esMetaDirAbs, `${eaIdStr}_esMeta.json`),
        path.join(ggMetaDirAbs, `${eaIdStr}_ggMeta.json`)
    ];

    for (const filePath of filesToDelete) {
        try {
            await fs.unlink(filePath);
            if (VERBOSE_LOGGING) console.log(`🗑️ Deleted stale file: ${filePath}`);
        } catch (error) {
            if (error.code === 'ENOENT') { // File not found, already deleted or never existed for this specific type
                if (VERBOSE_LOGGING) console.log(`ℹ️ Stale file not found during deletion (may not have existed or already removed): ${filePath}`);
            } else {
                console.warn(`⚠️ Error deleting stale file ${filePath}: ${error.message}`);
            }
        }
    }
}

/**
 * Scans directories for existing player data files and removes data for players
 * not present in the current playerEaIds list.
 * @param {Array<string|number>} currentPlayerEaIds - Array of current player EA IDs.
 * @param {string} ggDataDirAbs - Absolute path to fut.gg player item definitions directory.
 * @param {string} esMetaDirAbs - Absolute path to EasySBC meta ratings directory.
 * @param {string} ggMetaDirAbs - Absolute path to fut.gg metarank directory.
 */
async function cleanupStalePlayerData(currentPlayerEaIds, ggDataDirAbs, esMetaDirAbs, ggMetaDirAbs) {
    console.log("\n🧹 Starting cleanup of stale player data...");
    const currentPlayerEaIdsSet = new Set(currentPlayerEaIds.map(id => String(id)));
    const existingPlayerIdsInDirs = new Set();

    const directoriesToScan = [
        { dir: ggDataDirAbs, suffix: '_ggData.json' },
        { dir: esMetaDirAbs, suffix: '_esMeta.json' },
        { dir: ggMetaDirAbs, suffix: '_ggMeta.json' }
    ];

    for (const { dir, suffix } of directoriesToScan) {
        try {
            const files = await fs.readdir(dir);
            for (const file of files) {
                if (file.endsWith(suffix)) {
                    const eaIdStr = file.substring(0, file.length - suffix.length);
                    if (eaIdStr && !isNaN(parseInt(eaIdStr, 10))) { // Basic check to ensure it's likely an ID
                        existingPlayerIdsInDirs.add(eaIdStr);
                    }
                }
            }
        } catch (error) {
            if (error.code === 'ENOENT') {
                if (VERBOSE_LOGGING) console.log(`ℹ️ Directory not found during cleanup scan (may be created later): ${dir}`);
            } else {
                console.warn(`⚠️ Error reading directory ${dir} for cleanup: ${error.message}`);
            }
        }
    }

    let staleCount = 0;
    for (const existingIdStr of existingPlayerIdsInDirs) {
        if (!currentPlayerEaIdsSet.has(existingIdStr)) {
            console.log(`🗑️ Player ID ${existingIdStr} is stale (not in club_ids.json). Deleting its data.`);
            await deletePlayerData(existingIdStr, ggDataDirAbs, esMetaDirAbs, ggMetaDirAbs);
            staleCount++;
        }
    }

    if (staleCount > 0) {
        console.log(`✅ Cleanup finished. Deleted data for ${staleCount} stale player(s).`);
    } else {
        console.log(`✅ No stale player data found to delete.`);
    }
}


/**
 * Fetches data from a fut.gg URL using Puppeteer with retry logic.
 * @param {import('puppeteer').Browser} browser - The Puppeteer browser instance.
 * @param {string} url - The URL to fetch.
 * @param {string|number} identifier - Player EA ID for logging.
 * @param {string} dataType - Description of data type for logging (e.g., "details", "metarank").
 * @returns {Promise<object|null>} Parsed JSON data or null on failure.
 */
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
                console.log(`📦 [Fut.gg ${dataType}] Attempt ${attempt + 1}/${MAX_RETRIES_API + 1} for ID ${identifier} from ${url}`);
            }
            
            const responsePromise = page.goto(url, { waitUntil: 'domcontentloaded' });
            const timeoutPromise = new Promise((_, reject) => 
                setTimeout(() => reject(new Error(`⏰ Puppeteer page operation timed out after ${PUPPETEER_PAGE_TIMEOUT_MS}ms for URL: ${url}`)), PUPPETEER_PAGE_TIMEOUT_MS)
            );
            
            await Promise.race([responsePromise, timeoutPromise]);

            const content = await page.evaluate(() => document.body.innerText);
            await page.close();
            
            if (!content) {
                throw new Error("No content retrieved from page.");
            }
            try {
                return JSON.parse(content);
            } catch (parseError) {
                console.error(`❌ [Fut.gg ${dataType}] JSON Parse Error for ID ${identifier}. URL: ${url}. Content: ${content.substring(0,200)}...\n${parseError.stack || parseError}`);
                throw parseError; 
            }

        } catch (err) {
            if (page) await page.close().catch(e => console.warn(`⚠️ Error closing page for ID ${identifier} (Fut.gg ${dataType}): ${e.message}`));
            
            if (attempt < MAX_RETRIES_API) {
                console.warn(`⚠️ [Fut.gg ${dataType}] Error for ID ${identifier} (Attempt ${attempt + 1}). Retrying... Error: ${err.message}\n${err.stack || ''}`);
                await delay(1000 * (attempt + 1));
            } else {
                console.error(`❌ [Fut.gg ${dataType}] Failed for ID ${identifier} after ${MAX_RETRIES_API + 1} attempts. URL: ${url}. Error: ${err.message}\n${err.stack || ''}`);
                return null; 
            }
        }
    }
    return null; 
}

/**
 * Fetches data from EasySBC API using Axios with retry logic.
 * @param {string|number} eaId - Player EA ID.
 * @param {string} archetypeId - The archetype ID.
 * @returns {Promise<Array<object>|null>} Parsed JSON data (array of ratings) or null on failure.
 */
async function fetchEasySBCWithAxios(eaId, archetypeId) {
    const url = EASYSBC_META_URL_TEMPLATE(eaId, archetypeId);
    for (let attempt = 0; attempt <= MAX_RETRIES_API; attempt++) {
        try {
            if (VERBOSE_LOGGING || attempt > 0) {
                 console.log(`📦 [EasySBC] Attempt ${attempt + 1}/${MAX_RETRIES_API + 1} for ID ${eaId}, Archetype: ${archetypeId}`);
            }
            const response = await axios.get(url, { timeout: API_TIMEOUT_MS });
            if (Array.isArray(response.data)) {
                return response.data;
            } else {
                console.warn(`⚠️ [EasySBC] Unexpected data format for ID ${eaId}, Archetype: ${archetypeId}. Expected array, got: ${typeof response.data}. URL: ${url}`);
                return null; 
            }
        } catch (err) {
            if (attempt < MAX_RETRIES_API) {
                console.warn(`⚠️ [EasySBC] Error for ID ${eaId}, Archetype: ${archetypeId} (Attempt ${attempt + 1}). Retrying... Error: ${err.message}\n${err.response ? `Status: ${err.response.status}, Data: ${JSON.stringify(err.response.data).substring(0,100)}...` : ''}\n${err.stack || ''}`);
                await delay(1000 * (attempt + 1));
            } else {
                console.error(`❌ [EasySBC] Failed for ID ${eaId}, Archetype: ${archetypeId} after ${MAX_RETRIES_API + 1} attempts. URL: ${url}. Error: ${err.message}\n${err.response ? `Status: ${err.response.status}, Data: ${JSON.stringify(err.response.data).substring(0,100)}...` : ''}\n${err.stack || ''}`);
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
    const ggDataDirAbs = path.resolve(scriptDir, GG_DATA_DIR);
    const esMetaDirAbs = path.resolve(scriptDir, ES_META_DIR);
    const ggMetaDirAbs = path.resolve(scriptDir, GG_META_DIR);

    let playerEaIds;
    try {
        const rawIds = await fs.readFile(clubIdsFilePath, 'utf-8');
        playerEaIds = JSON.parse(rawIds);
        if (!Array.isArray(playerEaIds)) throw new Error("Club IDs file is not an array.");
        console.log(`ℹ️ Loaded ${playerEaIds.length} player EA IDs.`);
    } catch (error) {
        console.error(`❌ Fatal error reading ${CLUB_IDS_FILE}: ${error.message}\n${error.stack || ''}. Exiting.`);
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

    await cleanupStalePlayerData(playerEaIds, ggDataDirAbs, esMetaDirAbs, ggMetaDirAbs);

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
            const eaIdStr = String(eaId); // Use string version for file operations and logging

            // Check if data already exists for this player
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
                    if (VERBOSE_LOGGING) console.log(`✅ [Fut.gg details] Saved for ${eaIdStr}`);
                } catch (saveError) { playerSucceeded = false; }
            } else {
                console.error(`❌ [Fut.gg details] Fetch failed or empty for ${eaIdStr}: ${detailsResult.reason?.message || 'No data returned'}`);
                playerSucceeded = false;
            }

            if (metarankResult.status === 'fulfilled' && metarankResult.value) {
                try {
                    await saveData(path.join(ggMetaDirAbs, `${eaIdStr}_ggMeta.json`), metarankResult.value);
                    if (VERBOSE_LOGGING) console.log(`✅ [Fut.gg metarank] Saved for ${eaIdStr}`);
                } catch (saveError) { playerSucceeded = false; }
            } else {
                console.error(`❌ [Fut.gg metarank] Fetch failed or empty for ${eaIdStr}: ${metarankResult.reason?.message || 'No data returned'}`);
                playerSucceeded = false;
            }

            if (playerSucceeded && futGgDetailsData && futGgDetailsData.data) {
                const playerDefinition = futGgDetailsData.data;
                const position = playerDefinition.position; 
                const alternativePositionIds = playerDefinition.alternativePositionIds || []; 
                const archetypesToFetch = new Set();

                if (position !== undefined && positionIdToArchetype[String(position)]) {
                    positionIdToArchetype[String(position)].forEach(arch => archetypesToFetch.add(arch));
                }
                alternativePositionIds.forEach(altPosId => {
                    if (positionIdToArchetype[String(altPosId)]) {
                        positionIdToArchetype[String(altPosId)].forEach(arch => archetypesToFetch.add(arch));
                    }
                });
                
                if (archetypesToFetch.size > 0) {
                    if (VERBOSE_LOGGING) console.log(`ℹ️ [EasySBC] Derived archetypes for ${eaIdStr}: ${[...archetypesToFetch].join(', ')}`);
                    const allArchetypeApiResponses = [];
                    let esSbcFetchOverallSuccess = true;
                    
                    for (const archetype of archetypesToFetch) {
                        const esResponseArray = await fetchEasySBCWithAxios(eaId, archetype); // Use original eaId for API
                        if (esResponseArray) { 
                            allArchetypeApiResponses.push({ archetype: archetype, ratings: esResponseArray });
                        } else {
                            esSbcFetchOverallSuccess = false;
                        }
                        if (archetypesToFetch.size > 1) await delay(DELAY_BETWEEN_ARCHETYPE_CALLS_MS);
                    }

                    if (!esSbcFetchOverallSuccess) {
                        playerSucceeded = false;
                        console.warn(`⚠️ [EasySBC] One or more archetype fetches failed for ${eaIdStr}. Data might be incomplete.`);
                    }

                    if (allArchetypeApiResponses.length > 0) {
                        try {
                            await saveData(path.join(esMetaDirAbs, `${eaIdStr}_esMeta.json`), allArchetypeApiResponses);
                            if (VERBOSE_LOGGING) console.log(`✅ [EasySBC] Saved meta for ${eaIdStr} (${allArchetypeApiResponses.length} archetypes)`);
                        } catch (saveError) { playerSucceeded = false; }
                    } else {
                        if (VERBOSE_LOGGING && archetypesToFetch.size > 0 && esSbcFetchOverallSuccess) {
                            console.log(`ℹ️ [EasySBC] No successful responses to save for ${eaIdStr} (all archetypes returned no data or failed gracefully).`);
                        }
                    }
                } else {
                    if (VERBOSE_LOGGING) console.log(`ℹ️ [EasySBC] No archetypes derived for ${eaIdStr} from positions.`);
                }
            } else {
                if (playerSucceeded && VERBOSE_LOGGING) {
                    console.log(`ℹ️ [EasySBC] Skipping for ${eaIdStr} due to missing or failed fut.gg details data.`);
                }
            }
            if (VERBOSE_LOGGING) console.log(`--- Finished processing for EA ID: ${eaIdStr} ---`);
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
                    if (result.value.status === 'skipped_exists') {
                        skippedExistingCount++;
                    } else if (result.value.status === 'fetched_success') {
                        fetchedNewCount++;
                    }
                } else {
                    failedPlayerProcessing++;
                    currentBatchFail++;
                }
            } else { // Promise was rejected (should be rare with try/catch in helpers)
                failedPlayerProcessing++;
                currentBatchFail++;
                console.error(`❌ Critical error processing a player promise (rejected): ${result.reason?.stack || result.reason}`);
            }
        });

        batchesProcessedSinceRestart++;
        const batchEndTime = Date.now();
        console.log(`⏱ Batch ${batchNumber} completed in ${(batchEndTime - batchStartTime) / 1000}s. Success: ${currentBatchSuccess}, Fail/Partial: ${currentBatchFail}.`);
        
        if (i + MAX_CONCURRENT_PLAYERS_IN_BATCH < playerEaIds.length) {
            if (VERBOSE_LOGGING) console.log(`⏳ Delaying ${DELAY_BETWEEN_BATCHES_MS / 1000}s before next batch...`);
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