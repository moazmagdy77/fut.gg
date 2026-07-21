# ⚽ FUT.GG Player Analysis Pipeline

A multi-step data pipeline to scrape, clean, enrich, and model FIFA player data using Puppeteer (Node) and Python ML, with a Streamlit viewer.

## 🗂 Layout
- `data/`: All input/output files (JSON/CSV). Scraped raw files live under `data/raw/` (gitignored).
- `models/`: Trained model bundles (`*.pkl`, gitignored).
- `scripts/`: All pipeline scripts (flat). Shared game logic lives in `scripts/shared_utils.py`.

## 🚀 Club update (weekly / on-demand)
Export `club-analyzer.html` into `data/`, then run the orchestrator:
```bash
cd scripts
python pipeline.py
```
Steps: `club.1.get.ids.py` → `invalidate_upgraded_players.py` → (optional) `fetch.evolab.js` + `fetch.evo.esmeta.js` → `shared.fetch.data.js` (club) → `club.2b.fetch.prices.js` → `club.3.clean.py` → produces `data/club_final.json` (and auto-commits/pushes).

## 🧠 Model retraining
```bash
cd scripts
python retrain.py
```
Steps: `model.1.get_ids.js` → `shared.fetch.data.js` (model) → `model.6...` dataset + `build_all_players_summary.py` (parallel) → trains `model.7...` (ggMetaSub). esMeta and ggMeta are fetched exact (base API + `ggRatingStr` + EasySBC evo), so the only model is the anchored `ggMetaSub`.

## 📊 Viewer
```bash
streamlit run scripts/visualize.py
```

## 🔧 Setup
- Node deps: `npm install` (uses system Chrome if puppeteer's bundled Chrome is absent — see `scripts/browser.js`)
- Python deps: `pip install -r requirements.txt`
- Authed fetches (evolab + evo esMeta): run `node auth.login.js` once and log into fut.gg + easysbc.io (session stored in gitignored `.auth/`).
- Optional: `FUTBIN_SCRAPER_DIR` (sibling futbin-scraper location), `PUPPETEER_EXECUTABLE_PATH` (force a browser), `EASYSBC_TOKEN` (evo esMeta token).
