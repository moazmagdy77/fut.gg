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
Steps: `club.1.get.ids.py` → `invalidate_upgraded_players.py` → `shared.fetch.data.js` (club) → `club.2b.fetch.prices.js` → `club.3.clean.py` → produces `data/club_final.json` (and auto-commits/pushes).

## 🧠 Model retraining
```bash
cd scripts
python retrain.py
```
Steps: `model.1.get_ids.js` → `shared.fetch.data.js` (model) → builds datasets (`model.4...`, `model.6...`, `build_all_players_summary.py`, in parallel) → trains models (`model.5...`, `model.7...`).

## 📊 Viewer
```bash
streamlit run scripts/visualize.py
```

## 🔧 Setup
- Node deps: `npm install`
- Python deps: `pip install -r requirements.txt`
- Optional: set `FUTBIN_SCRAPER_DIR` if the sibling futbin-scraper repo isn't at `../futbin-scraper`.
