# ⚽ FUT.GG Player Analysis Pipeline

A multi-step data pipeline to scrape, clean, enrich, and model FIFA player data using Puppeteer and Python ML.

## 🗂 Folder Structure
- `data/`: All input/output files (JSON/CSV)
- `scripts/retrain_model/`: Weekly run scripts and model training
- `scripts/club/`: On-demand scripts for club-based player sets
- `scripts/evo/`: Evolution-related analysis scripts
- `scripts/visualize/`: Final data preparation and Streamlit viewer

## 🚀 To Run Weekly Pipeline
```bash
cd scripts/retrain_model
python 0.retrain_model.py