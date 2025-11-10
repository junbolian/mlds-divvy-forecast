# MLDS Divvy 15/30-Minute Forecast

**Goal.** Predict Divvy station status at H ∈ {15, 30} minutes — `EMPTY / FULL / OK` (multiclass) — and occupancy (0–1) using a single model family (**LightGBM**).  
**Why this repo.** Minimal, reproducible, and course-grade ready: data → features → training with probability calibration → evaluation (PR & calibration) → Streamlit map.

## Quick start
```bash
# 1) Create env and install deps
pip install -r requirements.txt

# 2) Seed synthetic data (no internet required)
python -m scripts.seed_demo_data

# 3) Build features & labels
python -m features.make_labels --bronze data/bronze --silver data/silver --lag 5

# 4) Train and evaluate
python -m models.train_lightgbm --silver data/silver --models models_store --plots outputs/plots
python -m models.evaluate --silver data/silver --models models_store --plots outputs/plots

# 5) Map app (optional, requires internet)
streamlit run app_streamlit/app.py
