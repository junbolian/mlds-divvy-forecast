# MLDS Divvy Forecast — Baseline & Debug Pipeline

A reproducible **end-to-end baseline** to sanity-check your local ML stack and produce ready-to-share artifacts. It implements a simple **Bronze → Silver → Reports** flow using synthetic data and a tiny LightGBM model for **classification (including multiclass)** and **regression**, with metrics and plots written to disk.

* **Zero external data required** to run.
* **Deterministic** via `--seed`.
* **Windows-friendly** commands (PowerShell, CMD, Git Bash).
* **Git PR workflow** documented.

---

## Contents

* [Prerequisites](#prerequisites)
* [Environment Setup](#environment-setup)
* [Repository Layout](#repository-layout)
* [CLI Usage](#cli-usage)

  * [Multiclass Classification (3 classes)](#multiclass-classification-3-classes)
  * [Regression](#regression)
  * [Outputs & Visualizations](#outputs--visualizations)
  * [Open figures on Windows / Git Bash](#open-figures-on-windows--git-bash)
* [Reproduce Everything (one-liners)](#reproduce-everything-one-liners)
* [Export Environment Files](#export-environment-files)
* [Git Branch & PR Workflow](#git-branch--pr-workflow)
* [Troubleshooting](#troubleshooting)
* [License](#license)

---

## Prerequisites

* **Python** 3.11 (recommended; tested on 3.11.14)
* **Conda** (Anaconda/Miniconda/Mamba)
* **Git**

> If you prefer `pip` only, it works too—see *Environment Setup*.

---

## Environment Setup

### Option A — create fresh Conda env (recommended)

**PowerShell / CMD**

```bat
conda create -n citybikes-311 python=3.11 -y
conda activate citybikes-311
pip install -U pip setuptools wheel packaging
pip install numpy pandas scikit-learn pyarrow lightgbm matplotlib
```

**Git Bash**

```bash
conda create -n citybikes-311 python=3.11 -y
conda activate citybikes-311
python -m pip install -U pip setuptools wheel packaging
python -m pip install numpy pandas scikit-learn pyarrow lightgbm matplotlib
```

### Option B — from repo files (if present)

```bat
conda env create -n citybikes-311 -f environment.yml
conda activate citybikes-311
# or:
pip install -r requirements.txt
```

---

## Repository Layout

```
mlds-divvy-forecast/
├─ scripts/
│  └─ debug_check.py         # v2 CLI: reg/clf, metrics, plots, JSON report
├─ data/
│  ├─ bronze/                # CSV (generated)
│  └─ silver/                # Parquet (generated)
├─ reports/                  # metrics, figures, preds, model dump (generated)
├─ .gitignore
├─ environment.yml           # exported from conda (optional)
└─ requirements.txt          # exported from pip (optional)
```

---

## CLI Usage

Script help:

```text
usage: debug_check.py [-h] [--bronze-dir BRONZE_DIR] [--silver-dir SILVER_DIR] [--reports-dir REPORTS_DIR]
                      [--mode {signal,noise}] [--task {reg,clf}] [--classes CLASSES] [--rows ROWS] [--skip-fit]
                      [--bronze BRONZE] [--silver SILVER] [--reports REPORTS] [--horizon HORIZON]

options:
  --bronze-dir BRONZE_DIR   Directory for CSV outputs (bronze).
  --silver-dir SILVER_DIR   Directory for Parquet outputs (silver).
  --reports-dir REPORTS_DIR Directory for reports and model artifacts.
  --mode {signal,noise}     Data generating process: learnable signal or pure noise.
  --task {reg,clf}          Task type: regression or classification.
  --classes CLASSES         Number of classes for classification (>=2).
  --rows ROWS               Row count (default depends on task/mode).
  --skip-fit                Skip model fitting (generate data only).
  --bronze/--silver/--reports  Shorthand aliases for the corresponding dirs.
  --horizon HORIZON         Accepted but unused (compat).
```

### Multiclass Classification (3 classes)

**PowerShell / CMD**

```bat
cd E:\mlds\mlds-divvy-forecast
conda activate citybikes-311
python scripts\debug_check.py ^
  --task clf --classes 3 --mode signal --rows 1500 --seed 42 --train-size 0.7 ^
  --plots cm,roc,pr,fi ^
  --bronze-dir data\bronze --silver-dir data\silver --reports-dir reports
```

**Git Bash**

```bash
cd /e/mlds/mlds-divvy-forecast
conda activate citybikes-311
python scripts/debug_check.py \
  --task clf --classes 3 --mode signal --rows 1500 --seed 42 --train-size 0.7 \
  --plots cm,roc,pr,fi \
  --bronze-dir data/bronze --silver-dir data/silver --reports-dir reports
```

### Regression

**PowerShell / CMD**

```bat
python scripts\debug_check.py ^
  --task reg --mode signal --rows 1200 --seed 42 --train-size 0.7 ^
  --plots fi,resid,pred,qq,hist ^
  --bronze-dir data\bronze --silver-dir data\silver --reports-dir reports
```

**Git Bash**

```bash
python scripts/debug_check.py \
  --task reg --mode signal --rows 1200 --seed 42 --train-size 0.7 \
  --plots fi,resid,pred,qq,hist \
  --bronze-dir data/bronze --silver-dir data/silver --reports-dir reports
```

### Outputs & Visualizations

The script writes:

* **Data**

  * `data/bronze/dummy_bronze.csv`
  * `data/silver/dummy_silver.parquet`
* **Reports (JSON + text + CSV)**

  * `reports/debug_report.json`  ← run summary (versions, paths, metrics)
  * `reports/classification_report.txt` (clf)
  * `reports/confusion_matrix.csv` (clf)
  * `reports/preds.csv` (clf) and `reports/preds_reg.csv` (reg)
  * `reports/model_lgbm.txt`  (LightGBM model dump)
* **Figures**

  * Classification: `confusion_matrix.png`, `roc.png`, `pr.png`, `fi.png`
  * Regression: `fi.png`, `pred_vs_true.png`, `residuals.png`, `residual_hist.png`, `qq.png`

The JSON report includes key metrics, for example:

```json
{
  "lgbm_fit": {
    "task": "clf",
    "backend": "lightgbm",
    "accuracy": 0.8667,
    "f1_macro": 0.8676,
    "logloss": 0.7378,
    "confusion_matrix_csv": "reports\\confusion_matrix.csv",
    "confusion_matrix_png": "reports\\confusion_matrix.png",
    "preds_csv": "reports\\preds.csv",
    "classification_report_txt": "reports\\classification_report.txt",
    "model_path": "reports\\model_lgbm.txt"
  },
  "status": "OK"
}
```

### Open figures on Windows / Git Bash

* **PowerShell / CMD**

  ```bat
  start reports\confusion_matrix.png
  ```
* **Git Bash**

  ```bash
  cmd.exe /c start reports/confusion_matrix.png
  ```

---

## Reproduce Everything (one-liners)

**PowerShell / CMD**

```bat
conda activate citybikes-311 && ^
python scripts\debug_check.py --task clf --classes 3 --mode signal --rows 1500 --seed 42 --train-size 0.7 --plots cm,roc,pr,fi --bronze-dir data\bronze --silver-dir data\silver --reports-dir reports && ^
python scripts\debug_check.py --task reg --mode signal --rows 1200 --seed 42 --train-size 0.7 --plots fi,resid,pred,qq,hist --bronze-dir data\bronze --silver-dir data\silver --reports-dir reports && ^
type reports\debug_report.json
```

**Git Bash**

```bash
conda activate citybikes-311 && \
python scripts/debug_check.py --task clf --classes 3 --mode signal --rows 1500 --seed 42 --train-size 0.7 --plots cm,roc,pr,fi --bronze-dir data/bronze --silver-dir data/silver --reports-dir reports && \
python scripts/debug_check.py --task reg --mode signal --rows 1200 --seed 42 --train-size 0.7 --plots fi,resid,pred,qq,hist --bronze-dir data/bronze --silver-dir data/silver --reports-dir reports && \
cat reports/debug_report.json
```

---

## Export Environment Files

```bat
conda env export -n citybikes-311 --from-history > environment.yml
python -m pip freeze > requirements.txt
```

---

## Git Branch & PR Workflow

**PowerShell / CMD**

```bat
git fetch origin
git switch main
git pull
git switch -c feat/full-publish-debug-check

REM run the two commands (clf & reg) from the sections above to generate artifacts

git add -A
git add -f data reports  # force-add generated artifacts if .gitignore excludes them
git commit -m "feat: publish code, reports, figures, and env files"
git push -u origin feat/full-publish-debug-check
```

**Git Bash**

```bash
git fetch origin && (git switch main || git checkout main) && git pull
(git switch -c feat/full-publish-debug-check || git switch feat/full-publish-debug-check)

# run the two debug_check commands to generate artifacts

git add -A && git add -f data reports
git commit -m "feat: publish code, reports, figures, and env files"
git push -u origin feat/full-publish-debug-check
```

Then open the suggested link from `git push` or visit:

```
https://github.com/<YOUR_USERNAME>/mlds-divvy-forecast/pull/new/feat/full-publish-debug-check
```

---

## Troubleshooting

* **`UserWarning: X does not have valid feature names`**
  Benign in this synthetic demo; scikit-learn warns when predicting with a raw `ndarray`. We intentionally create minimal features.

* **LightGBM `[Warning] No further splits with positive gain, best gain: -inf`**
  Expected if you run `--mode noise` or the toy dataset is too easy/hard. For sanity use `--mode signal` and increase `--rows`.

* **`Matplotlib` not found**
  `pip install matplotlib`

* **Paths and shells**

  * PowerShell/CMD use backslashes (`\`) and `start`.
  * Git Bash uses forward slashes (`/`) and `cmd.exe /c start`.

* **`fatal: not a git repository`**
  Run `git init`, then `git remote add origin https://github.com/<you>/mlds-divvy-forecast.git`.

* **Line endings warning (CRLF/LF)**
  Safe to ignore for images and text outputs. Add a `.gitattributes` if needed.

---

