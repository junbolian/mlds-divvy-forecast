#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score, mean_absolute_error,
    accuracy_score, f1_score, log_loss, classification_report,
    confusion_matrix, roc_curve, precision_recall_curve, auc
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.datasets import make_classification
from scipy import stats

# optional backends
try:
    import lightgbm as lgb
    _HAS_LGB = True
except Exception:
    _HAS_LGB = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

API_VERSION = "debug_check API v2 (2025-11-10)"

# ------------ utils ------------
def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{ts}] {msg}")

def ensure_dirs(*paths: str) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)

def write_json(obj: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def save_fig(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

def normp(p: str) -> str:
    try:
        return os.path.normpath(str(p))
    except Exception:
        return str(p)

def try_importances(model):
    if hasattr(model, "feature_name_"):
        names = list(model.feature_name_)
    elif hasattr(model, "feature_names_in_"):
        names = list(model.feature_names_in_)
    else:
        names = None
    if hasattr(model, "feature_importances_"):
        imps = np.asarray(model.feature_importances_, dtype=float)
        if names is None:
            names = [f"f{i}" for i in range(len(imps))]
        return names, imps
    return None

def parse_plots(arg: str | None) -> set[str]:
    if not arg:
        return set()
    return {p.strip().lower() for p in arg.split(",") if p.strip()}

# ------------ data gen ------------
def gen_regression(rows: int, mode: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    f1 = rng.normal(0, 1, rows)
    f2 = rng.normal(0, 1, rows)
    f3 = rng.uniform(-2, 2, rows)
    if mode == "signal":
        y = 3.0 * f1 - 2.0 * f2 + 3.0 * np.sin(f3) + rng.normal(0, 0.7, rows)
    else:
        y = rng.normal(0, 1.5, rows)
    return pd.DataFrame({"f1": f1, "f2": f2, "f3": f3, "y": y})

def gen_classification(rows: int, classes: int, mode: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if mode == "signal":
        X, y = make_classification(
            n_samples=rows, n_features=3, n_informative=3, n_redundant=0,
            n_classes=classes, n_clusters_per_class=1, flip_y=0.02, random_state=seed
        )
        df = pd.DataFrame(X, columns=["f1", "f2", "f3"])
        df["label"] = y
    else:
        f1 = rng.normal(0, 1, rows)
        f2 = rng.normal(0, 1, rows)
        f3 = rng.normal(0, 1, rows)
        y = rng.integers(0, classes, size=rows)
        df = pd.DataFrame({"f1": f1, "f2": f2, "f3": f3, "label": y})
    return df

def bronze_to_silver(bronze_csv: str, silver_parquet: str) -> dict:
    df = pd.read_csv(bronze_csv)
    Path(silver_parquet).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(silver_parquet, index=False)
    return {"rows": int(df.shape[0]), "parquet": normp(silver_parquet)}

# ------------ plots (reg) ------------
def plot_feature_importance(names, imps, out_png, top: int = 30):
    order = np.argsort(-imps)[:top]
    plt.figure(figsize=(6, 4))
    plt.barh(np.array(names)[order][::-1], imps[order][::-1])
    plt.xlabel("Importance"); plt.ylabel("Feature"); plt.title("Feature Importance")
    save_fig(out_png)

def plot_pred_vs_true(y_true, y_pred, out_png):
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, s=10, alpha=0.6)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims)
    plt.xlabel("True"); plt.ylabel("Pred"); plt.title("Predicted vs. True")
    save_fig(out_png)

def plot_residuals(y_true, y_pred, out_png):
    resid = y_true - y_pred
    plt.figure(figsize=(6, 4))
    plt.scatter(y_pred, resid, s=10, alpha=0.6)
    plt.axhline(0.0)
    plt.xlabel("Pred"); plt.ylabel("Residual"); plt.title("Residuals vs. Fitted")
    save_fig(out_png)

def plot_residual_hist(y_true, y_pred, out_png):
    resid = y_true - y_pred
    plt.figure(figsize=(6, 4))
    plt.hist(resid, bins=30, alpha=0.9)
    plt.xlabel("Residual"); plt.ylabel("Count"); plt.title("Residual Histogram")
    save_fig(out_png)

def plot_qq(y_true, y_pred, out_png):
    resid = y_true - y_pred
    plt.figure(figsize=(5, 5))
    stats.probplot(resid, dist="norm", plot=plt)
    plt.title("Residual QQ Plot")
    save_fig(out_png)

# ------------ plots (clf) ------------
def plot_confusion(cm: np.ndarray, classes: list[int], out_png: str):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes); plt.yticks(tick_marks, classes)
    plt.xlabel("Predicted"); plt.ylabel("True")
    save_fig(out_png)

def plot_roc_multiclass(y_true: np.ndarray, proba: np.ndarray, out_png: str):
    n_classes = proba.shape[1]
    y_bin = np.eye(n_classes)[y_true]
    plt.figure(figsize=(6, 5))
    for k in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, k], proba[:, k])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {k} AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC (OvR)")
    plt.legend(fontsize=8)
    save_fig(out_png)

def plot_pr_multiclass(y_true: np.ndarray, proba: np.ndarray, out_png: str):
    n_classes = proba.shape[1]
    y_bin = np.eye(n_classes)[y_true]
    plt.figure(figsize=(6, 5))
    from sklearn.metrics import precision_recall_curve
    for k in range(n_classes):
        prec, rec, _ = precision_recall_curve(y_bin[:, k], proba[:, k])
        pr_auc = auc(rec, prec)
        plt.plot(rec, prec, label=f"Class {k} AUC={pr_auc:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR Curve (OvR)")
    plt.legend(fontsize=8)
    save_fig(out_png)

# ------------ fitters ------------
def fit_regression(df: pd.DataFrame, reports: str, plots: set[str], seed: int, train_size: float) -> dict:
    X = df[["f1", "f2", "f3"]]
    y = df["y"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, train_size=train_size, random_state=seed)

    backend = "lightgbm" if _HAS_LGB else "sklearn_rf"
    if _HAS_LGB:
        model = lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=-1,
            subsample=0.9, colsample_bytree=0.9, random_state=seed
        )
    else:
        model = RandomForestRegressor(n_estimators=400, random_state=seed, n_jobs=-1)

    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    r2 = float(r2_score(y_te, y_pred))
    mae = float(mean_absolute_error(y_te, y_pred))

    preds_csv = str(Path(reports, "preds_reg.csv"))
    pd.DataFrame({"y_true": y_te.to_numpy(), "y_pred": y_pred}).to_csv(preds_csv, index=False)

    plots_out = {}
    if _HAS_MPL:
        if "fi" in plots:
            imp = try_importances(model)
            if imp is not None:
                names, imps = imp
                fi_png = str(Path(reports, "fi.png"))
                plot_feature_importance(names, imps, fi_png)
                log(f"Saved feature importance -> {normp(fi_png)}")
                plots_out["fi"] = normp(fi_png)
        if "pred" in plots:
            pv_png = str(Path(reports, "pred_vs_true.png"))
            plot_pred_vs_true(y_te.to_numpy(), y_pred, pv_png)
            log(f"Saved pred vs. true -> {normp(pv_png)}")
            plots_out["pred_vs_true"] = normp(pv_png)
        if "resid" in plots:
            rvf_png = str(Path(reports, "residuals.png"))
            plot_residuals(y_te.to_numpy(), y_pred, rvf_png)
            log(f"Saved residuals -> {normp(rvf_png)}")
            plots_out["residuals"] = normp(rvf_png)
        if "hist" in plots:
            rh_png = str(Path(reports, "residual_hist.png"))
            plot_residual_hist(y_te.to_numpy(), y_pred, rh_png)
            log(f"Saved residual histogram -> {normp(rh_png)}")
            plots_out["residual_hist"] = normp(rh_png)
        if "qq" in plots:
            qq_png = str(Path(reports, "qq.png"))
            plot_qq(y_te.to_numpy(), y_pred, qq_png)
            log(f"Saved QQ plot -> {normp(qq_png)}")
            plots_out["qq"] = normp(qq_png)

    model_path = str(Path(reports, "model_lgbm.txt" if backend == "lightgbm" else "model_rf.txt"))
    try:
        if backend == "lightgbm":
            model.booster_.save_model(model_path)
        else:
            write_json({"backend": backend, "params": model.get_params()}, model_path + ".json")
    except Exception:
        pass

    return {
        "skipped": False,
        "task": "reg",
        "backend": backend,
        "r2": r2,
        "mae": mae,
        "preds_csv": normp(preds_csv),
        "plots": {k: normp(v) for k, v in plots_out.items()},
        "model_path": normp(model_path),
    }

def fit_classification(df: pd.DataFrame, reports: str, classes: int, plots: set[str], seed: int, train_size: float) -> dict:
    X = df[["f1", "f2", "f3"]]
    y = df["label"].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, train_size=train_size, stratify=y, random_state=seed)

    backend = "lightgbm" if _HAS_LGB else "sklearn_rf"
    if _HAS_LGB:
        model = lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
            objective="multiclass", num_class=classes, random_state=seed
        )
    else:
        model = RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=-1)

    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    acc = float(accuracy_score(y_te, y_pred))
    f1m = float(f1_score(y_te, y_pred, average="macro"))

    proba = None
    ll = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_te)
        if isinstance(proba, list):
            proba = np.stack([p[:, 1] for p in proba], axis=1)
        try:
            ll = float(log_loss(y_te, proba))
        except Exception:
            ll = None

    preds_csv = str(Path(reports, "preds.csv"))
    pd.DataFrame({"y_true": y_te.to_numpy(), "y_pred": y_pred}).to_csv(preds_csv, index=False)

    cls_txt = str(Path(reports, "classification_report.txt"))
    with open(cls_txt, "w", encoding="utf-8") as f:
        f.write(classification_report(y_te, y_pred, digits=4))

    cm = confusion_matrix(y_te, y_pred, labels=list(range(classes)))
    cm_csv = str(Path(reports, "confusion_matrix.csv"))
    pd.DataFrame(cm, index=range(classes), columns=range(classes)).to_csv(cm_csv)

    plots_out = {}
    if _HAS_MPL:
        if "cm" in plots or not plots:
            cm_png = str(Path(reports, "confusion_matrix.png"))
            plot_confusion(cm, list(range(classes)), cm_png)
            log(f"Saved confusion matrix plot -> {normp(cm_png)}")
            plots_out["confusion_matrix"] = normp(cm_png)
        if proba is not None and "roc" in plots:
            roc_png = str(Path(reports, "roc.png"))
            plot_roc_multiclass(y_te.to_numpy(), proba, roc_png)
            log(f"Saved ROC plot -> {normp(roc_png)}")
            plots_out["roc"] = normp(roc_png)
        if proba is not None and "pr" in plots:
            pr_png = str(Path(reports, "pr.png"))
            plot_pr_multiclass(y_te.to_numpy(), proba, pr_png)
            log(f"Saved PR plot -> {normp(pr_png)}")
            plots_out["pr"] = normp(pr_png)
        if "fi" in plots:
            imp = try_importances(model)
            if imp is not None:
                names, imps = imp
                fi_png = str(Path(reports, "fi.png"))
                plot_feature_importance(names, imps, fi_png)
                log(f"Saved feature importance -> {normp(fi_png)}")
                plots_out["fi"] = normp(fi_png)

    model_path = str(Path(reports, "model_lgbm.txt" if backend == "lightgbm" else "model_rf.txt"))
    try:
        if backend == "lightgbm":
            model.booster_.save_model(model_path)
        else:
            write_json({"backend": backend, "params": model.get_params()}, model_path + ".json")
    except Exception:
        pass

    return {
        "skipped": False,
        "task": "clf",
        "backend": backend,
        "accuracy": acc,
        "f1_macro": f1m,
        "logloss": ll,
        "confusion_matrix_csv": normp(cm_csv),
        "preds_csv": normp(preds_csv),
        "classification_report_txt": normp(cls_txt),
        "plots": {k: normp(v) for k, v in plots_out.items()},
        "model_path": normp(model_path),
    }

# ------------ main ------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bronze-dir", default="data/bronze", help="Directory for CSV outputs (bronze).")
    parser.add_argument("--silver-dir", default="data/silver", help="Directory for Parquet outputs (silver).")
    parser.add_argument("--reports-dir", default="reports", help="Directory for reports and model artifacts.")
    parser.add_argument("--mode", choices=["signal", "noise"], default="signal", help="Data generating process.")
    parser.add_argument("--task", choices=["reg", "clf"], default="reg", help="Task type.")
    parser.add_argument("--classes", type=int, default=3, help="Number of classes for classification (>=2).")
    parser.add_argument("--rows", type=int, default=None, help="Row count. Defaults depend on task/mode.")
    parser.add_argument("--skip-fit", action="store_true", help="Skip tiny fit.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--train-size", type=float, default=0.7, help="Train size fraction.")
    parser.add_argument("--plots", type=str, default="", help="Comma list: fi,resid,pred,qq,hist,cm,roc,pr.")
    # aliases / compat
    parser.add_argument("--bronze", dest="bronze_alias", default=None, help="Alias of --bronze-dir.")
    parser.add_argument("--silver", dest="silver_alias", default=None, help="Alias of --silver-dir.")
    parser.add_argument("--reports", dest="reports_alias", default=None, help="Alias of --reports-dir.")
    parser.add_argument("--horizon", type=int, default=None, help="Accepted but unused (compat).")
    args = parser.parse_args()

    bronze_dir = args.bronze_alias or args.bronze_dir
    silver_dir = args.silver_alias or args.silver_dir
    reports_dir = args.reports_alias or args.reports_dir
    ensure_dirs(bronze_dir, silver_dir, reports_dir)

    log("Checking Python environment...")
    py_info = {
        "api_version": API_VERSION,
        "python_version": sys.version,
        "executable": sys.executable,
        "cwd": str(Path.cwd()),
    }

    log("Importing packages and collecting versions...")
    versions = {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": __import__('sklearn').__version__,
        "pyarrow": __import__('pyarrow').__version__,
        "lightgbm": (lgb.__version__ if _HAS_LGB else "unavailable"),
    }

    # default rows
    if args.rows is None:
        if args.task == "reg":
            args.rows = 1200 if args.mode == "signal" else 1000
        else:
            args.rows = 1500 if args.mode == "signal" else 1200

    # bronze
    log("Creating dummy bronze CSV...")
    if args.task == "reg":
        df = gen_regression(args.rows, args.mode, args.seed)
    else:
        df = gen_classification(args.rows, max(2, args.classes), args.mode, args.seed)
    bronze_csv = str(Path(bronze_dir, "dummy_bronze.csv"))
    df.to_csv(bronze_csv, index=False)
    log(f"bronze_csv: {normp(bronze_csv)}")

    # silver
    log("Transforming bronze -> silver (Parquet)...")
    silver_path = str(Path(silver_dir, "dummy_silver.parquet"))
    silver_stats = bronze_to_silver(bronze_csv, silver_path)
    log(f"silver_stats: {silver_stats}")

    report = {
        "python": py_info,
        "versions": versions,
        "bronze_csv": normp(bronze_csv),
        "silver": silver_stats,
    }

    if args.skip_fit:
        report["lgbm_fit"] = {"skipped": True, "reason": "skip-fit"}
    else:
        plots = parse_plots(args.plots)
        log("Optional tiny model fit...")
        if args.task == "reg":
            fit_stats = fit_regression(df, reports_dir, plots, seed=args.seed, train_size=args.train_size)
            log(f"Regression metrics -> r2: {fit_stats.get('r2'):.4f}, mae: {fit_stats.get('mae'):.4f}")
        else:
            fit_stats = fit_classification(df, reports_dir, max(2, args.classes), plots, seed=args.seed, train_size=args.train_size)
            if "confusion_matrix" in fit_stats.get("plots", {}):
                log(f"Classification metrics -> accuracy: {fit_stats.get('accuracy'):.4f}, "
                    f"f1_macro: {fit_stats.get('f1_macro'):.4f}, "
                    f"logloss: {fit_stats.get('logloss') if fit_stats.get('logloss') is not None else float('nan'):.4f}")
        report["lgbm_fit"] = fit_stats

    out_json = str(Path(reports_dir, "debug_report.json"))
    write_json(report, out_json)
    log(f"Wrote report: {normp(out_json)}")

if __name__ == "__main__":
    main()
