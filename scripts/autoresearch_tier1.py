"""B1 — Autoresearch Tier 1: grid over simple-pool embeddings × classifiers.

Small, deterministic grid. Runs in minutes on CPU. Produces results.csv +
comparison heatmap. Logistic Regression / SVM / Random Forest / XGBoost
under StratifiedGroupKFold(groups=patient_id), class-balanced where
applicable, with optional PCA(100) pre-step.

Usage:
    python scripts/autoresearch_tier1.py

Outputs:
    results/autoresearch/tier1/
        results.csv
        best_config.yaml
        auroc_heatmap.png
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

OUTDIR = Path("results/autoresearch/tier1")
EMB_ROOT = Path("results/embeddings")
CLINICAL = Path("results/data/clinical_table.csv")

CLASSIFIER_GRID = {
    "lr": lambda: LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
    "svm_rbf": lambda: SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42),
    "rf": lambda: RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                          n_jobs=-1, random_state=42),
}
if HAS_XGB:
    # 19% MSI-H → scale_pos_weight ≈ 81/19 ≈ 4.26
    CLASSIFIER_GRID["xgb"] = lambda: xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        scale_pos_weight=4.26, n_jobs=-1, random_state=42,
        eval_metric="auc", use_label_encoder=False, tree_method="hist",
    )

FEAT_ENG = ["raw", "pca100"]


def load_embedding(emb_dir: Path) -> tuple[np.ndarray, pd.DataFrame]:
    X = np.load(emb_dir / "embeddings.npy")
    meta = pd.read_csv(emb_dir / "metadata.csv")
    cl = pd.read_csv(CLINICAL)[["PATIENT", "isMSIH"]]
    cl["y"] = (cl["isMSIH"] == "MSI-H").astype(int)
    df = meta.rename(columns={"patient_id": "PATIENT"}).merge(cl[["PATIENT", "y"]], on="PATIENT")
    if len(df) != len(X):
        # metadata/embeddings aligned; labels may drop some
        df_full = meta.rename(columns={"patient_id": "PATIENT"})
        df_full["_row"] = np.arange(len(df_full))
        merged = df_full.merge(cl[["PATIENT", "y"]], on="PATIENT", how="inner")
        X = X[merged["_row"].values]
        df = merged.drop(columns=["_row"]).reset_index(drop=True)
    return X, df


def build_pipeline(feat_eng: str, classifier_key: str) -> Pipeline:
    steps = [("scaler", StandardScaler())]
    if feat_eng == "pca100":
        steps.append(("pca", PCA(n_components=100, random_state=42)))
    steps.append(("clf", CLASSIFIER_GRID[classifier_key]()))
    return Pipeline(steps)


def cv_eval(pipe: Pipeline, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict:
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    aurocs, auprcs = [], []
    for tr, te in cv.split(X, y, groups=groups):
        pipe.fit(X[tr], y[tr])
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(X[te])[:, 1]
        else:
            proba = pipe.decision_function(X[te])
        aurocs.append(roc_auc_score(y[te], proba))
        auprcs.append(average_precision_score(y[te], proba))
    return {
        "auroc_mean": float(np.mean(aurocs)),
        "auroc_std": float(np.std(aurocs)),
        "auprc_mean": float(np.mean(auprcs)),
        "auprc_std": float(np.std(auprcs)),
    }


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    emb_dirs = [d for d in sorted(EMB_ROOT.iterdir()) if (d / "embeddings.npy").exists()]

    rows = []
    for emb_dir in emb_dirs:
        X, df = load_embedding(emb_dir)
        y = df["y"].values
        groups = df["PATIENT"].values
        print(f"--- {emb_dir.name}  ({X.shape[0]} slides, {X.shape[1]}D, "
              f"{groups.__class__.__name__} with {len(set(groups))} patients) ---")
        for clf_key, feat_eng in itertools.product(CLASSIFIER_GRID.keys(), FEAT_ENG):
            pipe = build_pipeline(feat_eng, clf_key)
            try:
                m = cv_eval(pipe, X, y, groups)
            except Exception as e:
                print(f"  {clf_key:8s} {feat_eng:8s}  FAILED: {e}")
                continue
            print(f"  {clf_key:8s} {feat_eng:8s}  "
                  f"AUROC={m['auroc_mean']:.3f} ± {m['auroc_std']:.3f}   "
                  f"AUPRC={m['auprc_mean']:.3f} ± {m['auprc_std']:.3f}")
            rows.append({
                "embedding": emb_dir.name,
                "classifier": clf_key,
                "feat_eng": feat_eng,
                **m,
            })

    results = pd.DataFrame(rows)
    results.to_csv(OUTDIR / "results.csv", index=False)

    if len(results):
        best = results.sort_values("auroc_mean", ascending=False).iloc[0]
        (OUTDIR / "best_config.yaml").write_text(yaml.safe_dump(best.to_dict(), sort_keys=False))
        print(f"\nBEST: {best['embedding']} / {best['classifier']} / {best['feat_eng']}  "
              f"AUROC = {best['auroc_mean']:.3f} ± {best['auroc_std']:.3f}")

        # Heatmap: embedding × (classifier_feat_eng)
        results["config"] = results["classifier"] + "_" + results["feat_eng"]
        pivot = results.pivot(index="embedding", columns="config", values="auroc_mean")
        fig, ax = plt.subplots(figsize=(1 * len(pivot.columns) + 3, 0.5 * len(pivot.index) + 2), dpi=150)
        im = ax.imshow(pivot.values, vmin=0.4, vmax=0.8, cmap="RdBu_r", aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=35, ha="right", fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=9)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            color="white" if (v < 0.55 or v > 0.72) else "black", fontsize=8)
        ax.set_title("B1 — Tier 1 AUROC (embedding × classifier/feat-eng)")
        fig.colorbar(im, ax=ax, shrink=0.8, label="AUROC")
        fig.tight_layout()
        fig.savefig(OUTDIR / "auroc_heatmap.png", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {OUTDIR/'results.csv'}")
        print(f"Saved: {OUTDIR/'auroc_heatmap.png'}")
        print(f"Saved: {OUTDIR/'best_config.yaml'}")


if __name__ == "__main__":
    main()
