"""Step 2d + 2e — multi-model late fusion and few-shot methods.

Evaluates whether combining our raw slide embeddings buys anything over the
best single-model result. Two evaluation regimes run side by side:

  * patient_cv      — StratifiedGroupKFold(5) over patient_id (within-cohort)
  * loso            — leave-one-site-out, with patient-level leakage guard
                      (any patient with ≥1 slide in the held-out site is
                      dropped from train)

Strategies:
  1. Single-model LR baseline (for reference)
  2. Late-average fusion — mean of per-model predicted probabilities over all
     2/3/4-model subsets of the base model list
  3. Concat + PCA(100) + LR on stacked features
  4. Stacking — per-fold inner CV base-model predictions → meta-LR
  5. k-NN few-shot (cosine, k ∈ {3,5,9})
  6. Prototypical few-shot

Uses raw embeddings only (Harmony was not a generic win — see
docs/c1_failure_diagnosis.md).

Usage:
    python scripts/fusion.py
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

OUTDIR = Path("results/analysis/fusion")
EMB_ROOT = Path("results/embeddings")
CLINICAL = Path("results/data/clinical_table.csv")
SLIDE_TABLE = Path("results/data/slide_table_pyramidal.csv")

BASE_MODELS = ["conch_v1.5_mean", "ctranspath_mean", "uni2_mean", "virchow2_mean"]
# Aggregator variants also included for reference but not in fusion subsets.
EXTRA_MODELS = ["conch_v1.5_titan", "virchow2_prism"]

SEED = 42


# -------------------------------------------------------------- loading --

def load_aligned(emb_names: list[str]) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Return a dict {name: X (n_slides × d)} and a shared metadata frame.

    All models are re-indexed onto the intersection of their slide_ids so the
    rows correspond across embeddings.
    """
    # Clinical labels
    cl = pd.read_csv(CLINICAL)[["PATIENT", "isMSIH"]]
    cl["y"] = (cl["isMSIH"] == "MSI-H").astype(int)
    st = pd.read_csv(SLIDE_TABLE)
    st["slide_id"] = st["FILENAME"].apply(lambda p: Path(p).stem)
    st = st[["slide_id", "PATIENT", "SITE"]]

    meta_frames = {}
    mats = {}
    for name in emb_names:
        emb_dir = EMB_ROOT / name
        meta = pd.read_csv(emb_dir / "metadata.csv")
        X = np.load(emb_dir / "embeddings.npy")
        if len(meta) != len(X):
            raise ValueError(f"{name}: metadata/emb mismatch")
        meta = meta.rename(columns={"patient_id": "PATIENT"})
        meta["_row"] = np.arange(len(meta))
        meta_frames[name] = meta
        mats[name] = X

    # Intersection of slide_ids
    shared = set(meta_frames[emb_names[0]]["slide_id"])
    for name in emb_names[1:]:
        shared &= set(meta_frames[name]["slide_id"])
    shared = sorted(shared)
    base = (meta_frames[emb_names[0]]
            [meta_frames[emb_names[0]]["slide_id"].isin(shared)]
            .drop_duplicates("slide_id")
            .set_index("slide_id").loc[shared].reset_index())
    base = base.merge(cl[["PATIENT", "y"]], on="PATIENT", how="inner")
    base = base.merge(st[["slide_id", "SITE"]], on="slide_id", how="left")
    base = base[base["SITE"].notna() & base["y"].notna()].reset_index(drop=True)

    # Reorder each matrix to match `base`
    aligned = {}
    for name in emb_names:
        m = meta_frames[name].set_index("slide_id")
        idx = m.loc[base["slide_id"], "_row"].values
        aligned[name] = mats[name][idx]
    return aligned, base


# --------------------------------------------------------- OOF pipelines -

def _proba(clf, Xte):
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(Xte)[:, 1]
    d = clf.decision_function(Xte)
    return 1.0 / (1.0 + np.exp(-d))


def patient_cv_oof(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                   model_ctor=None) -> np.ndarray:
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=np.float32)
    for tr, te in cv.split(X, y, groups=groups):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        clf = (model_ctor or (lambda: LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=SEED)))()
        clf.fit(Xtr, y[tr])
        oof[te] = _proba(clf, Xte)
    return oof


def loso_oof(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
             sites: np.ndarray, model_ctor=None) -> np.ndarray:
    """Leave-one-site-out with patient-level leakage guard. Returns OOF vector
    covering every row (each row gets predicted when its site is held out)."""
    oof = np.full(len(y), np.nan, dtype=np.float32)
    for site in np.unique(sites):
        held_patients = set(groups[sites == site])
        te = np.array([g in held_patients for g in groups])
        tr = ~te
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        clf = (model_ctor or (lambda: LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=SEED)))()
        clf.fit(Xtr, y[tr])
        oof[te] = _proba(clf, Xte)
    return oof


# --------------------------------------------------------- fusion heads --

def concat_pca_lr_oof(mats: dict[str, np.ndarray], df: pd.DataFrame,
                      regime: str, pca_dim: int = 100) -> np.ndarray:
    X_cat = np.hstack([StandardScaler().fit_transform(mats[m]) for m in mats]).astype(np.float32)
    y = df["y"].values
    groups = df["PATIENT"].values
    sites = df["SITE"].values

    if regime == "patient_cv":
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        oof = np.zeros(len(y), dtype=np.float32)
        for tr, te in cv.split(X_cat, y, groups=groups):
            pca = PCA(n_components=min(pca_dim, X_cat.shape[1], len(tr) - 1),
                      random_state=SEED)
            Xtr = pca.fit_transform(X_cat[tr])
            Xte = pca.transform(X_cat[te])
            clf = LogisticRegression(max_iter=2000, class_weight="balanced",
                                     random_state=SEED).fit(Xtr, y[tr])
            oof[te] = _proba(clf, Xte)
        return oof
    else:  # loso
        oof = np.full(len(y), np.nan, dtype=np.float32)
        for site in np.unique(sites):
            held = set(groups[sites == site])
            te = np.array([g in held for g in groups]); tr = ~te
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            pca = PCA(n_components=min(pca_dim, X_cat.shape[1], int(tr.sum()) - 1),
                      random_state=SEED)
            Xtr = pca.fit_transform(X_cat[tr])
            Xte = pca.transform(X_cat[te])
            clf = LogisticRegression(max_iter=2000, class_weight="balanced",
                                     random_state=SEED).fit(Xtr, y[tr])
            oof[te] = _proba(clf, Xte)
        return oof


def stacking_oof(per_model_oof: dict[str, np.ndarray], df: pd.DataFrame,
                 regime: str) -> np.ndarray:
    """Stack base OOF probs → meta-LR. OOF vector of stacked predictions."""
    names = list(per_model_oof)
    M = np.column_stack([per_model_oof[n] for n in names]).astype(np.float32)
    y = df["y"].values
    groups = df["PATIENT"].values
    sites = df["SITE"].values

    valid = ~np.isnan(M).any(axis=1)
    oof = np.full(len(y), np.nan, dtype=np.float32)

    if regime == "patient_cv":
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        idx = np.where(valid)[0]
        M_v, y_v, g_v = M[idx], y[idx], groups[idx]
        for tr, te in cv.split(M_v, y_v, groups=g_v):
            meta = LogisticRegression(max_iter=2000, class_weight="balanced",
                                      random_state=SEED).fit(M_v[tr], y_v[tr])
            oof[idx[te]] = _proba(meta, M_v[te])
    else:
        for site in np.unique(sites):
            held = set(groups[sites == site])
            te = np.array([g in held for g in groups]) & valid
            tr = (~np.array([g in held for g in groups])) & valid
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            meta = LogisticRegression(max_iter=2000, class_weight="balanced",
                                      random_state=SEED).fit(M[tr], y[tr])
            oof[te] = _proba(meta, M[te])
    return oof


# --------------------------------------------------------- few-shot -----

def knn_oof(X: np.ndarray, y: np.ndarray, groups: np.ndarray, sites: np.ndarray,
            regime: str, k: int) -> np.ndarray:
    oof = np.full(len(y), np.nan, dtype=np.float32)
    splits = []
    if regime == "patient_cv":
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        splits = list(cv.split(X, y, groups=groups))
    else:
        for site in np.unique(sites):
            held = set(groups[sites == site])
            te = np.array([g in held for g in groups]); tr = ~te
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            splits.append((np.where(tr)[0], np.where(te)[0]))
    for tr, te in splits:
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        knn = KNeighborsClassifier(n_neighbors=min(k, len(tr) - 1), metric="cosine")
        knn.fit(Xtr, y[tr])
        oof[te] = knn.predict_proba(Xte)[:, 1]
    return oof


def prototypical_oof(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                     sites: np.ndarray, regime: str) -> np.ndarray:
    oof = np.full(len(y), np.nan, dtype=np.float32)
    splits = []
    if regime == "patient_cv":
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        splits = list(cv.split(X, y, groups=groups))
    else:
        for site in np.unique(sites):
            held = set(groups[sites == site])
            te = np.array([g in held for g in groups]); tr = ~te
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            splits.append((np.where(tr)[0], np.where(te)[0]))
    for tr, te in splits:
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        p1 = Xtr[y[tr] == 1].mean(axis=0)
        p0 = Xtr[y[tr] == 0].mean(axis=0)
        p1 = p1 / (np.linalg.norm(p1) + 1e-9)
        p0 = p0 / (np.linalg.norm(p0) + 1e-9)
        Xn = Xte / (np.linalg.norm(Xte, axis=1, keepdims=True) + 1e-9)
        s1 = Xn @ p1
        s0 = Xn @ p0
        d = s1 - s0
        oof[te] = 1.0 / (1.0 + np.exp(-5 * d))  # temperature-sharpened sigmoid
    return oof


# --------------------------------------------------------- metrics ------

def score(y: np.ndarray, p: np.ndarray) -> dict:
    mask = ~np.isnan(p)
    if mask.sum() == 0 or len(np.unique(y[mask])) < 2:
        return dict(n=int(mask.sum()), auroc=float("nan"),
                    auprc=float("nan"), bal_acc=float("nan"))
    yi, pi = y[mask], p[mask]
    return dict(
        n=int(mask.sum()),
        auroc=float(roc_auc_score(yi, pi)),
        auprc=float(average_precision_score(yi, pi)),
        bal_acc=float(balanced_accuracy_score(yi, (pi >= 0.5).astype(int))),
    )


def score_per_site(y, p, sites) -> pd.DataFrame:
    rows = []
    for s in np.unique(sites):
        m = (sites == s) & ~np.isnan(p)
        if m.sum() == 0 or len(np.unique(y[m])) < 2:
            rows.append(dict(site=s, n=int(m.sum()), auroc=float("nan")))
            continue
        rows.append(dict(site=s, n=int(m.sum()),
                         auroc=float(roc_auc_score(y[m], p[m])),
                         prevalence=float(y[m].mean())))
    return pd.DataFrame(rows)


# --------------------------------------------------------- main ---------

def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    use = BASE_MODELS + EXTRA_MODELS
    mats, df = load_aligned(use)
    print(f"Aligned: {len(df)} slides, {df['PATIENT'].nunique()} patients, "
          f"prevalence {df['y'].mean():.3f}")

    y = df["y"].values
    groups = df["PATIENT"].values
    sites = df["SITE"].values

    # --- Per-model baselines (LR) -----------------------------------------
    per_model_oof_cv = {}
    per_model_oof_loso = {}
    rows = []
    for name in use:
        Xi = mats[name]
        oof_cv = patient_cv_oof(Xi, y, groups)
        oof_lo = loso_oof(Xi, y, groups, sites)
        per_model_oof_cv[name] = oof_cv
        per_model_oof_loso[name] = oof_lo
        rows.append(dict(strategy=f"single:{name}", regime="patient_cv", **score(y, oof_cv)))
        rows.append(dict(strategy=f"single:{name}", regime="loso", **score(y, oof_lo)))

    # Save per-model OOF arrays for reproducibility.
    oof_df = pd.DataFrame({f"cv__{k}": v for k, v in per_model_oof_cv.items()})
    for k, v in per_model_oof_loso.items():
        oof_df[f"loso__{k}"] = v
    oof_df.insert(0, "y", y)
    oof_df.insert(0, "site", sites)
    oof_df.insert(0, "PATIENT", groups)
    oof_df.insert(0, "slide_id", df["slide_id"].values)
    oof_df.to_csv(OUTDIR / "per_model_oof.csv", index=False)

    # --- Late-average fusion (all 2/3/4-subsets of BASE_MODELS) -----------
    for r in range(2, len(BASE_MODELS) + 1):
        for combo in itertools.combinations(BASE_MODELS, r):
            tag = "+".join(m.replace("_mean", "") for m in combo)
            p_cv = np.mean([per_model_oof_cv[m] for m in combo], axis=0)
            p_lo = np.nanmean([per_model_oof_loso[m] for m in combo], axis=0)
            rows.append(dict(strategy=f"late_avg:{tag}", regime="patient_cv", **score(y, p_cv)))
            rows.append(dict(strategy=f"late_avg:{tag}", regime="loso", **score(y, p_lo)))

    # --- Concat + PCA + LR (BASE_MODELS only) -----------------------------
    base_mats = {m: mats[m] for m in BASE_MODELS}
    p_cv = concat_pca_lr_oof(base_mats, df, "patient_cv")
    p_lo = concat_pca_lr_oof(base_mats, df, "loso")
    rows.append(dict(strategy="concat_pca100_lr:BASE", regime="patient_cv", **score(y, p_cv)))
    rows.append(dict(strategy="concat_pca100_lr:BASE", regime="loso", **score(y, p_lo)))

    # --- Stacking (meta-LR on base OOF) -----------------------------------
    base_oof_cv = {m: per_model_oof_cv[m] for m in BASE_MODELS}
    base_oof_lo = {m: per_model_oof_loso[m] for m in BASE_MODELS}
    p_cv = stacking_oof(base_oof_cv, df, "patient_cv")
    p_lo = stacking_oof(base_oof_lo, df, "loso")
    rows.append(dict(strategy="stack_meta_lr:BASE", regime="patient_cv", **score(y, p_cv)))
    rows.append(dict(strategy="stack_meta_lr:BASE", regime="loso", **score(y, p_lo)))

    # --- Few-shot (k-NN, prototypical) on each embedding ------------------
    for name in use:
        Xi = mats[name]
        for k in (3, 5, 9):
            p_cv = knn_oof(Xi, y, groups, sites, "patient_cv", k=k)
            p_lo = knn_oof(Xi, y, groups, sites, "loso", k=k)
            rows.append(dict(strategy=f"knn{k}:{name}", regime="patient_cv", **score(y, p_cv)))
            rows.append(dict(strategy=f"knn{k}:{name}", regime="loso", **score(y, p_lo)))
        p_cv = prototypical_oof(Xi, y, groups, sites, "patient_cv")
        p_lo = prototypical_oof(Xi, y, groups, sites, "loso")
        rows.append(dict(strategy=f"proto:{name}", regime="patient_cv", **score(y, p_cv)))
        rows.append(dict(strategy=f"proto:{name}", regime="loso", **score(y, p_lo)))

    results = pd.DataFrame(rows)
    results.to_csv(OUTDIR / "fusion_results.csv", index=False)

    # Print a ranked summary per regime
    print("\n=== Top 15 by patient_cv AUROC ===")
    print(results[results.regime == "patient_cv"]
          .sort_values("auroc", ascending=False).head(15)
          [["strategy", "auroc", "auprc"]].to_string(index=False))
    print("\n=== Top 15 by LOSO AUROC ===")
    print(results[results.regime == "loso"]
          .sort_values("auroc", ascending=False).head(15)
          [["strategy", "auroc", "auprc"]].to_string(index=False))

    # Plot: best-of-each-strategy-class for each regime
    def _bucket(s):
        if s.startswith("single:"):       return "single"
        if s.startswith("late_avg:"):     return "late_avg"
        if s.startswith("concat_pca"):    return "concat_pca"
        if s.startswith("stack_"):        return "stacking"
        if s.startswith("knn"):           return "knn"
        if s.startswith("proto:"):        return "proto"
        return "other"
    results["bucket"] = results["strategy"].apply(_bucket)
    best = (results.loc[results.groupby(["bucket", "regime"])["auroc"].idxmax()]
                   [["bucket", "regime", "strategy", "auroc", "auprc"]]
                   .sort_values(["regime", "auroc"], ascending=[True, False]))
    best.to_csv(OUTDIR / "best_per_bucket.csv", index=False)

    buckets_order = ["single", "late_avg", "concat_pca", "stacking", "knn", "proto"]
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
    x = np.arange(len(buckets_order))
    w = 0.4
    for i, reg in enumerate(["patient_cv", "loso"]):
        vals = [best[(best["bucket"] == b) & (best["regime"] == reg)]["auroc"]
                .iloc[0] if ((best["bucket"] == b) & (best["regime"] == reg)).any()
                else np.nan for b in buckets_order]
        ax.bar(x + (i - 0.5) * w, vals, w, label=reg,
               color=("#4477AA" if reg == "patient_cv" else "#CC3311"))
    ax.set_xticks(x)
    ax.set_xticklabels(buckets_order)
    ax.axhline(0.5, ls="--", c="grey", lw=0.7)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Best AUROC")
    ax.set_title("Step 2d / 2e — best of each strategy class\n(patient_cv vs leave-one-site-out)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUTDIR / "best_per_bucket.png", bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved: {OUTDIR/'fusion_results.csv'}")
    print(f"Saved: {OUTDIR/'per_model_oof.csv'}")
    print(f"Saved: {OUTDIR/'best_per_bucket.csv'}")
    print(f"Saved: {OUTDIR/'best_per_bucket.png'}")


if __name__ == "__main__":
    main()
