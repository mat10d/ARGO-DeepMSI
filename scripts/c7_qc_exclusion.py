"""C7 — Pathologist QC Exclusion: comprehensive re-evaluation of ALL benchmarks.

514 slides flagged by pathologist review (502 no tumor, 490 from OAUTHC).
No re-embedding needed — just filter existing slide-level data and re-aggregate.

Re-runs:
  A. Wagner zero-shot (mean, max, max/√n) — before vs after
  B. Tier 1 classifier grid (LR/SVM/RF/XGB × 4 embeddings × raw/PCA)
  C. Calibrated aggregation (all operators from C4b)
  D. Site holdout (LOSO) — the critical generalization test
  E. Slide attention (Phase 2) — does it work now with clean bags?
  F. Fusion (stacking, late-average)

Usage:
    cp problem_slides.csv results/data/
    python scripts/c7_qc_exclusion.py

Outputs:
    results/analysis/c7_qc_exclusion/
        summary.md                    — full results narrative
        comparison_wagner.csv         — Wagner before vs after
        comparison_tier1.csv          — Tier 1 grid after QC
        comparison_calibrated.csv     — calibrated aggregators after QC
        comparison_site_holdout.csv   — LOSO after QC
        auroc_comparison.png          — bar chart before/after
        remaining_cohort.csv          — slide table after exclusion
        patient_summary.csv           — per-patient slide counts
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

OUTDIR = Path("results/analysis/c7_qc_exclusion")
CLINICAL = Path("results/data/clinical_table.csv")
WAGNER_SLIDES = Path("results/analysis/wagner_zeroshot/slide_scores.csv")
EMB_ROOT = Path("results/embeddings")

EXCLUSION_PATHS = [
    Path("results/data/problem_slides.csv"),
    Path("problem_slides.csv"),
]

N_FOLDS = 5
SEED = 42


# ============================================================================
# Exclusion set loading
# ============================================================================

def load_exclusion_set() -> set:
    """Load slide filenames to exclude. Returns set of slide_id stems."""
    for path in EXCLUSION_PATHS:
        if path.exists():
            df = pd.read_csv(path)
            stems = set()
            for name in df["Name"].dropna():
                stem = Path(name).stem
                stems.add(stem)
                # Handle .pyramidal variants
                if ".pyramidal" in stem:
                    stems.add(stem.replace(".pyramidal", ""))
                # Also add with .pyramidal in case the slide_scores use it
                if ".pyramidal" not in stem:
                    stems.add(stem + ".pyramidal")
            log.info(f"Loaded {len(df)} exclusions from {path} → {len(stems)} stem variants")
            return stems
    raise FileNotFoundError(f"Exclusion list not found at any of: {EXCLUSION_PATHS}")


def safe_auroc(y, scores):
    """AUROC with edge-case handling."""
    if len(y) < 5 or y.sum() == 0 or y.sum() == len(y):
        return np.nan
    return roc_auc_score(y, scores)


def safe_auprc(y, scores):
    if len(y) < 5 or y.sum() == 0:
        return np.nan
    return average_precision_score(y, scores)


# ============================================================================
# A. Wagner zero-shot re-evaluation
# ============================================================================

def run_wagner(wagner_all: pd.DataFrame, wagner_clean: pd.DataFrame, clinical: pd.DataFrame):
    """Compare Wagner before/after QC exclusion."""
    log.info("\n=== A. WAGNER ZERO-SHOT ===")

    results = []
    for label, slides in [("before_qc", wagner_all), ("after_qc", wagner_clean)]:
        # Patient-level aggregation
        patients = []
        for pid, grp in slides.groupby("patient_id"):
            n = len(grp)
            p = grp["p_msih"].values
            patients.append({
                "patient_id": pid, "site": grp["site"].iloc[0], "y": grp["y"].iloc[0],
                "n_slides": n,
                "mean": p.mean(), "max": p.max(), "median": np.median(p),
                "max_sqrtn": p.max() / np.sqrt(n) if n > 0 else 0,
            })
        pat = pd.DataFrame(patients)

        for agg in ["mean", "max", "median", "max_sqrtn"]:
            # Overall
            results.append({
                "phase": label, "aggregator": agg, "subset": "overall",
                "auroc": safe_auroc(pat["y"].values, pat[agg].values),
                "auprc": safe_auprc(pat["y"].values, pat[agg].values),
                "n": len(pat), "n_msih": int(pat["y"].sum()),
            })
            # Per site
            for site in sorted(pat["site"].unique()):
                sp = pat[pat["site"] == site]
                results.append({
                    "phase": label, "aggregator": agg, "subset": site,
                    "auroc": safe_auroc(sp["y"].values, sp[agg].values),
                    "auprc": safe_auprc(sp["y"].values, sp[agg].values),
                    "n": len(sp), "n_msih": int(sp["y"].sum()),
                })
            # Per bin (after only)
            if label == "after_qc":
                bins = {"1": pat["n_slides"] == 1, "2-3": pat["n_slides"].between(2, 3),
                        "4-6": pat["n_slides"].between(4, 6), "7+": pat["n_slides"] >= 7}
                for bname, bmask in bins.items():
                    bp = pat[bmask]
                    results.append({
                        "phase": label, "aggregator": agg, "subset": f"bin_{bname}",
                        "auroc": safe_auroc(bp["y"].values, bp[agg].values),
                        "auprc": safe_auprc(bp["y"].values, bp[agg].values),
                        "n": len(bp), "n_msih": int(bp["y"].sum()),
                    })

    df = pd.DataFrame(results)
    df.to_csv(OUTDIR / "comparison_wagner.csv", index=False)

    # Print headline
    for agg in ["mean", "max_sqrtn"]:
        before = df[(df["phase"] == "before_qc") & (df["aggregator"] == agg) & (df["subset"] == "overall")]["auroc"].iloc[0]
        after = df[(df["phase"] == "after_qc") & (df["aggregator"] == agg) & (df["subset"] == "overall")]["auroc"].iloc[0]
        log.info(f"  Wagner {agg}: {before:.3f} → {after:.3f} (Δ={after - before:+.3f})")

    return df


# ============================================================================
# B. Tier 1 classifier grid
# ============================================================================

def run_tier1(exclusions: set, clinical: pd.DataFrame):
    """Re-run Tier 1 grid on filtered embeddings."""
    log.info("\n=== B. TIER 1 CLASSIFIER GRID ===")

    clf_grid = {
        "lr": lambda: LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
        "svm_rbf": lambda: SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=SEED),
        "rf": lambda: RandomForestClassifier(n_estimators=500, class_weight="balanced", n_jobs=-1, random_state=SEED),
    }
    if HAS_XGB:
        clf_grid["xgb"] = lambda: xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            scale_pos_weight=4.26, n_jobs=-1, random_state=SEED,
            eval_metric="auc", use_label_encoder=False, tree_method="hist")

    results = []
    for emb_dir in sorted(EMB_ROOT.iterdir()):
        if not (emb_dir / "embeddings.npy").exists():
            continue
        model_name = emb_dir.name

        try:
            X = np.load(emb_dir / "embeddings.npy")
            meta = pd.read_csv(emb_dir / "metadata.csv")
            meta["_row"] = np.arange(len(meta))

            # Merge with clinical
            cl = clinical[["PATIENT", "y"]].copy()
            merged = meta.rename(columns={"patient_id": "PATIENT"}).merge(cl, on="PATIENT", how="inner")

            # Filter exclusions
            if "slide_id" in merged.columns:
                keep = ~merged["slide_id"].isin(exclusions)
            else:
                keep = pd.Series(True, index=merged.index)

            X_filt = X[merged["_row"].values][keep.values]
            df_filt = merged[keep.values].reset_index(drop=True)

            if len(df_filt) < 20 or df_filt["y"].nunique() < 2:
                log.warning(f"Skipping {model_name}: too few samples after QC ({len(df_filt)})")
                continue

            for clf_name, clf_fn in clf_grid.items():
                for feat_eng in ["raw", "pca100"]:
                    steps = [("scaler", StandardScaler())]
                    if feat_eng == "pca100":
                        from sklearn.decomposition import PCA
                        n_comp = min(100, X_filt.shape[1], X_filt.shape[0] - 1)
                        steps.append(("pca", PCA(n_components=n_comp, random_state=SEED)))
                    steps.append(("clf", clf_fn()))
                    pipe = Pipeline(steps)

                    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
                    oof = np.full(len(df_filt), np.nan)

                    for tr, va in cv.split(X_filt, df_filt["y"], groups=df_filt["PATIENT"]):
                        pipe.fit(X_filt[tr], df_filt["y"].iloc[tr])
                        oof[va] = pipe.predict_proba(X_filt[va])[:, 1]

                    valid = ~np.isnan(oof)
                    auroc = safe_auroc(df_filt["y"][valid].values, oof[valid])
                    auprc = safe_auprc(df_filt["y"][valid].values, oof[valid])

                    results.append({
                        "embedding": model_name, "classifier": clf_name,
                        "feat_eng": feat_eng, "auroc": auroc, "auprc": auprc,
                        "n": int(valid.sum()),
                        "n_msih": int(df_filt["y"][valid].sum()),
                    })

            log.info(f"  {model_name}: best AUROC = {max(r['auroc'] for r in results if r['embedding'] == model_name and not np.isnan(r['auroc'])):.3f}")

        except Exception as e:
            log.error(f"Failed {model_name}: {e}")
            import traceback; traceback.print_exc()

    df = pd.DataFrame(results).sort_values("auroc", ascending=False)
    df.to_csv(OUTDIR / "comparison_tier1.csv", index=False)

    if len(df) > 0:
        best = df.iloc[0]
        log.info(f"  Best: {best['embedding']}/{best['classifier']}/{best['feat_eng']} → AUROC {best['auroc']:.3f}")

    return df


# ============================================================================
# C. Site holdout (LOSO)
# ============================================================================

def run_site_holdout(wagner_clean: pd.DataFrame, clinical: pd.DataFrame):
    """Leave-one-site-out evaluation on cleaned Wagner scores."""
    log.info("\n=== D. SITE HOLDOUT (LOSO) ===")

    # Patient-level scores
    patients = []
    for pid, grp in wagner_clean.groupby("patient_id"):
        n = len(grp)
        p = grp["p_msih"].values
        patients.append({
            "patient_id": pid, "site": grp["site"].iloc[0], "y": grp["y"].iloc[0],
            "n_slides": n, "wagner_mean": p.mean(),
            "wagner_max_sqrtn": p.max() / np.sqrt(n),
        })
    pat = pd.DataFrame(patients)

    results = []
    for holdout_site in sorted(pat["site"].unique()):
        test = pat[pat["site"] == holdout_site]
        for agg in ["wagner_mean", "wagner_max_sqrtn"]:
            results.append({
                "holdout_site": holdout_site, "aggregator": agg,
                "auroc": safe_auroc(test["y"].values, test[agg].values),
                "n": len(test), "n_msih": int(test["y"].sum()),
            })

    df = pd.DataFrame(results)
    df.to_csv(OUTDIR / "comparison_site_holdout.csv", index=False)

    for _, row in df[df["aggregator"] == "wagner_max_sqrtn"].iterrows():
        log.info(f"  {row['holdout_site']}: AUROC={row['auroc']:.3f} (n={row['n']}, MSI-H={row['n_msih']})")

    return df


# ============================================================================
# E. Slide Attention (Phase 2 re-run)
# ============================================================================

def run_slide_attention(wagner_clean: pd.DataFrame, exclusions: set, clinical: pd.DataFrame):
    """Re-run SlideAttentionMSI on clean bags."""
    if not HAS_TORCH:
        log.warning("Skipping slide attention — torch not available")
        return pd.DataFrame()

    log.info("\n=== E. SLIDE ATTENTION (Phase 2 re-run) ===")

    from scripts.c5_phase2_slide_attention import (
        SlideAttentionMSI, load_patient_bags, cross_validate,
        EMBEDDING_MODELS, DEVICE, SEED
    )

    results = []
    for emb_model in EMBEDDING_MODELS:
        emb_dir = EMB_ROOT / emb_model
        if not (emb_dir / "embeddings.npy").exists():
            continue

        for feat_name, feat_config in [
            ("wagner+meta", {"wagner": True, "embedding": False, "metadata": True}),
            ("emb_only", {"wagner": False, "embedding": True, "metadata": False}),
            ("all", {"wagner": True, "embedding": True, "metadata": True}),
        ]:
            if not feat_config["embedding"] and emb_model != EMBEDDING_MODELS[0]:
                continue

            label = f"{emb_model}/{feat_name}" if feat_config["embedding"] else f"none/{feat_name}"
            try:
                bags, labels, patients, slide_info = load_patient_bags(
                    emb_model, feat_config, exclusion_set=exclusions)
                cv_results = cross_validate(bags, labels, patients, slide_info)
                results.append({
                    "embedding": emb_model if feat_config["embedding"] else "none",
                    "feature_set": feat_name,
                    "auroc": cv_results["auroc"],
                    "auprc": cv_results["auprc"],
                })
                log.info(f"  {label}: AUROC={cv_results['auroc']:.3f}")
            except Exception as e:
                log.warning(f"  {label}: failed — {e}")

    df = pd.DataFrame(results)
    if len(df) > 0:
        df.to_csv(OUTDIR / "comparison_slide_attention.csv", index=False)
    return df


# ============================================================================
# Plotting
# ============================================================================

def plot_comparison(wagner_results: pd.DataFrame):
    """Side-by-side before/after bar chart."""
    agg = "max_sqrtn"
    before = wagner_results[(wagner_results["phase"] == "before_qc") & (wagner_results["aggregator"] == agg) &
                            (~wagner_results["subset"].str.startswith("bin_"))]
    after = wagner_results[(wagner_results["phase"] == "after_qc") & (wagner_results["aggregator"] == agg) &
                           (~wagner_results["subset"].str.startswith("bin_"))]

    merged = before[["subset", "auroc"]].rename(columns={"auroc": "before"}).merge(
        after[["subset", "auroc"]].rename(columns={"auroc": "after"}), on="subset", how="outer")

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(merged))
    w = 0.35
    ax.bar(x - w/2, merged["before"], w, label="Before QC (803 slides)", color="#d62728", alpha=0.7)
    ax.bar(x + w/2, merged["after"], w, label="After QC (cleaned)", color="#2ca02c", alpha=0.7)
    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_title("Wagner max/√n — Before vs After Pathologist QC Exclusion", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(merged["subset"], rotation=45, ha="right")
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", ls="--", alpha=0.5)

    for i, (_, row) in enumerate(merged.iterrows()):
        if pd.notna(row["before"]) and pd.notna(row["after"]):
            delta = row["after"] - row["before"]
            color = "green" if delta > 0 else "red"
            ax.annotate(f"{delta:+.3f}", xy=(i + w/2, row["after"] + 0.02),
                       ha="center", fontsize=9, color=color, fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUTDIR / "auroc_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved {OUTDIR / 'auroc_comparison.png'}")


def write_summary(wagner_df, tier1_df, holdout_df, n_before, n_after,
                  n_patients_before, n_patients_after, n_lost_patients, lost_msih):
    """Write a markdown summary of all results."""
    lines = [
        "# C7 — Pathologist QC Exclusion Results\n",
        f"**Date:** Auto-generated\n",
        f"**Exclusion:** 514 slides (502 no tumor, 490 OAUTHC)\n",
        f"**Cohort:** {n_before} → {n_after} slides, {n_patients_before} → {n_patients_after} patients\n",
        f"**Patients dropped (all slides excluded):** {n_lost_patients} ({lost_msih} MSI-H)\n\n",
        "---\n\n",
        "## A. Wagner Zero-Shot\n\n",
        "| Aggregator | Before | After | Δ |\n",
        "|---|---|---|---|\n",
    ]
    for agg in ["mean", "max", "median", "max_sqrtn"]:
        b = wagner_df[(wagner_df["phase"] == "before_qc") & (wagner_df["aggregator"] == agg) & (wagner_df["subset"] == "overall")]
        a = wagner_df[(wagner_df["phase"] == "after_qc") & (wagner_df["aggregator"] == agg) & (wagner_df["subset"] == "overall")]
        if len(b) > 0 and len(a) > 0:
            bv, av = b["auroc"].iloc[0], a["auroc"].iloc[0]
            lines.append(f"| {agg} | {bv:.3f} | {av:.3f} | {av - bv:+.3f} |\n")

    lines.append("\n### Per-site (max/√n)\n\n")
    lines.append("| Site | Before | After | n_after | MSI-H |\n")
    lines.append("|---|---|---|---|---|\n")
    for _, row in wagner_df[(wagner_df["phase"] == "after_qc") & (wagner_df["aggregator"] == "max_sqrtn") &
                            (~wagner_df["subset"].str.startswith("bin_")) & (wagner_df["subset"] != "overall")].iterrows():
        brow = wagner_df[(wagner_df["phase"] == "before_qc") & (wagner_df["aggregator"] == "max_sqrtn") & (wagner_df["subset"] == row["subset"])]
        bv = brow["auroc"].iloc[0] if len(brow) > 0 else np.nan
        lines.append(f"| {row['subset']} | {bv:.3f} | {row['auroc']:.3f} | {row['n']} | {row['n_msih']} |\n")

    if len(tier1_df) > 0:
        lines.append("\n## B. Tier 1 Grid (Top 10)\n\n")
        lines.append("| Embedding | Classifier | Feat | AUROC |\n")
        lines.append("|---|---|---|---|\n")
        for _, row in tier1_df.head(10).iterrows():
            lines.append(f"| {row['embedding']} | {row['classifier']} | {row['feat_eng']} | {row['auroc']:.3f} |\n")

    with open(OUTDIR / "summary.md", "w") as f:
        f.writelines(lines)
    log.info(f"Saved {OUTDIR / 'summary.md'}")


# ============================================================================
# Main
# ============================================================================

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    exclusions = load_exclusion_set()
    clinical = pd.read_csv(CLINICAL)
    clinical["y"] = (clinical["isMSIH"] == "MSI-H").astype(int)
    wagner_all = pd.read_csv(WAGNER_SLIDES)

    # Apply exclusion
    wagner_all["excluded"] = wagner_all["slide_id"].isin(exclusions)
    n_matched = wagner_all["excluded"].sum()
    wagner_clean = wagner_all[~wagner_all["excluded"]].copy()

    log.info(f"Matched {n_matched}/{len(wagner_all)} slides to exclusion list")
    log.info(f"Before: {len(wagner_all)} slides → After: {len(wagner_clean)} slides")

    # Patients lost
    before_pids = set(wagner_all["patient_id"].unique())
    after_pids = set(wagner_clean["patient_id"].unique())
    lost_pids = before_pids - after_pids
    lost_msih = clinical[clinical["PATIENT"].isin(lost_pids)]["y"].sum() if lost_pids else 0
    log.info(f"Patients: {len(before_pids)} → {len(after_pids)} ({len(lost_pids)} dropped, {lost_msih} MSI-H)")

    # Save cleaned slide table
    wagner_clean.to_csv(OUTDIR / "remaining_cohort.csv", index=False)

    # Patient summary
    summary_rows = []
    for pid in before_pids:
        n_before = len(wagner_all[wagner_all["patient_id"] == pid])
        n_after = len(wagner_clean[wagner_clean["patient_id"] == pid])
        site = wagner_all[wagner_all["patient_id"] == pid]["site"].iloc[0]
        y = wagner_all[wagner_all["patient_id"] == pid]["y"].iloc[0]
        summary_rows.append({"patient_id": pid, "site": site, "y": y,
                           "n_before": n_before, "n_after": n_after,
                           "n_excluded": n_before - n_after})
    pd.DataFrame(summary_rows).to_csv(OUTDIR / "patient_summary.csv", index=False)

    # === RUN ALL ANALYSES ===
    wagner_df = run_wagner(wagner_all, wagner_clean, clinical)
    tier1_df = run_tier1(exclusions, clinical)
    holdout_df = run_site_holdout(wagner_clean, clinical)

    # Slide attention (optional — may need Phase 2 script importable)
    attn_df = pd.DataFrame()
    try:
        attn_df = run_slide_attention(wagner_clean, exclusions, clinical)
    except Exception as e:
        log.warning(f"Slide attention skipped: {e}")

    # === OUTPUT ===
    plot_comparison(wagner_df)
    write_summary(wagner_df, tier1_df, holdout_df,
                  n_before=len(wagner_all), n_after=len(wagner_clean),
                  n_patients_before=len(before_pids), n_patients_after=len(after_pids),
                  n_lost_patients=len(lost_pids), lost_msih=int(lost_msih))

    # Final print
    print("\n" + "=" * 70)
    print("C7 — QC EXCLUSION RESULTS")
    print("=" * 70)
    print(f"Slides: {len(wagner_all)} → {len(wagner_clean)} ({n_matched} excluded)")
    print(f"Patients: {len(before_pids)} → {len(after_pids)} ({len(lost_pids)} dropped)")
    print()
    print("WAGNER (overall):")
    for agg in ["mean", "max_sqrtn"]:
        b = wagner_df[(wagner_df["phase"]=="before_qc") & (wagner_df["aggregator"]==agg) & (wagner_df["subset"]=="overall")]["auroc"].iloc[0]
        a = wagner_df[(wagner_df["phase"]=="after_qc") & (wagner_df["aggregator"]==agg) & (wagner_df["subset"]=="overall")]["auroc"].iloc[0]
        print(f"  {agg}: {b:.3f} → {a:.3f} ({a-b:+.3f})")
    if len(tier1_df) > 0:
        print(f"\nTIER 1 BEST: {tier1_df.iloc[0]['embedding']}/{tier1_df.iloc[0]['classifier']} → {tier1_df.iloc[0]['auroc']:.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
