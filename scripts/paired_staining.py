"""A1 — Paired staining analysis (within-patient, across-staining).

For retrospective patients who have slides stained both at MSKCC and in
Nigeria (same patient, same scanner — all imaging done in Nigeria), compare
the MSI-H predictions produced by each staining origin.

Methodology:
    1. Load slide-level embeddings (default: conch_v1.5_mean, the top
       Phase 1 baseline).
    2. Attach MSI labels (from clinical_table.csv) and staining metadata
       (from slide_table_pyramidal.csv).
    3. Compute per-slide OOF P(MSI-H) via StratifiedGroupKFold on
       patient_id — no patient appears in both its own train and test folds.
    4. Subset to retrospective patients with slides from both MSKCC and
       ANY Nigerian stain location (OAUTHC / LUTH / LASUTH / UITH).
    5. Aggregate to (patient × staining_origin): mean P(MSI-H).
    6. Report concordance metrics and save a paired scatter plot.

Usage:
    python scripts/paired_staining.py
    python scripts/paired_staining.py --embeddings results/embeddings/virchow2_mean

Outputs:
    results/analysis/paired_staining/
        paired_scores.csv       — one row per (patient, stain_origin)
        metrics.json            — Cohen's kappa, Wilcoxon p, Pearson r, N
        paired_scatter.png      — P(MSI-H | MSK) vs P(MSI-H | Nigeria)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

NIGERIA_STAINS = {"OAUTHC", "LUTH", "LASUTH", "UITH"}
MSK_STAIN = "MSKCC"


def load_slide_scores(embeddings_dir: Path, clinical: Path, slide_table: Path) -> pd.DataFrame:
    meta = pd.read_csv(embeddings_dir / "metadata.csv")
    X = np.load(embeddings_dir / "embeddings.npy")
    if len(meta) != len(X):
        raise ValueError(f"metadata/embeddings row mismatch: {len(meta)} vs {len(X)}")

    cl = pd.read_csv(clinical)[["PATIENT", "isMSIH"]]
    cl["y"] = (cl["isMSIH"] == "MSI-H").astype(int)

    st = pd.read_csv(slide_table)[["FILENAME", "PATIENT", "SITE", "stain_location"]]
    st["slide_id"] = st["FILENAME"].apply(lambda p: Path(p).stem)

    df = (
        meta.rename(columns={"patient_id": "PATIENT"})
        .merge(cl[["PATIENT", "y"]], on="PATIENT", how="inner")
        .merge(st[["slide_id", "SITE", "stain_location"]], on="slide_id", how="left")
    )
    keep = df.index[df["y"].notna() & df["stain_location"].notna()]
    df = df.loc[keep].reset_index(drop=True)
    X = X[keep]

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    proba = cross_val_predict(
        model, Xs, df["y"].values, cv=cv, groups=df["PATIENT"].values, method="predict_proba"
    )
    df["p_msih"] = proba[:, 1]
    return df


def build_paired(df: pd.DataFrame) -> pd.DataFrame:
    retro = df[df["SITE"].str.startswith("retrospective", na=False)].copy()
    retro["stain_origin"] = np.where(
        retro["stain_location"].eq(MSK_STAIN),
        "MSK",
        np.where(retro["stain_location"].isin(NIGERIA_STAINS), "Nigeria", "other"),
    )
    retro = retro[retro["stain_origin"].isin(["MSK", "Nigeria"])]
    per_patient = (
        retro.groupby(["PATIENT", "stain_origin"])
        .agg(p_msih=("p_msih", "mean"), y=("y", "first"), n_slides=("slide_id", "count"))
        .reset_index()
    )
    wide = per_patient.pivot(index="PATIENT", columns="stain_origin", values="p_msih")
    wide = wide.dropna(subset=["MSK", "Nigeria"])
    label = per_patient.groupby("PATIENT")["y"].first().reindex(wide.index)
    wide["y"] = label
    return wide.reset_index()


def compute_metrics(paired: pd.DataFrame) -> dict:
    msk = paired["MSK"].values
    nig = paired["Nigeria"].values
    threshold = 0.5
    kappa = cohen_kappa_score(msk > threshold, nig > threshold)
    pearson_r, pearson_p = pearsonr(msk, nig)
    try:
        stat, wilc_p = wilcoxon(msk, nig)
    except ValueError:
        stat, wilc_p = np.nan, np.nan
    return {
        "n_patients": int(len(paired)),
        "cohen_kappa_binary@0.5": float(kappa),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "wilcoxon_statistic": float(stat),
        "wilcoxon_p": float(wilc_p),
        "mean_abs_diff": float(np.abs(msk - nig).mean()),
        "mean_msk_minus_nigeria": float(np.mean(msk - nig)),
    }


def plot_paired(paired: pd.DataFrame, outfile: Path, title: str) -> None:
    colors = paired["y"].map({1: "tab:red", 0: "tab:blue"})
    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    ax.scatter(paired["MSK"], paired["Nigeria"], c=colors, s=30, alpha=0.75, edgecolors="none")
    ax.plot([0, 1], [0, 1], ls="--", c="gray", lw=1)
    ax.axhline(0.5, c="lightgray", lw=0.5)
    ax.axvline(0.5, c="lightgray", lw=0.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("P(MSI-H) — MSK-stained slides")
    ax.set_ylabel("P(MSI-H) — Nigeria-stained slides")
    ax.set_title(title)
    for label, color in [("MSI-H", "tab:red"), ("MSS", "tab:blue")]:
        ax.scatter([], [], c=color, label=label, s=30)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outfile)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings", default="results/embeddings/conch_v1.5_mean")
    p.add_argument("--clinical", default="results/data/clinical_table.csv")
    p.add_argument("--slide-table", default="results/data/slide_table_pyramidal.csv")
    p.add_argument("--outdir", default="results/analysis/paired_staining")
    args = p.parse_args()

    emb = Path(args.embeddings)
    outdir = Path(args.outdir) / emb.name
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_slide_scores(emb, Path(args.clinical), Path(args.slide_table))
    paired = build_paired(df)
    metrics = compute_metrics(paired)

    paired.to_csv(outdir / "paired_scores.csv", index=False)
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_paired(paired, outdir / "paired_scatter.png", title=f"Paired staining — {emb.name}")

    print(f"\nEmbedding: {emb.name}")
    print(f"Retrospective patients with paired MSK + Nigeria slides: {metrics['n_patients']}")
    print(f"  MSI-H: {int((paired['y'] == 1).sum())}  MSS: {int((paired['y'] == 0).sum())}")
    print(f"Cohen's kappa (binary @ 0.5):  {metrics['cohen_kappa_binary@0.5']:.3f}")
    print(f"Pearson r:                     {metrics['pearson_r']:.3f}  (p={metrics['pearson_p']:.3g})")
    print(f"Wilcoxon signed-rank p:        {metrics['wilcoxon_p']:.3g}")
    print(f"Mean |MSK - Nigeria|:          {metrics['mean_abs_diff']:.3f}")
    print(f"Mean (MSK - Nigeria):          {metrics['mean_msk_minus_nigeria']:+.3f}")
    print(f"\nSaved to {outdir}")


if __name__ == "__main__":
    main()
