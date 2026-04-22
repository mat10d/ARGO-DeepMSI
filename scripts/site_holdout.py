"""A2 — Leave-one-site-out evaluation.

For each Nigerian site S, train on all patients NOT present in S, test on
the slides from S. Explicit patient-level guard: any patient with even one
slide in the held-out site is fully excluded from training (this matters
for retrospective patients whose PATIENT id spans retrospective_msk and
retrospective_oau).

Produces a single-number per (embedding, holdout_site) of test AUROC +
AUPRC, plus a comparison figure across embeddings.

Usage:
    python scripts/site_holdout.py
    python scripts/site_holdout.py --embeddings results/embeddings/virchow2_mean
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

RESULT_COLS = ["embedding", "holdout_site", "n_train_slides", "n_train_patients",
               "n_test_slides", "n_test_patients", "test_prevalence",
               "auroc", "auprc", "balanced_acc"]


def load_frame(embeddings_dir: Path, clinical: Path, slide_table: Path) -> tuple[np.ndarray, pd.DataFrame]:
    meta = pd.read_csv(embeddings_dir / "metadata.csv")
    X = np.load(embeddings_dir / "embeddings.npy")
    if len(meta) != len(X):
        raise ValueError("metadata/embeddings mismatch")

    cl = pd.read_csv(clinical)[["PATIENT", "isMSIH"]]
    cl["y"] = (cl["isMSIH"] == "MSI-H").astype(int)

    st = pd.read_csv(slide_table)[["FILENAME", "PATIENT", "SITE"]]
    st["slide_id"] = st["FILENAME"].apply(lambda p: Path(p).stem)

    df = (
        meta.rename(columns={"patient_id": "PATIENT"})
        .merge(cl[["PATIENT", "y"]], on="PATIENT", how="inner")
        .merge(st[["slide_id", "SITE"]], on="slide_id", how="left")
    )
    keep = df.index[df["y"].notna() & df["SITE"].notna()]
    df = df.loc[keep].reset_index(drop=True)
    X = X[keep]
    return X, df


def run_site(X: np.ndarray, df: pd.DataFrame, holdout: str) -> dict:
    # Patient-level leakage guard: hold out EVERY slide of any patient who
    # has at least one slide in the held-out site.
    held_patients = set(df.loc[df["SITE"] == holdout, "PATIENT"])
    test_mask = df["PATIENT"].isin(held_patients).values
    train_mask = ~test_mask

    y = df["y"].values
    if len(np.unique(y[train_mask])) < 2 or len(np.unique(y[test_mask])) < 2:
        return dict(
            holdout_site=holdout,
            n_train_slides=int(train_mask.sum()),
            n_train_patients=df.loc[train_mask, "PATIENT"].nunique(),
            n_test_slides=int(test_mask.sum()),
            n_test_patients=df.loc[test_mask, "PATIENT"].nunique(),
            test_prevalence=float(y[test_mask].mean()) if test_mask.any() else float("nan"),
            auroc=float("nan"), auprc=float("nan"), balanced_acc=float("nan"),
            note="skipped: test or train has only one class",
        )

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X[train_mask])
    Xte = scaler.transform(X[test_mask])

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    clf.fit(Xtr, y[train_mask])
    proba = clf.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)

    return dict(
        holdout_site=holdout,
        n_train_slides=int(train_mask.sum()),
        n_train_patients=df.loc[train_mask, "PATIENT"].nunique(),
        n_test_slides=int(test_mask.sum()),
        n_test_patients=df.loc[test_mask, "PATIENT"].nunique(),
        test_prevalence=float(y[test_mask].mean()),
        auroc=float(roc_auc_score(y[test_mask], proba)),
        auprc=float(average_precision_score(y[test_mask], proba)),
        balanced_acc=float(balanced_accuracy_score(y[test_mask], pred)),
    )


def run_one_embedding(emb_dir: Path, clinical: Path, slide_table: Path, outdir: Path) -> pd.DataFrame:
    X, df = load_frame(emb_dir, clinical, slide_table)
    sites = sorted(df["SITE"].dropna().unique())
    rows = []
    for site in sites:
        res = run_site(X, df, site)
        res["embedding"] = emb_dir.name
        rows.append(res)
    out = pd.DataFrame(rows)
    outdir.mkdir(parents=True, exist_ok=True)
    out.to_csv(outdir / "per_site.csv", index=False)
    return out


def summary_figure(df: pd.DataFrame, outfile: Path) -> None:
    pivot = df.pivot(index="embedding", columns="holdout_site", values="auroc")
    fig, ax = plt.subplots(figsize=(1.2 * len(pivot.columns) + 2, 0.6 * len(pivot.index) + 2), dpi=150)
    im = ax.imshow(pivot.values, vmin=0.3, vmax=0.9, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if (v < 0.45 or v > 0.75) else "black", fontsize=9)
    ax.set_xlabel("Held-out site")
    ax.set_ylabel("Embedding")
    ax.set_title("A2 — Leave-one-site-out AUROC")
    fig.colorbar(im, ax=ax, shrink=0.8, label="AUROC")
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings", default=None, help="Single embedding dir; if omitted, sweep all")
    p.add_argument("--clinical", default="results/data/clinical_table.csv")
    p.add_argument("--slide-table", default="results/data/slide_table_pyramidal.csv")
    p.add_argument("--outdir", default="results/analysis/site_holdout")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.embeddings:
        emb_dirs = [Path(args.embeddings)]
    else:
        emb_dirs = sorted(Path("results/embeddings").iterdir())
        emb_dirs = [d for d in emb_dirs if (d / "embeddings.npy").exists()]

    frames = []
    for emb in emb_dirs:
        print(f"--- {emb.name} ---")
        sub_out = outdir / emb.name
        df = run_one_embedding(emb, Path(args.clinical), Path(args.slide_table), sub_out)
        frames.append(df)
        for _, row in df.iterrows():
            if np.isnan(row["auroc"]):
                print(f"  {row['holdout_site']:20s} — skipped ({row.get('note','')})")
            else:
                print(f"  {row['holdout_site']:20s}  n_test={int(row['n_test_slides']):4d}  "
                      f"prev={row['test_prevalence']:.2f}  "
                      f"AUROC={row['auroc']:.3f}  AUPRC={row['auprc']:.3f}")

    combined = pd.concat(frames, ignore_index=True)[RESULT_COLS]
    combined.to_csv(outdir / "summary.csv", index=False)
    summary_figure(combined, outdir / "summary_heatmap.png")

    print(f"\nSaved: {outdir/'summary.csv'}")
    print(f"Saved: {outdir/'summary_heatmap.png'}")


if __name__ == "__main__":
    main()
