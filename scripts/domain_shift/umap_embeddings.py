"""Visualise slide embeddings with UMAP.

For each embedding directory under results/embeddings, compute a 2D UMAP of
the slides and produce a 2x3 panel:
    (1) coloured by SITE
    (2) coloured by batch/arm (prospective vs retrospective)
    (3) coloured by MSI-H / MSS
    (4) coloured by cmo_msi_score (where available)
    (5) coloured by Wagner P(MSI-H)
    (6) coloured by Wagner prediction correctness

Saved per embedding as
`results/analysis/umap_embeddings/<emb>.png` + a CSV with the 2D coords.

Runs in a few minutes on CPU; threads are capped via the .sh wrapper.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from sklearn.preprocessing import StandardScaler

# Deterministic UMAP per embedding; 60/0.1 is the stable default.
UMAP_KW = dict(n_neighbors=30, min_dist=0.1, metric="cosine", random_state=42)

RESULTS = Path("results/analysis/umap_embeddings")
EMB_ROOT = Path("results/embeddings")
CLINICAL = Path("results/data/clinical_table.csv")
SLIDE_TABLE = Path("results/data/slide_table_pyramidal.csv")
WAGNER = Path("results/analysis/wagner_zeroshot/slide_scores.csv")


def load_metadata() -> pd.DataFrame:
    cl = pd.read_csv(CLINICAL)
    cl = cl[["PATIENT", "isMSIH", "cmo_msi_score",
             "redcap_data_access_group"]].rename(columns={"PATIENT": "patient_id",
                                                          "redcap_data_access_group": "DAG"})
    cl["y"] = (cl["isMSIH"] == "MSI-H").astype(int)
    cl["arm"] = np.where(cl["DAG"].isin(["oauthc"]) & cl["cmo_msi_score"].isna(),
                         "oauthc_retro", cl["DAG"])
    # Use SITE from slide_table (has retrospective_msk / retrospective_oau)
    st = pd.read_csv(SLIDE_TABLE)
    st["slide_id"] = st["FILENAME"].apply(lambda p: Path(p).stem)
    st = st[["slide_id", "PATIENT", "SITE", "stain_location"]].rename(
        columns={"PATIENT": "patient_id"})
    wag = pd.read_csv(WAGNER)[["slide_id", "p_msih", "y"]].rename(
        columns={"p_msih": "wagner_p", "y": "y_slide"})
    return cl, st, wag


def umap_one(emb_dir: Path, cl: pd.DataFrame, st: pd.DataFrame,
             wag: pd.DataFrame, outdir: Path) -> None:
    X = np.load(emb_dir / "embeddings.npy")
    meta = pd.read_csv(emb_dir / "metadata.csv")
    if "patient_id" not in meta.columns and "PATIENT" in meta.columns:
        meta = meta.rename(columns={"PATIENT": "patient_id"})
    df = meta.merge(cl, on="patient_id", how="left")
    df = df.merge(st[["slide_id", "SITE", "stain_location"]], on="slide_id", how="left")
    df = df.merge(wag, on="slide_id", how="left")

    print(f"  {emb_dir.name}: {X.shape[0]} slides × {X.shape[1]} dims")
    Xs = StandardScaler().fit_transform(X)
    reducer = umap.UMAP(**UMAP_KW, n_components=2)
    Z = reducer.fit_transform(Xs)

    df["umap_1"] = Z[:, 0]
    df["umap_2"] = Z[:, 1]
    df[["slide_id", "patient_id", "SITE", "isMSIH", "cmo_msi_score",
        "wagner_p", "umap_1", "umap_2"]].to_csv(
            outdir / f"{emb_dir.name}.csv", index=False)

    # --- Figure ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=150)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")

    # (1) SITE
    ax = axes[0, 0]
    sites = sorted(df["SITE"].dropna().unique())
    palette = plt.cm.tab10(np.linspace(0, 1, len(sites)))
    for c, site in zip(palette, sites):
        s = df[df["SITE"] == site]
        ax.scatter(s["umap_1"], s["umap_2"], s=6, alpha=0.7, c=[c], label=f"{site} ({len(s)})")
    ax.legend(fontsize=7, loc="best")
    ax.set_title("SITE")

    # (2) arm
    ax = axes[0, 1]
    for c, arm in zip(["#4477AA", "#CC3311", "#228833", "#AA3377", "#EE7733", "#888888"],
                      sorted(df["arm"].dropna().unique())):
        s = df[df["arm"] == arm]
        ax.scatter(s["umap_1"], s["umap_2"], s=6, alpha=0.7, c=c, label=f"{arm} ({len(s)})")
    ax.legend(fontsize=7, loc="best")
    ax.set_title("Arm (DAG + retrospective flag)")

    # (3) MSI-H / MSS
    ax = axes[0, 2]
    for c, lbl, mask in [("#CC3311", "MSI-H", df["isMSIH"] == "MSI-H"),
                         ("#4477AA", "MSS",   df["isMSIH"] == "MSS"),
                         ("#CCCCCC", "NA",    df["isMSIH"].isna())]:
        s = df[mask]
        if len(s):
            ax.scatter(s["umap_1"], s["umap_2"], s=6, alpha=0.7, c=c,
                       label=f"{lbl} ({len(s)})")
    ax.legend(fontsize=7, loc="best")
    ax.set_title("MSI status (binary)")

    # (4) cmo_msi_score (log1p colour)
    ax = axes[1, 0]
    has = df["cmo_msi_score"].notna()
    bg = df[~has]
    ax.scatter(bg["umap_1"], bg["umap_2"], s=4, alpha=0.25, c="#BBBBBB",
               label=f"no score ({len(bg)})")
    fg = df[has]
    sc = ax.scatter(fg["umap_1"], fg["umap_2"], s=8, alpha=0.85,
                    c=np.log1p(fg["cmo_msi_score"].astype(float)),
                    cmap="viridis", vmin=0, vmax=np.log1p(40))
    fig.colorbar(sc, ax=ax, shrink=0.7, label="log1p(cmo_msi_score)")
    ax.set_title("cmo_msi_score (prospective only)")

    # (5) Wagner P
    ax = axes[1, 1]
    has = df["wagner_p"].notna()
    fg = df[has]
    sc = ax.scatter(fg["umap_1"], fg["umap_2"], s=8, alpha=0.85,
                    c=fg["wagner_p"], cmap="coolwarm", vmin=0, vmax=1)
    fig.colorbar(sc, ax=ax, shrink=0.7, label="Wagner P(MSI-H)")
    ax.set_title("Wagner probability")

    # (6) Wagner correctness
    ax = axes[1, 2]
    has = df["wagner_p"].notna() & df["y_slide"].notna()
    sub = df[has].copy()
    sub["correct"] = ((sub["wagner_p"] >= 0.5).astype(int) == sub["y_slide"]).astype(int)
    for c, lbl, mask in [("#228833", "correct", sub["correct"] == 1),
                         ("#CC3311", "wrong",   sub["correct"] == 0)]:
        s = sub[mask]
        ax.scatter(s["umap_1"], s["umap_2"], s=6, alpha=0.7, c=c,
                   label=f"{lbl} ({len(s)})")
    ax.legend(fontsize=7, loc="best")
    ax.set_title("Wagner prediction (binary correctness)")

    fig.suptitle(f"UMAP — {emb_dir.name}   ({X.shape[0]} slides, {X.shape[1]} dims)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(outdir / f"{emb_dir.name}.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings", default=None,
                   help="Optional single embedding dir; default all raw (skip _harmony).")
    p.add_argument("--outdir", default=str(RESULTS))
    args = p.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    cl, st, wag = load_metadata()

    if args.embeddings:
        emb_dirs = [Path(args.embeddings)]
    else:
        emb_dirs = sorted(EMB_ROOT.iterdir())
        emb_dirs = [d for d in emb_dirs
                    if (d / "embeddings.npy").exists() and not d.name.endswith("_harmony")]

    for d in emb_dirs:
        umap_one(d, cl, st, wag, outdir)
        print(f"  saved: {outdir/(d.name+'.png')}")

    print(f"\nDone → {outdir}")


if __name__ == "__main__":
    main()
