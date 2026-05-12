"""Step 5 — Harmony batch correction on slide embeddings.

For each embedding directory with an h5ad, regress out SITE using
scanpy.external.pp.harmony_integrate. The corrected embedding is written
to a sibling directory `<emb>_harmony/` in the same
(embeddings.npy + metadata.csv + embeddings.h5ad) format so all
downstream scripts (site_holdout, train, autoresearch) pick it up with
zero edits.

Then re-run leave-one-site-out on both raw and corrected to produce a
before/after comparison.

Usage:
    python scripts/harmony_integrate.py                     # all embedding dirs
    python scripts/harmony_integrate.py --embeddings results/embeddings/virchow2_mean
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

PCA_DIM = 50


def _attach_clinical(adata: ad.AnnData, clinical: pd.DataFrame, slide_table: pd.DataFrame) -> ad.AnnData:
    """Ensure adata.obs has PATIENT, SITE, isMSIH."""
    obs = adata.obs.copy()
    if "slide_id" not in obs.columns:
        obs["slide_id"] = obs.index.astype(str)
    if "patient_id" in obs.columns and "PATIENT" not in obs.columns:
        obs["PATIENT"] = obs["patient_id"]
    if "SITE" not in obs.columns:
        if "site" in obs.columns:
            obs["SITE"] = obs["site"]
        else:
            obs = obs.merge(slide_table[["slide_id", "SITE"]], on="slide_id", how="left")
    if "isMSIH" not in obs.columns:
        obs = obs.merge(clinical[["PATIENT", "isMSIH"]], on="PATIENT", how="left")
    obs = obs.reset_index(drop=True)
    adata.obs = obs
    return adata


def harmony_one(emb_dir: Path, clinical: pd.DataFrame, slide_table: pd.DataFrame,
                out_dir: Path) -> Path:
    h5ad_path = emb_dir / "embeddings.h5ad"
    if h5ad_path.exists():
        adata = sc.read_h5ad(h5ad_path)
    else:
        X = np.load(emb_dir / "embeddings.npy")
        meta = pd.read_csv(emb_dir / "metadata.csv")
        adata = ad.AnnData(X=X.astype(np.float32), obs=meta.copy())
    adata = _attach_clinical(adata, clinical, slide_table)

    # Drop rows lacking SITE (can't harmonise them)
    keep = adata.obs["SITE"].notna().values
    adata = adata[keep].copy()

    # Harmony on the standardised PCA (50 dims) is far faster and more stable
    # than on raw 2560-D foundation-model features. Convergence on raw features
    # was O(hours) with no apparent benefit.
    X_std = StandardScaler().fit_transform(adata.X)
    n_comp = min(PCA_DIM, X_std.shape[0] - 1, X_std.shape[1])
    pca = PCA(n_components=n_comp, random_state=0)
    X_pca = pca.fit_transform(X_std).astype(np.float32)
    adata.obsm["X_pca"] = X_pca

    print(f"  PCA {X_std.shape[1]} → {n_comp}  (explained var {pca.explained_variance_ratio_.sum():.3f})",
          flush=True)
    print(f"  running Harmony on {X_pca.shape} over {adata.obs['SITE'].nunique()} sites…",
          flush=True)

    # Call harmonypy directly — scanpy's wrapper hits a shape bug with the
    # PyTorch backend that flips Z_corr orientation before assignment.
    import harmonypy as hm
    ho = hm.run_harmony(X_pca, adata.obs, vars_use=["SITE"], max_iter_harmony=20)
    Z = np.asarray(ho.Z_corr)
    if Z.shape[0] == X_pca.shape[0]:
        X_corr = Z.astype(np.float32)
    else:
        X_corr = Z.T.astype(np.float32)
    assert X_corr.shape == X_pca.shape, f"harmony shape {X_corr.shape} != {X_pca.shape}"

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "embeddings.npy", X_corr)

    meta_cols = [c for c in ["slide_id", "patient_id", "PATIENT", "SITE", "site",
                             "n_tiles", "zarr_path", "isMSIH"] if c in adata.obs.columns]
    meta_out = adata.obs[meta_cols].copy()
    # Keep legacy column names that downstream scripts expect.
    if "patient_id" not in meta_out.columns and "PATIENT" in meta_out.columns:
        meta_out["patient_id"] = meta_out["PATIENT"]
    if "site" not in meta_out.columns and "SITE" in meta_out.columns:
        meta_out["site"] = meta_out["SITE"]
    cols_final = [c for c in ["slide_id", "patient_id", "site", "n_tiles", "zarr_path"]
                  if c in meta_out.columns]
    meta_out[cols_final].to_csv(out_dir / "metadata.csv", index=False)

    # Write a corrected h5ad with the same obs
    adata_out = ad.AnnData(X=X_corr, obs=adata.obs.reset_index(drop=True))
    adata_out.write_h5ad(out_dir / "embeddings.h5ad")
    return out_dir


def site_holdout(X: np.ndarray, df: pd.DataFrame, holdout: str) -> dict:
    held = set(df.loc[df["SITE"] == holdout, "PATIENT"])
    test_mask = df["PATIENT"].isin(held).values
    train_mask = ~test_mask
    y = df["y"].values
    if len(np.unique(y[train_mask])) < 2 or len(np.unique(y[test_mask])) < 2:
        return dict(holdout=holdout, auroc=float("nan"), auprc=float("nan"),
                    n_test=int(test_mask.sum()))
    sc_ = StandardScaler()
    Xtr = sc_.fit_transform(X[train_mask])
    Xte = sc_.transform(X[test_mask])
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    clf.fit(Xtr, y[train_mask])
    p = clf.predict_proba(Xte)[:, 1]
    pred = (p >= 0.5).astype(int)
    return dict(
        holdout=holdout,
        n_test=int(test_mask.sum()),
        prevalence=float(y[test_mask].mean()),
        auroc=float(roc_auc_score(y[test_mask], p)),
        auprc=float(average_precision_score(y[test_mask], p)),
        balanced_acc=float(balanced_accuracy_score(y[test_mask], pred)),
    )


def run_site_holdout(emb_dir: Path, clinical: pd.DataFrame,
                     slide_table: pd.DataFrame) -> pd.DataFrame:
    X = np.load(emb_dir / "embeddings.npy")
    meta = pd.read_csv(emb_dir / "metadata.csv")
    if len(meta) != len(X):
        raise ValueError(f"{emb_dir.name}: metadata/emb mismatch")
    cl = clinical[["PATIENT", "isMSIH"]].copy()
    cl["y"] = (cl["isMSIH"] == "MSI-H").astype(int)
    st = slide_table[["slide_id", "SITE", "PATIENT"]]
    df = meta.rename(columns={"patient_id": "PATIENT"})
    if "PATIENT" not in df.columns:
        df["PATIENT"] = df["patient_id"] if "patient_id" in df.columns else None
    if "SITE" not in df.columns:
        df = df.rename(columns={"site": "SITE"}) if "site" in df.columns else df
        if "SITE" not in df.columns:
            df = df.merge(st[["slide_id", "SITE"]], on="slide_id", how="left")
    df = df.merge(cl[["PATIENT", "y"]], on="PATIENT", how="inner").reset_index(drop=True)
    keep = df.index[df["y"].notna() & df["SITE"].notna()]
    df = df.loc[keep].reset_index(drop=True)
    X = X[keep]

    rows = [dict(site=s, **site_holdout(X, df, s)) for s in sorted(df["SITE"].unique())]
    return pd.DataFrame(rows)


def compare_plot(df_long: pd.DataFrame, out: Path) -> None:
    piv = df_long.pivot_table(index="embedding", columns=["variant", "holdout"], values="auroc")
    # Simple side-by-side bars per embedding × site
    embs = sorted(df_long["embedding"].unique())
    sites = sorted(df_long["holdout"].unique())
    fig, axes = plt.subplots(len(embs), 1, figsize=(1.2 * len(sites) + 3, 2.2 * len(embs)),
                             dpi=150, sharex=True)
    if len(embs) == 1:
        axes = [axes]
    for ax, emb in zip(axes, embs):
        sub = df_long[df_long["embedding"] == emb]
        x = np.arange(len(sites))
        w = 0.38
        raw = sub[sub["variant"] == "raw"].set_index("holdout")["auroc"].reindex(sites)
        har = sub[sub["variant"] == "harmony"].set_index("holdout")["auroc"].reindex(sites)
        ax.bar(x - w / 2, raw.values, w, color="#777", label="raw")
        ax.bar(x + w / 2, har.values, w, color="#CC3311", label="harmony")
        ax.set_xticks(x)
        ax.set_xticklabels(sites, rotation=35, ha="right")
        ax.set_ylim(0, 1)
        ax.axhline(0.5, ls="--", c="k", lw=0.6)
        ax.set_ylabel("AUROC")
        ax.set_title(emb)
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("Step 5 — Leave-one-site-out AUROC: raw vs Harmony-corrected")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings", default=None)
    p.add_argument("--clinical", default="results/data/clinical_table.csv")
    p.add_argument("--slide-table", default="results/data/slide_table_pyramidal.csv")
    p.add_argument("--outdir", default="results/analysis/harmony")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    clinical = pd.read_csv(args.clinical)
    st = pd.read_csv(args.slide_table)
    st["slide_id"] = st["FILENAME"].apply(lambda fp: Path(fp).stem)

    if args.embeddings:
        emb_dirs = [Path(args.embeddings)]
    else:
        emb_dirs = sorted(Path("results/embeddings").iterdir())
        emb_dirs = [d for d in emb_dirs
                    if (d / "embeddings.npy").exists() and not d.name.endswith("_harmony")]

    long_rows = []
    for emb in emb_dirs:
        name = emb.name
        print(f"\n=== {name} ===")
        # Raw
        raw = run_site_holdout(emb, clinical, st)
        for _, r in raw.iterrows():
            long_rows.append(dict(embedding=name, variant="raw", **r.to_dict()))
        # Harmony
        corrected = Path(str(emb) + "_harmony")
        print(f"  harmony → {corrected.name}")
        harmony_one(emb, clinical, st, corrected)
        har = run_site_holdout(corrected, clinical, st)
        for _, r in har.iterrows():
            long_rows.append(dict(embedding=name, variant="harmony", **r.to_dict()))

        # Inline table
        merged = raw[["site", "auroc"]].rename(columns={"auroc": "raw"}).merge(
            har[["site", "auroc"]].rename(columns={"auroc": "harmony"}), on="site")
        merged["Δ"] = merged["harmony"] - merged["raw"]
        print(merged.to_string(index=False))

    long = pd.DataFrame(long_rows)
    long.to_csv(outdir / "site_holdout_long.csv", index=False)

    summary = (long.groupby(["embedding", "variant"])["auroc"]
                   .mean().unstack("variant"))
    summary["Δ"] = summary["harmony"] - summary["raw"]
    summary.to_csv(outdir / "summary.csv")
    print("\nMean AUROC across sites (raw vs harmony):")
    print(summary.round(3).to_string())

    compare_plot(long, outdir / "site_holdout_compare.png")
    print(f"\nSaved: {outdir/'summary.csv'}")
    print(f"Saved: {outdir/'site_holdout_compare.png'}")


if __name__ == "__main__":
    main()
