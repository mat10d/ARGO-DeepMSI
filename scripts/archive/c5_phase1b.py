"""C5 Phase 1b — UMAP-outlier-based slide filtering (orthogonal to Wagner P).

For each UMAP in results/analysis/umap_embeddings/:
  1. Run HDBSCAN on the 2D coords — label=-1 is "noise" / outlier island.
  2. Characterize each cluster by site, MSI-H prevalence, tile count, Wagner P.
  3. Compute distance from the retrospective_msk centroid (these are curated
     tumor) — flag slides beyond the 95th/99th percentile.
  4. Test filtering: drop outlier-cluster slides, drop far-from-centroid
     slides. Recompute patient AUROC by site × slide-count bin × aggregator.
  5. Report MSS specificity separately from MSI-H sensitivity (Phase 0
     missed this — overall AUROC hides MSS bag-size inflation).

Outputs → results/analysis/c5_phase1b/
  cluster_labels_<emb>.csv           slide_id → cluster_label + distance
  cluster_summary_<emb>.csv          per-cluster stats (n, site%, MSI-H%, ...)
  filter_auroc.csv                   filter × emb × site × n_bin × aggregator
  filter_auroc_by_class.csv          same but split into MSS-specificity vs MSI-H-sensitivity
  umap_clusters_<emb>.png            UMAP coloured by HDBSCAN cluster id
  umap_centroid_dist_<emb>.png       UMAP coloured by distance from MSK centroid
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)

OUTDIR = Path("results/analysis/c5_phase1b")
OUTDIR.mkdir(parents=True, exist_ok=True)
UMAP_DIR = Path("results/analysis/umap_embeddings")

BINS = [0, 1, 3, 6, 1000]
BIN_LABELS = ["1", "2-3", "4-6", "7+"]

# ---------------------------------------------------------------- load ----

ws = pd.read_csv("results/analysis/wagner_zeroshot/slide_scores.csv")
cl = pd.read_csv("results/data/clinical_table.csv")[
    ["PATIENT", "isMSIH", "cmo_msi_score"]
].rename(columns={"PATIENT": "patient_id"})
ws = ws.merge(cl, on="patient_id", how="left")
ws_key = ws[["slide_id", "patient_id", "site", "n_tiles", "y", "p_msih",
             "isMSIH", "cmo_msi_score"]]

print(f"loaded {len(ws)} slides / {ws['patient_id'].nunique()} patients")

# -------------------------------------------------------- cluster + dist -

def _patient_auroc(ws_df: pd.DataFrame, tag: str, drop_mask: pd.Series | None):
    """Return rows of patient-AUROC results under the given drop mask.

    Bias: ONLY the given slides are dropped; patients who lose all slides
    become NaN and are counted separately.
    """
    if drop_mask is not None:
        kept = ws_df.loc[~drop_mask]
    else:
        kept = ws_df
    pat = (kept.groupby("patient_id")
                .agg(p_mean=("p_msih", "mean"),
                     p_max=("p_msih", "max"),
                     y=("y", "first"),
                     site=("site", "first"),
                     n_slides=("slide_id", "count"))
                .reset_index())
    if len(pat) == 0:
        return []
    pat["n_bin"] = pd.cut(pat["n_slides"], bins=BINS, labels=BIN_LABELS)

    rows = []
    for agg_col in ["p_mean", "p_max"]:
        # Overall
        rows.append(_slice_metrics(pat, agg_col, tag, "__overall__", "all"))
        for site, sub in pat.groupby("site"):
            rows.append(_slice_metrics(sub, agg_col, tag, site, "all"))
        for nbin, sub in pat.groupby("n_bin", observed=True):
            rows.append(_slice_metrics(sub, agg_col, tag,
                                       "__overall__", str(nbin)))
        oau = pat[pat["site"] == "OAUTHC"]
        for nbin, sub in oau.groupby("n_bin", observed=True):
            rows.append(_slice_metrics(sub, agg_col, tag, "OAUTHC", str(nbin)))
    return [r for r in rows if r is not None]


def _slice_metrics(sub, agg_col, tag, site, nbin):
    if len(sub) < 3 or sub["y"].nunique() < 2:
        return None
    p = sub[agg_col].to_numpy()
    y = sub["y"].to_numpy()
    return dict(
        filter=tag,
        aggregator=agg_col.replace("p_", ""),
        site=site,
        n_bin=nbin,
        n_patients=len(sub),
        n_msih=int(y.sum()),
        auroc=float(roc_auc_score(y, p)),
        auprc=float(average_precision_score(y, p)),
        brier=float(brier_score_loss(y, p)),
        mean_pred_mss=float(p[y == 0].mean()),
        mean_pred_msih=float(p[y == 1].mean()),
    )


cluster_pal = plt.cm.tab20.colors

all_filter_rows = []
# Reference (no filter) — only compute once since it doesn't depend on embedding
all_filter_rows.extend(_patient_auroc(ws_key, "none", None))

for csv in sorted(UMAP_DIR.glob("*.csv")):
    emb = csv.stem
    df = pd.read_csv(csv).merge(
        ws_key[["slide_id", "p_msih", "n_tiles", "isMSIH", "y"]],
        on="slide_id", how="left",
    )
    df = df.dropna(subset=["umap_1", "umap_2"]).reset_index(drop=True)
    coords = df[["umap_1", "umap_2"]].to_numpy()

    # ---- HDBSCAN ---------------------------------------------------------
    clusterer = HDBSCAN(min_cluster_size=20, min_samples=5)
    labels = clusterer.fit_predict(coords)
    df["cluster"] = labels

    # ---- Distance from retrospective_msk centroid -----------------------
    msk = df[df["SITE"] == "retrospective_msk"]
    if len(msk) > 3:
        msk_centroid = msk[["umap_1", "umap_2"]].mean().to_numpy()
        d = np.linalg.norm(coords - msk_centroid, axis=1)
    else:
        d = np.zeros(len(df))
    df["centroid_dist"] = d

    # thresholds
    p95 = float(np.percentile(d, 95))
    p99 = float(np.percentile(d, 99))
    df["dist_outlier_p95"] = (d >= p95).astype(int)
    df["dist_outlier_p99"] = (d >= p99).astype(int)

    df.to_csv(OUTDIR / f"cluster_labels_{emb}.csv", index=False)

    # ---- Cluster summary ------------------------------------------------
    rows = []
    for cid, sub in df.groupby("cluster"):
        site_pct = (sub["SITE"].value_counts(normalize=True).round(3)
                               .to_dict())
        rows.append(dict(
            cluster=int(cid),
            n_slides=len(sub),
            n_patients=sub["patient_id"].nunique() if "patient_id" in sub else np.nan,
            is_outlier=bool(cid == -1),
            mean_centroid_dist=float(sub["centroid_dist"].mean()),
            mean_wagner_p=float(sub["p_msih"].mean()),
            median_n_tiles=float(sub["n_tiles"].median()),
            msih_prevalence=float(sub["y"].mean()),
            top_site=max(site_pct, key=site_pct.get) if site_pct else "",
            top_site_pct=max(site_pct.values()) if site_pct else np.nan,
        ))
    csum = pd.DataFrame(rows).sort_values("cluster")
    csum.to_csv(OUTDIR / f"cluster_summary_{emb}.csv", index=False)

    print(f"\n== {emb} ==")
    print(f"  HDBSCAN: {(labels == -1).sum()} noise, "
          f"{len(set(labels)) - (1 if -1 in labels else 0)} clusters")
    print(csum.round(3).to_string(index=False))

    # ---- Filter tests ---------------------------------------------------
    # Build drop masks on the ws_key (by slide_id) for each filter rule.
    cluster_outlier_mask = ws_key["slide_id"].isin(
        df.loc[df["cluster"] == -1, "slide_id"])
    dist_p95_mask = ws_key["slide_id"].isin(
        df.loc[df["dist_outlier_p95"] == 1, "slide_id"])
    dist_p99_mask = ws_key["slide_id"].isin(
        df.loc[df["dist_outlier_p99"] == 1, "slide_id"])

    for tag, mask in [(f"{emb}:hdbscan_noise", cluster_outlier_mask),
                      (f"{emb}:msk_dist_p95", dist_p95_mask),
                      (f"{emb}:msk_dist_p99", dist_p99_mask)]:
        n_dropped = int(mask.sum())
        n_msih_dropped = int(ws_key.loc[mask, "y"].sum())
        print(f"  filter={tag}: drop n={n_dropped} slides "
              f"({n_msih_dropped} MSI-H)")
        all_filter_rows.extend(_patient_auroc(ws_key, tag, mask))

    # ---- Plots ----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), dpi=150)
    uniq = sorted(df["cluster"].unique())
    for i, cid in enumerate(uniq):
        sub = df[df["cluster"] == cid]
        col = "lightgrey" if cid == -1 else cluster_pal[i % 20]
        axes[0].scatter(sub["umap_1"], sub["umap_2"], s=10, c=[col],
                        alpha=0.75,
                        label=f"cluster {cid} (n={len(sub)})")
    axes[0].legend(fontsize=7, loc="best")
    axes[0].set_title(f"{emb} — HDBSCAN clusters (noise = grey)")
    axes[0].set_xlabel("UMAP 1"); axes[0].set_ylabel("UMAP 2")

    sc = axes[1].scatter(df["umap_1"], df["umap_2"], c=df["centroid_dist"],
                         cmap="viridis", s=10, alpha=0.85)
    far = df[df["dist_outlier_p95"] == 1]
    axes[1].scatter(far["umap_1"], far["umap_2"], marker="x",
                    c="red", s=36, lw=1, label=f"≥ p95 (n={len(far)})")
    fig.colorbar(sc, ax=axes[1], label="dist from MSK centroid")
    axes[1].set_title(f"{emb} — distance from retrospective_msk centroid")
    axes[1].set_xlabel("UMAP 1"); axes[1].set_ylabel("UMAP 2")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTDIR / f"umap_clusters_{emb}.png", bbox_inches="tight")
    plt.close(fig)

# -------------------------------------------------------------- summary --

fdf = pd.DataFrame(all_filter_rows)
fdf.to_csv(OUTDIR / "filter_auroc.csv", index=False)

print("\n== Headline: overall patient AUROC by filter (mean agg) ==")
ov = fdf[(fdf["site"] == "__overall__") & (fdf["n_bin"] == "all") &
         (fdf["aggregator"] == "mean")]
print(ov[["filter", "n_patients", "auroc", "auprc",
          "brier", "mean_pred_mss", "mean_pred_msih"]]
        .round(3).to_string(index=False))

print("\n== OAUTHC 7+ recovery by filter (mean agg) ==")
o7 = fdf[(fdf["site"] == "OAUTHC") & (fdf["n_bin"] == "7+") &
         (fdf["aggregator"] == "mean")]
print(o7[["filter", "n_patients", "n_msih", "auroc", "mean_pred_mss",
          "mean_pred_msih"]]
        .round(3).to_string(index=False))

print("\n== MSS 4-6 bin — specificity under filter (mean_pred_mss should drop) ==")
m46 = fdf[(fdf["site"] == "__overall__") & (fdf["n_bin"] == "4-6") &
          (fdf["aggregator"] == "mean")]
print(m46[["filter", "n_patients", "mean_pred_mss", "mean_pred_msih", "auroc"]]
        .round(3).to_string(index=False))

print(f"\nDone → {OUTDIR}/")
