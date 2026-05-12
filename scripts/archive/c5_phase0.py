"""C5 Phase 0 — characterize low-Wagner-P slides before any filtering.

0a. Threshold exploration: how many slides fall below each cutoff, by site,
    by MSI-H prevalence, by tile count, by multi-slide patient.
0b. UMAP overlay: for each embedding, plot slides coloured by Wagner P with
    low-P slides highlighted as x markers.
0c. Patient-level impact: recompute patient-level Wagner score after dropping
    slides below the threshold; report AUROC overall and by site / slide-count
    bin. Critical test: does the 7+ OAUTHC cohort recover?

Outputs → results/analysis/c5_phase0/
    threshold_sweep.csv              — one row per (threshold, site) breakdown
    threshold_sweep_overall.csv      — one row per threshold (pooled)
    low_p_per_patient.csv            — per-patient low-P count by threshold
    patient_auroc_filtered.csv       — AUROC by site × slide-bin × threshold × aggregator
    auroc_delta_7plus.csv            — focused OAUTHC 7+ recovery table
    umap_overlay_<embedding>.png     — UMAP with low-P slides x-marked
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

OUTDIR = Path("results/analysis/c5_phase0")
OUTDIR.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25]
UMAP_DIR = Path("results/analysis/umap_embeddings")

# ---------------------------------------------------------------- load ----

ws = pd.read_csv("results/analysis/wagner_zeroshot/slide_scores.csv")
cl = pd.read_csv("results/data/clinical_table.csv")[
    ["PATIENT", "isMSIH", "cmo_msi_score", "cmo_msi_status"]
].rename(columns={"PATIENT": "patient_id"})
ws = ws.merge(cl, on="patient_id", how="left")

n_total_slides = len(ws)
n_total_patients = ws["patient_id"].nunique()
print(f"loaded {n_total_slides} slides / {n_total_patients} patients")

# ====================================================== 0a: threshold ----

sweep_overall = []
sweep_by_site = []
low_p_per_patient_rows = []

for t in THRESHOLDS:
    below = ws["p_msih"] < t

    sweep_overall.append(
        dict(
            threshold=t,
            n_below=int(below.sum()),
            pct_below=float(below.mean() * 100),
            median_tiles_below=float(ws.loc[below, "n_tiles"].median())
            if below.any() else np.nan,
            median_tiles_above=float(ws.loc[~below, "n_tiles"].median()),
            msih_prev_below=float(ws.loc[below, "y"].mean())
            if below.any() else np.nan,
            msih_prev_above=float(ws.loc[~below, "y"].mean()),
        )
    )

    for site, sub in ws.groupby("site"):
        mask = sub["p_msih"] < t
        sweep_by_site.append(
            dict(
                threshold=t,
                site=site,
                n_slides=len(sub),
                n_below=int(mask.sum()),
                pct_below=float(mask.mean() * 100),
                msih_prev_below=float(sub.loc[mask, "y"].mean())
                if mask.any() else np.nan,
                median_tiles_below=float(sub.loc[mask, "n_tiles"].median())
                if mask.any() else np.nan,
            )
        )

    # Per-patient low-P count
    for pid, sub in ws.groupby("patient_id"):
        low_p_per_patient_rows.append(
            dict(
                patient_id=pid,
                threshold=t,
                n_slides=len(sub),
                n_low=int((sub["p_msih"] < t).sum()),
                n_kept=int((sub["p_msih"] >= t).sum()),
                site=sub["site"].iloc[0],
                y=int(sub["y"].iloc[0]),
            )
        )

sweep_overall_df = pd.DataFrame(sweep_overall)
sweep_by_site_df = pd.DataFrame(sweep_by_site)
low_p_pp = pd.DataFrame(low_p_per_patient_rows)

sweep_overall_df.to_csv(OUTDIR / "threshold_sweep_overall.csv", index=False)
sweep_by_site_df.to_csv(OUTDIR / "threshold_sweep.csv", index=False)
low_p_pp.to_csv(OUTDIR / "low_p_per_patient.csv", index=False)

print("\n== 0a. Threshold sweep (pooled) ==")
print(sweep_overall_df.round(3).to_string(index=False))
print("\n== 0a. Per-site breakdown ==")
print(sweep_by_site_df.round(3).to_string(index=False))

# Multi-slide patients with mixed low-P / high-P
print("\n== 0a. Patients losing all slides at each threshold ==")
for t in THRESHOLDS:
    sub = low_p_pp[low_p_pp["threshold"] == t]
    lose_all = sub[sub["n_kept"] == 0]
    print(f"  t={t}: {len(lose_all)} patients would have 0 slides "
          f"({int(lose_all['y'].sum())} MSI-H, {len(lose_all) - int(lose_all['y'].sum())} MSS)")

# ====================================================== 0c: patient AUROC

# Reference (no filtering) AUROC
def _patient_agg(df: pd.DataFrame, p_col: str = "p_msih") -> pd.DataFrame:
    return (df.groupby("patient_id")
              .agg(p_mean=(p_col, "mean"),
                   p_max=(p_col, "max"),
                   y=("y", "first"),
                   site=("site", "first"),
                   n_slides=("slide_id", "count"))
              .reset_index())


def _auroc(df: pd.DataFrame, col: str) -> float:
    if df["y"].nunique() < 2 or df["p_mean"].isna().any():
        return np.nan
    try:
        return float(roc_auc_score(df["y"], df[col]))
    except ValueError:
        return np.nan


BINS = [0, 1, 3, 6, 1000]
BIN_LABELS = ["1", "2-3", "4-6", "7+"]

rows = []

for t in [0.0] + THRESHOLDS:
    kept = ws[ws["p_msih"] >= t] if t > 0 else ws
    pat = _patient_agg(kept)
    # Patients who lose all slides are excluded from AUROC (no score).
    lost = n_total_patients - len(pat)

    for agg_col in ["p_mean", "p_max"]:
        # Overall
        rows.append(
            dict(threshold=t, site="__overall__", n_bin="all",
                 n_patients=len(pat), n_patients_lost=lost,
                 aggregator=agg_col.replace("p_", ""),
                 auroc=_auroc(pat, agg_col))
        )
        # By site
        for site, sub in pat.groupby("site"):
            rows.append(
                dict(threshold=t, site=site, n_bin="all",
                     n_patients=len(sub), n_patients_lost=np.nan,
                     aggregator=agg_col.replace("p_", ""),
                     auroc=_auroc(sub, agg_col))
            )

        # By slide-count bin (pooled)
        pat["n_bin"] = pd.cut(pat["n_slides"], bins=BINS, labels=BIN_LABELS)
        for binlab, sub in pat.groupby("n_bin", observed=True):
            rows.append(
                dict(threshold=t, site="__overall__", n_bin=str(binlab),
                     n_patients=len(sub), n_patients_lost=np.nan,
                     aggregator=agg_col.replace("p_", ""),
                     auroc=_auroc(sub, agg_col))
            )

        # OAUTHC × bin (the critical test)
        oau = pat[pat["site"] == "OAUTHC"].copy()
        if len(oau):
            oau["n_bin"] = pd.cut(oau["n_slides"], bins=BINS, labels=BIN_LABELS)
            for binlab, sub in oau.groupby("n_bin", observed=True):
                rows.append(
                    dict(threshold=t, site="OAUTHC", n_bin=str(binlab),
                         n_patients=len(sub), n_patients_lost=np.nan,
                         aggregator=agg_col.replace("p_", ""),
                         auroc=_auroc(sub, agg_col))
                )

auroc_df = pd.DataFrame(rows)
auroc_df.to_csv(OUTDIR / "patient_auroc_filtered.csv", index=False)

print("\n== 0c. Overall patient AUROC vs threshold (mean / max) ==")
ov = auroc_df[(auroc_df["site"] == "__overall__") & (auroc_df["n_bin"] == "all")]
print(ov.pivot(index="threshold", columns="aggregator", values="auroc")
        .round(3).to_string())

print("\n== 0c. OAUTHC 7+ bin — recovery test ==")
oau7 = auroc_df[(auroc_df["site"] == "OAUTHC") & (auroc_df["n_bin"] == "7+")]
print(oau7.pivot(index="threshold", columns="aggregator", values="auroc")
         .round(3).to_string())
oau7.to_csv(OUTDIR / "auroc_delta_7plus.csv", index=False)

print("\n== 0c. OAUTHC by bin (mean aggregator) ==")
oau_all = auroc_df[(auroc_df["site"] == "OAUTHC") & (auroc_df["aggregator"] == "mean")]
print(oau_all.pivot(index="threshold", columns="n_bin", values="auroc")
        .round(3).to_string())

# ====================================================== 0b: UMAP overlay -

print("\n== 0b. UMAP overlays ==")
THRESHOLD_HIGHLIGHT = 0.10

for csv in sorted(UMAP_DIR.glob("*.csv")):
    emb = csv.stem
    df = pd.read_csv(csv)
    # Merge against the Wagner table on slide_id
    df = df.merge(ws[["slide_id", "p_msih"]], on="slide_id", how="left")
    df = df.dropna(subset=["umap_1", "umap_2", "p_msih"])
    if len(df) == 0:
        print(f"  {emb}: no overlap, skipping")
        continue

    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    sc = ax.scatter(df["umap_1"], df["umap_2"], c=df["p_msih"],
                    cmap="RdBu_r", vmin=0, vmax=1, s=14, alpha=0.85,
                    edgecolors="none")
    low = df[df["p_msih"] < THRESHOLD_HIGHLIGHT]
    ax.scatter(low["umap_1"], low["umap_2"], marker="x",
               c="black", s=42, lw=1.2,
               label=f"Wagner P < {THRESHOLD_HIGHLIGHT}  (n={len(low)})")
    cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label("Wagner P(MSI-H)")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_title(f"{emb} — Wagner P overlay")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTDIR / f"umap_overlay_{emb}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  {emb}: n={len(df)} slides, {len(low)} low-P")

# ====================================================== summary fig ------

fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
for agg_col, c in [("mean", "#4477AA"), ("max", "#CC3311")]:
    sub = auroc_df[(auroc_df["site"] == "__overall__") &
                   (auroc_df["n_bin"] == "all") &
                   (auroc_df["aggregator"] == agg_col)]
    ax.plot(sub["threshold"], sub["auroc"], "-o", label=f"overall {agg_col}",
            color=c, lw=2)
for agg_col, c, ls in [("mean", "#117733", "--"), ("max", "#AA4499", ":")]:
    sub = auroc_df[(auroc_df["site"] == "OAUTHC") &
                   (auroc_df["n_bin"] == "7+") &
                   (auroc_df["aggregator"] == agg_col)]
    ax.plot(sub["threshold"], sub["auroc"], ls + "o",
            label=f"OAUTHC 7+ {agg_col}", color=c, lw=2)
ax.axhline(0.5, ls="--", c="grey", lw=0.7)
ax.set_xlabel("Wagner P threshold (drop slides below)")
ax.set_ylabel("Patient AUROC")
ax.set_title("AUROC recovery vs filtering threshold")
ax.set_ylim(0, 1)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUTDIR / "auroc_vs_threshold.png", bbox_inches="tight")
plt.close(fig)

print(f"\nDone → {OUTDIR}/")
