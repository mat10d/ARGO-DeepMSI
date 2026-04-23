"""Per-patient multi-slide aggregation and failure-by-slide-count analysis.

Hypothesis: OAUTHC prospective has many slides per patient. If most slides
from a given patient are MSI-informative and some are not (fat, margin,
normal mucosa, stromal-heavy), the patient-level mean over Wagner
probabilities dilutes the signal. The max (most-suspicious slide) might
be a better aggregator.

Outputs:
    results/analysis/multislide/
        n_slides_per_patient.csv    counts + per-patient summary
        aggregation_comparison.csv  patient AUROC for mean/max/median/p75
        per_site_slide_counts.png   distribution + AUROC vs slide-count tercile
        agg_auroc_by_site.png       aggregator comparison per site
        oauthc_patient_stats.csv    per-OAUTHC-patient slide-level variance
        failure_by_slide_count.csv  slide-level accuracy as n_slides grows
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

OUTDIR = Path("results/analysis/multislide")
OUTDIR.mkdir(parents=True, exist_ok=True)

ws = pd.read_csv("results/analysis/wagner_zeroshot/slide_scores.csv")
cl = pd.read_csv("results/data/clinical_table.csv")[
    ["PATIENT", "isMSIH", "cmo_msi_score", "cmo_msi_status",
     "redcap_data_access_group"]].rename(columns={"PATIENT": "patient_id",
                                                   "redcap_data_access_group": "DAG"})
ws = ws.merge(cl, on="patient_id", how="left")

# ----------------------------------------------------- slide counts ------

counts = (ws.groupby("patient_id").agg(n_slides=("slide_id", "count"),
                                       site=("site", "first"),
                                       y=("y", "first"))
            .reset_index())
counts.to_csv(OUTDIR / "n_slides_per_patient.csv", index=False)

print("Slide-count distribution per site:")
tab = (counts.groupby("site")["n_slides"]
             .describe()[["count", "min", "50%", "max", "mean"]]
             .rename(columns={"count": "n_patients", "50%": "median"}))
print(tab.round(2).to_string())

# ----------------------------------------------------- aggregators -------

def _safe_auroc(sub):
    if sub["y"].nunique() < 2:
        return np.nan
    return roc_auc_score(sub["y"], sub["p"])

rows = []
for tag, fn in [("mean",   lambda g: g.mean()),
                ("max",    lambda g: g.max()),
                ("median", lambda g: g.median()),
                ("p75",    lambda g: g.quantile(.75)),
                ("top3mean", lambda g: g.nlargest(3).mean())]:
    pat = (ws.groupby("patient_id").agg(p=("p_msih", fn),
                                        y=("y", "first"),
                                        site=("site", "first"),
                                        n_slides=("slide_id", "count"),
                                        cmo=("cmo_msi_score", "first"))
             .reset_index())
    # Overall + per-site AUROC
    rows.append(dict(aggregator=tag, site="__overall__",
                     n=len(pat), auroc=_safe_auroc(pat),
                     auprc=float(average_precision_score(pat["y"], pat["p"]))
                           if pat["y"].nunique() == 2 else np.nan))
    for site, sub in pat.groupby("site"):
        rows.append(dict(aggregator=tag, site=site, n=len(sub),
                         auroc=_safe_auroc(sub),
                         auprc=float(average_precision_score(sub["y"], sub["p"]))
                               if sub["y"].nunique() == 2 else np.nan))

agg = pd.DataFrame(rows)
agg.to_csv(OUTDIR / "aggregation_comparison.csv", index=False)
print("\nPatient-level AUROC by aggregator × site:")
piv = agg.pivot(index="site", columns="aggregator", values="auroc").round(3)
print(piv.to_string())

# Plot aggregator comparison
sites_order = ["__overall__", "LASUTH", "LUTH", "OAUTHC", "UITH",
               "retrospective_msk", "retrospective_oau"]
sites_present = [s for s in sites_order if s in piv.index]
aggr_order = ["mean", "max", "median", "p75", "top3mean"]
fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(sites_present) + 2), 5), dpi=150)
x = np.arange(len(sites_present))
w = 0.15
palette = dict(mean="#4477AA", max="#CC3311", median="#228833",
               p75="#AA3377", top3mean="#EE7733")
for i, tag in enumerate(aggr_order):
    vals = [piv.loc[s, tag] if s in piv.index else np.nan for s in sites_present]
    ax.bar(x + (i - 2) * w, vals, w, color=palette[tag], label=tag)
ax.set_xticks(x); ax.set_xticklabels(sites_present, rotation=35, ha="right")
ax.set_ylim(0, 1)
ax.axhline(0.5, ls="--", c="grey", lw=0.7)
ax.set_ylabel("Wagner AUROC (patient level)")
ax.set_title("Per-patient aggregator comparison across sites")
ax.legend(ncol=5, loc="lower center", fontsize=8)
fig.tight_layout()
fig.savefig(OUTDIR / "agg_auroc_by_site.png", bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------- slide-count terciles ---

# Patient AUROC using mean aggregation, stratified by slide-count terciles
pat = (ws.groupby("patient_id").agg(p_mean=("p_msih", "mean"),
                                    p_max=("p_msih", "max"),
                                    p_std=("p_msih", "std"),
                                    y=("y", "first"),
                                    site=("site", "first"),
                                    n_slides=("slide_id", "count"))
          .reset_index())
# Fixed bins rather than qcut — n_slides has many ties (single-slide
# patients at non-OAUTHC sites), so qcut can't partition cleanly.
pat["n_bin"] = pd.cut(pat["n_slides"], bins=[0, 1, 3, 6, 1000],
                      labels=["1", "2-3", "4-6", "7+"])

tc_rows = []
for ter, sub in pat.groupby("n_bin", observed=True):
    if sub["y"].nunique() < 2: continue
    tc_rows.append(dict(n_bin=str(ter),
                        n_patients=len(sub),
                        n_slide_min=int(sub["n_slides"].min()),
                        n_slide_median=float(sub["n_slides"].median()),
                        n_slide_max=int(sub["n_slides"].max()),
                        auroc_mean=float(roc_auc_score(sub["y"], sub["p_mean"])),
                        auroc_max=float(roc_auc_score(sub["y"], sub["p_max"]))))
tc = pd.DataFrame(tc_rows)
print("\nAUROC by slide-count bin (mean vs max aggregator):")
print(tc.to_string(index=False))
tc.to_csv(OUTDIR / "auroc_by_slide_count.csv", index=False)

# Per-site, per-bin (deep dive for OAUTHC)
print("\nOAUTHC slide-count bin AUROC (mean vs max):")
oau = pat[pat["site"] == "OAUTHC"].copy()
if len(oau) >= 9:
    oau["n_bin"] = pd.cut(oau["n_slides"], bins=[0, 1, 3, 6, 1000],
                          labels=["1", "2-3", "4-6", "7+"])
    for ter, sub in oau.groupby("n_bin", observed=True):
        if sub["y"].nunique() < 2:
            print(f"  {ter:5s} n={len(sub):3d} slides={sub['n_slides'].median():.0f}  "
                  f"prev={sub['y'].mean():.2f}  (single-class)")
            continue
        print(f"  {ter:5s} n={len(sub):3d} slides={sub['n_slides'].median():.0f}  "
              f"prev={sub['y'].mean():.2f}  "
              f"AUROC_mean={roc_auc_score(sub['y'], sub['p_mean']):.3f}  "
              f"AUROC_max={roc_auc_score(sub['y'], sub['p_max']):.3f}")

# Slide-count distribution plot + AUROC overlay
fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
# Left: histogram per site
for c, site in zip(plt.cm.tab10(np.linspace(0, 1, counts["site"].nunique())),
                   sorted(counts["site"].dropna().unique())):
    axes[0].hist(counts.loc[counts["site"] == site, "n_slides"],
                 bins=np.arange(0, counts["n_slides"].max() + 2),
                 alpha=0.55, label=f"{site} (n={(counts['site']==site).sum()})",
                 color=c)
axes[0].set_xlabel("n slides per patient"); axes[0].set_ylabel("patients")
axes[0].legend(fontsize=8)
axes[0].set_title("Slide-count distribution per site")

# Right: overall mean vs max AUROC by tercile
axes[1].bar(np.arange(len(tc)) - 0.2, tc["auroc_mean"], 0.4, label="mean",
            color="#4477AA")
axes[1].bar(np.arange(len(tc)) + 0.2, tc["auroc_max"], 0.4, label="max",
            color="#CC3311")
for i, row in tc.iterrows():
    axes[1].text(i, max(row["auroc_mean"], row["auroc_max"]) + 0.02,
                 f"n={row['n_patients']}\n({int(row['n_slide_min'])}–{int(row['n_slide_max'])})",
                 ha="center", fontsize=8)
axes[1].set_xticks(range(len(tc))); axes[1].set_xticklabels(tc["n_bin"])
axes[1].set_ylim(0, 1); axes[1].axhline(0.5, ls="--", c="grey", lw=0.6)
axes[1].set_ylabel("Wagner patient AUROC")
axes[1].set_title("AUROC vs slide-count tercile (overall)")
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUTDIR / "per_site_slide_counts.png", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------ OAUTHC within-patient variance

oau_pat = (ws[ws["site"] == "OAUTHC"].groupby("patient_id")
              .agg(n_slides=("slide_id", "count"),
                   p_min=("p_msih", "min"),
                   p_max=("p_msih", "max"),
                   p_mean=("p_msih", "mean"),
                   p_std=("p_msih", "std"),
                   y=("y", "first"),
                   cmo=("cmo_msi_score", "first"))
              .reset_index())
oau_pat["p_range"] = oau_pat["p_max"] - oau_pat["p_min"]
oau_pat.to_csv(OUTDIR / "oauthc_patient_stats.csv", index=False)

print("\nOAUTHC patient-level within-patient variation (Wagner P):")
print(oau_pat[["n_slides", "p_min", "p_mean", "p_max", "p_range", "p_std"]]
         .describe().round(3)[["n_slides","p_min","p_mean","p_max","p_range","p_std"]]
         .to_string())

print("\nDone → results/analysis/multislide/")
