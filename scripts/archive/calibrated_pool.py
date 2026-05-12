"""Bag-size-normalised aggregators for Wagner patient-level scoring.

Follow-up to C4. The naive `max(Wagner P)` aggregator inverts on OAUTHC
7+-slide patients because MSS `max` saturates toward 1 as n grows. Before
committing engineering effort to ABMIL, check whether any *deterministic*
calibration of the max operator rescues the 7+ cohort.

Aggregators evaluated:
  raw_mean, raw_max, raw_median, raw_p75      (baselines, from C4)
  top3_mean                                    (fixed-k; bounded in n)
  top5_mean                                    (fixed-k; bounded in n)
  p95                                          (quantile; smooth version of max)
  max_minus_Emax_MSS                           (center max by expected-under-MSS)
  max_pct_under_MSS                            (percentile of max vs MSS-max-of-n null)
  ecdf_then_max                                (within-patient ECDF, then max rank)

The MSS null is built from the slide-level Wagner P values of every slide
whose true label is MSS. For each bag size n, we bootstrap `max_of_n | MSS`
(1000 draws) and compute its mean and quantile function. That gives a
per-n calibration curve any aggregator can use.

Outputs:
    results/analysis/calibrated_aggregation/
        aggregator_auroc.csv            per aggregator × site × slide-count bin
        mss_null_curve.csv              E[max_n] and quantile fns for n=1..50
        aggregator_comparison.png       bar charts: overall vs per-bin
        oauthc_by_bin.png               OAUTHC-only zoom on the 7+ failure
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

OUTDIR = Path("results/analysis/calibrated_aggregation")
OUTDIR.mkdir(parents=True, exist_ok=True)

N_BOOT = 1000
MAX_N = 60  # cover P_0050 with 41 and P_0058-ish with up to 48
SEED = 42

ws = pd.read_csv("results/analysis/wagner_zeroshot/slide_scores.csv")
cl = pd.read_csv("results/data/clinical_table.csv")[
    ["PATIENT", "isMSIH", "cmo_msi_score", "cmo_msi_status"]
].rename(columns={"PATIENT": "patient_id"})
ws = ws.merge(cl, on="patient_id", how="left")

# -----------------------------------------------------------------------
# 1) Build the MSS null calibration table for max-of-n and various quantiles
# -----------------------------------------------------------------------

rng = np.random.default_rng(SEED)
mss_slide_p = ws.loc[ws["y"] == 0, "p_msih"].to_numpy()
print(f"MSS slide pool for null: n={len(mss_slide_p)}  "
      f"min={mss_slide_p.min():.3f}  median={np.median(mss_slide_p):.3f}  "
      f"max={mss_slide_p.max():.3f}")

null_rows = []
for n in range(1, MAX_N + 1):
    draws = rng.choice(mss_slide_p, size=(N_BOOT, n), replace=True)
    mx = draws.max(axis=1)
    null_rows.append(dict(n=n,
                          Emax=float(mx.mean()),
                          q25=float(np.quantile(mx, 0.25)),
                          q50=float(np.quantile(mx, 0.50)),
                          q75=float(np.quantile(mx, 0.75)),
                          q90=float(np.quantile(mx, 0.90)),
                          q95=float(np.quantile(mx, 0.95))))
null_df = pd.DataFrame(null_rows).set_index("n")
null_df.to_csv(OUTDIR / "mss_null_curve.csv")

def _max_percentile_under_mss_null(p_max: float, n: int) -> float:
    """Given the observed p_max for a bag of n, what quantile is it under
    the MSS-max-of-n null? Higher means more anomalous vs MSS expectation."""
    n_eff = int(np.clip(n, 1, MAX_N))
    draws = rng.choice(mss_slide_p, size=(N_BOOT,) if n_eff == 1 else (N_BOOT, n_eff),
                       replace=True)
    mx = draws if n_eff == 1 else draws.max(axis=1)
    return float(np.mean(p_max >= mx))


# -----------------------------------------------------------------------
# 2) Compute per-patient aggregators
# -----------------------------------------------------------------------

def _within_patient_ecdf_max(series: pd.Series) -> float:
    # Rank-then-normalise within bag; max of the normalised rank vector is
    # always 1.0 — instead we return the max raw value scaled by 1/n so
    # bigger bags get penalised. (Equivalent: Wagner P divided by bag size.)
    return series.max() / np.sqrt(len(series))

def _max_minus_Emax(series: pd.Series) -> float:
    n = len(series)
    return series.max() - null_df.loc[min(n, MAX_N), "Emax"]

def _max_pct(series: pd.Series) -> float:
    return _max_percentile_under_mss_null(series.max(), len(series))

def _topk_mean(series: pd.Series, k: int) -> float:
    arr = series.to_numpy()
    k = min(k, len(arr))
    return float(np.sort(arr)[-k:].mean())

def _q(series: pd.Series, q: float) -> float:
    return float(np.quantile(series, q))

aggregators = {
    "raw_mean":            lambda s: float(s.mean()),
    "raw_max":             lambda s: float(s.max()),
    "raw_median":          lambda s: float(s.median()),
    "raw_p75":             lambda s: _q(s, 0.75),
    "top3_mean":           lambda s: _topk_mean(s, 3),
    "top5_mean":           lambda s: _topk_mean(s, 5),
    "p95":                 lambda s: _q(s, 0.95),
    "max_minus_Emax_MSS":  _max_minus_Emax,
    "max_pct_under_MSS":   _max_pct,
    "max_over_sqrtn":      _within_patient_ecdf_max,
}

rows = []
patient_tab = None
for tag, fn in aggregators.items():
    grp = ws.groupby("patient_id")
    agg = grp["p_msih"].apply(fn).rename(tag).reset_index()
    meta = (grp.agg(y=("y", "first"),
                    site=("site", "first"),
                    n_slides=("slide_id", "count"))
                .reset_index())
    agg = agg.merge(meta, on="patient_id")
    if patient_tab is None:
        patient_tab = agg[["patient_id", "site", "n_slides", "y"]].copy()
    patient_tab[tag] = agg[tag].values

patient_tab["n_bin"] = pd.cut(patient_tab["n_slides"], bins=[0, 1, 3, 6, 1000],
                              labels=["1", "2-3", "4-6", "7+"])
patient_tab.to_csv(OUTDIR / "patient_aggregator_table.csv", index=False)

# -----------------------------------------------------------------------
# 3) AUROC per aggregator × site × slide-count bin
# -----------------------------------------------------------------------

def _safe_auroc(y, p):
    if len(np.unique(y)) < 2: return np.nan
    return float(roc_auc_score(y, p))

groups = [("__overall__", patient_tab)]
for site, sub in patient_tab.groupby("site"):
    groups.append((site, sub))
for n_bin, sub in patient_tab.groupby("n_bin", observed=True):
    groups.append((f"bin_{n_bin}", sub))
# OAUTHC × bin
for n_bin, sub in patient_tab[patient_tab["site"] == "OAUTHC"].groupby("n_bin", observed=True):
    groups.append((f"OAUTHC_bin_{n_bin}", sub))

rows = []
for group_tag, sub in groups:
    for tag in aggregators:
        rows.append(dict(group=group_tag, aggregator=tag,
                         n=len(sub), n_pos=int(sub["y"].sum()),
                         auroc=_safe_auroc(sub["y"].values, sub[tag].values)))
res = pd.DataFrame(rows)
res.to_csv(OUTDIR / "aggregator_auroc.csv", index=False)

# Pretty pivot print
piv = res.pivot_table(index="aggregator", columns="group", values="auroc").round(3)
# Reorder columns
col_order = (["__overall__"]
             + [c for c in piv.columns if c.startswith("bin_")]
             + sorted([c for c in piv.columns if not c.startswith("bin_") and c != "__overall__"
                       and not c.startswith("OAUTHC_bin_")])
             + sorted([c for c in piv.columns if c.startswith("OAUTHC_bin_")]))
col_order = [c for c in col_order if c in piv.columns]
piv = piv[col_order]
agg_order = ["raw_mean", "raw_max", "raw_median", "raw_p75", "top3_mean",
             "top5_mean", "p95", "max_minus_Emax_MSS", "max_pct_under_MSS",
             "max_over_sqrtn"]
piv = piv.loc[[a for a in agg_order if a in piv.index]]

print("\nAUROC by aggregator × group:\n")
print(piv.to_string())

# -----------------------------------------------------------------------
# 4) Plots
# -----------------------------------------------------------------------

# Bar: overall vs per-slide-count bin
focus_groups = ["__overall__", "bin_1", "bin_2-3", "bin_4-6", "bin_7+"]
focus_groups = [g for g in focus_groups if g in piv.columns]
fig, ax = plt.subplots(figsize=(max(9, 0.9 * len(focus_groups) + 6), 5.5), dpi=150)
x = np.arange(len(focus_groups))
w = 0.075
palette = plt.cm.tab20(np.linspace(0, 1, len(agg_order)))
for i, tag in enumerate(agg_order):
    if tag not in piv.index: continue
    ax.bar(x + (i - len(agg_order) / 2) * w, piv.loc[tag, focus_groups].values,
           w, color=palette[i], label=tag)
ax.set_xticks(x); ax.set_xticklabels(focus_groups, rotation=20, ha="right")
ax.axhline(0.5, ls="--", c="grey", lw=0.6)
ax.set_ylim(0, 1)
ax.set_ylabel("Wagner patient AUROC")
ax.set_title("Calibrated aggregators — overall vs slide-count bin")
ax.legend(ncol=3, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.25))
fig.tight_layout()
fig.savefig(OUTDIR / "aggregator_comparison.png", bbox_inches="tight")
plt.close(fig)

# OAUTHC-only bin zoom
focus_groups = [g for g in piv.columns if g.startswith("OAUTHC_bin_")]
if focus_groups:
    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(focus_groups) + 6), 5), dpi=150)
    x = np.arange(len(focus_groups))
    for i, tag in enumerate(agg_order):
        if tag not in piv.index: continue
        vals = piv.loc[tag, focus_groups].values
        ax.bar(x + (i - len(agg_order) / 2) * w, vals, w,
               color=palette[i], label=tag)
    ax.set_xticks(x); ax.set_xticklabels(focus_groups, rotation=20, ha="right")
    ax.axhline(0.5, ls="--", c="grey", lw=0.6)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Wagner patient AUROC")
    ax.set_title("Calibrated aggregators — OAUTHC slide-count bins only")
    ax.legend(ncol=3, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.25))
    fig.tight_layout()
    fig.savefig(OUTDIR / "oauthc_by_bin.png", bbox_inches="tight")
    plt.close(fig)

print(f"\nSaved: {OUTDIR/'aggregator_auroc.csv'}")
print(f"Saved: {OUTDIR/'mss_null_curve.csv'}")
print(f"Saved: {OUTDIR/'aggregator_comparison.png'}")
print(f"Saved: {OUTDIR/'oauthc_by_bin.png'}")
