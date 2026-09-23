"""C5 Phase 1c (step 2) — apply trained tissue classifier to all slide
zarr tile features and test tumor-fraction filtering.

For each slide with a `ctranspath_tiles` table in its zarr:
    1. Load the per-tile CTransPath feature matrix X (n_tiles × 768).
    2. Predict 9-class tissue labels (per-tile argmax of LR head).
    3. Compute slide-level summaries:
         - tumor_fraction (TUM predictions / total)
         - mean P(TUM)
         - class composition (fraction per class)
    4. Append to a per-slide table.

Then test filtering at several tumor_fraction thresholds against the same
patient-level AUROC grid as Phase 0/1b.

Outputs → results/analysis/c5_phase1c/
    per_slide_tissue_composition.csv   one row per slide
    tumor_fraction_histogram.png       overall + by-site distribution
    filter_auroc_tumor_fraction.csv    filter × site × n_bin × aggregator
"""

from __future__ import annotations

import logging
from pathlib import Path

import anndata as ad
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("c5_1c_apply")

import os

OUTDIR = Path("results/analysis/c5_phase1c")
OUTDIR.mkdir(parents=True, exist_ok=True)

_VARIANT = os.environ.get("ARGO_NCT_VARIANT", "nonorm").lower()
HEAD_PATH = OUTDIR / f"tissue_head_ctranspath_{_VARIANT}.joblib"
CACHE_NAME = f"per_slide_tissue_composition_{_VARIANT}.csv"
FILTER_CSV = f"filter_auroc_tumor_fraction_{_VARIANT}.csv"
HIST_PNG = f"tumor_fraction_histogram_{_VARIANT}.png"
MERGED_CSV = f"wagner_with_tumor_fraction_{_VARIANT}.csv"

BINS = [0, 1, 3, 6, 1000]
BIN_LABELS = ["1", "2-3", "4-6", "7+"]


def _load_X_from_zarr(zarr_path: Path) -> np.ndarray | None:
    """Read the ctranspath_tiles AnnData from a zarr store and return X."""
    try:
        adata = ad.read_zarr(str(zarr_path / "tables" / "ctranspath_tiles"))
    except Exception as e:
        log.debug(f"anndata read failed for {zarr_path.name}: {e}")
        # Fallback — direct zarr read
        try:
            z = zarr.open_group(str(zarr_path / "tables" / "ctranspath_tiles"),
                                mode="r")
            X = np.asarray(z["X"])
            return X
        except Exception as e2:
            log.warning(f"zarr read also failed for {zarr_path.name}: {e2}")
            return None
    return np.asarray(adata.X)


def compose_per_slide_table(slide_tbl: pd.DataFrame, clf, labels: list[str]):
    tum_id = labels.index("TUM")
    rows = []
    for _, row in tqdm(slide_tbl.iterrows(), total=len(slide_tbl)):
        svs_path = Path(row["FILENAME"])
        zarr_path = svs_path.with_suffix(".zarr")
        if not zarr_path.exists():
            rows.append(dict(slide_id=svs_path.stem, status="no_zarr"))
            continue
        X = _load_X_from_zarr(zarr_path)
        if X is None or X.size == 0:
            rows.append(dict(slide_id=svs_path.stem, status="no_features"))
            continue
        # Predict
        P = clf.predict_proba(X)  # (n_tiles, 9)
        yhat = P.argmax(axis=1)
        n = len(yhat)
        rec = dict(
            slide_id=svs_path.stem,
            n_tiles_used=int(n),
            tumor_fraction=float((yhat == tum_id).mean()),
            mean_p_tum=float(P[:, tum_id].mean()),
            status="ok",
        )
        for i, lab in enumerate(labels):
            rec[f"frac_{lab}"] = float((yhat == i).mean())
            rec[f"meanP_{lab}"] = float(P[:, i].mean())
        rows.append(rec)
    return pd.DataFrame(rows)


def _slice_metrics(sub, agg_col, tag, site, nbin):
    if len(sub) < 3 or sub["y"].nunique() < 2:
        return None
    p = sub[agg_col].to_numpy()
    y = sub["y"].to_numpy()
    return dict(
        filter=tag, aggregator=agg_col.replace("p_", ""),
        site=site, n_bin=nbin,
        n_patients=len(sub), n_msih=int(y.sum()),
        auroc=float(roc_auc_score(y, p)),
        auprc=float(average_precision_score(y, p)),
        brier=float(brier_score_loss(y, p)),
        mean_pred_mss=float(p[y == 0].mean()),
        mean_pred_msih=float(p[y == 1].mean()),
    )


def _patient_auroc(ws: pd.DataFrame, tag: str, drop_mask: pd.Series | None):
    kept = ws.loc[~drop_mask] if drop_mask is not None else ws
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
    for agg in ["p_mean", "p_max"]:
        rows.append(_slice_metrics(pat, agg, tag, "__overall__", "all"))
        for site, sub in pat.groupby("site"):
            rows.append(_slice_metrics(sub, agg, tag, site, "all"))
        for nb, sub in pat.groupby("n_bin", observed=True):
            rows.append(_slice_metrics(sub, agg, tag, "__overall__", str(nb)))
        oau = pat[pat["site"] == "OAUTHC"]
        for nb, sub in oau.groupby("n_bin", observed=True):
            rows.append(_slice_metrics(sub, agg, tag, "OAUTHC", str(nb)))
    return [r for r in rows if r is not None]


def main():
    head = joblib.load(HEAD_PATH)
    clf, labels = head["clf"], head["labels"]
    log.info(f"loaded head: {clf} | labels={labels}")

    slide_tbl = pd.read_csv("results/data/slide_table_pyramidal.csv")
    log.info(f"slide_tbl: {len(slide_tbl)} rows")

    cache = OUTDIR / CACHE_NAME
    if cache.exists():
        per_slide = pd.read_csv(cache)
        log.info(f"loaded cached per-slide table: {len(per_slide)}")
    else:
        per_slide = compose_per_slide_table(slide_tbl, clf, labels)
        per_slide.to_csv(cache, index=False)

    ok = per_slide[per_slide["status"] == "ok"].copy()
    log.info(f"valid slides: {len(ok)} / {len(per_slide)}")

    # ---- distribution plot ------------------------------------------------
    ws = pd.read_csv("results/analysis/wagner_zeroshot/slide_scores.csv")
    cl = pd.read_csv("results/data/clinical_table.csv")[
        ["PATIENT", "isMSIH", "cmo_msi_score"]].rename(
            columns={"PATIENT": "patient_id"})
    ws = ws.merge(cl, on="patient_id", how="left")
    ws = ws.merge(ok[["slide_id", "tumor_fraction", "mean_p_tum",
                      "frac_ADI", "frac_NORM", "frac_STR", "frac_DEB"]],
                  on="slide_id", how="left")
    ws.to_csv(OUTDIR / MERGED_CSV, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
    sites = sorted(ws["site"].dropna().unique())
    for site in sites:
        sub = ws[ws["site"] == site]["tumor_fraction"].dropna()
        axes[0].hist(sub, bins=30, alpha=0.5, label=f"{site} (n={len(sub)})",
                     density=True)
    axes[0].set_xlabel("tumor_fraction (argmax-TUM / tiles)")
    axes[0].set_ylabel("density")
    axes[0].set_title("Tumor fraction per slide by site")
    axes[0].legend(fontsize=8)

    axes[1].scatter(ws["tumor_fraction"], ws["p_msih"],
                    c=ws["y"], cmap="coolwarm", s=10, alpha=0.7)
    axes[1].set_xlabel("tumor_fraction")
    axes[1].set_ylabel("Wagner P")
    axes[1].set_title("Wagner P vs tumor_fraction (MSI-H coloured)")
    fig.tight_layout()
    fig.savefig(OUTDIR / HIST_PNG, bbox_inches="tight")
    plt.close(fig)

    # ---- filter tests ----------------------------------------------------
    ws_key = ws[["slide_id", "patient_id", "site", "y", "p_msih",
                 "tumor_fraction"]].dropna(subset=["tumor_fraction"])
    rows = []
    rows.extend(_patient_auroc(ws_key, "none", None))
    for thr in [0.1, 0.2, 0.3, 0.4, 0.5]:
        mask = ws_key["tumor_fraction"] < thr
        tag = f"tumor_frac<{thr:.1f}"
        log.info(f"{tag}: drop n={int(mask.sum())} slides "
                 f"({int(ws_key.loc[mask, 'y'].sum())} MSI-H)")
        rows.extend(_patient_auroc(ws_key, tag, mask))

    fdf = pd.DataFrame(rows)
    fdf.to_csv(OUTDIR / FILTER_CSV, index=False)

    print("\n== Headline: overall patient AUROC by tumor_frac filter (mean agg) ==")
    ov = fdf[(fdf["site"] == "__overall__") & (fdf["n_bin"] == "all") &
             (fdf["aggregator"] == "mean")]
    print(ov[["filter", "n_patients", "auroc", "auprc",
              "brier", "mean_pred_mss", "mean_pred_msih"]]
            .round(3).to_string(index=False))

    print("\n== OAUTHC 7+ recovery by tumor_frac filter (mean agg) ==")
    o7 = fdf[(fdf["site"] == "OAUTHC") & (fdf["n_bin"] == "7+") &
             (fdf["aggregator"] == "mean")]
    print(o7[["filter", "n_patients", "n_msih", "auroc", "mean_pred_mss",
              "mean_pred_msih"]]
            .round(3).to_string(index=False))

    print("\n== MSS 4-6 bin — specificity under tumor_frac filter (mean agg) ==")
    m46 = fdf[(fdf["site"] == "__overall__") & (fdf["n_bin"] == "4-6") &
              (fdf["aggregator"] == "mean")]
    print(m46[["filter", "n_patients", "mean_pred_mss", "mean_pred_msih", "auroc"]]
            .round(3).to_string(index=False))

    log.info(f"done → {OUTDIR}/")


if __name__ == "__main__":
    main()
