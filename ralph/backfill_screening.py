"""E00 backfill: add the screening operating-point block to every existing
scorer's metrics.json, computed from its own cached slide_scores.csv + the
pathologist QC exclusion. No model inference — pure re-scoring of cached
predictions, so it honours the no-new-data regime trivially.

For each results/scorers/<name>/:
  - read slide_scores.csv (has slide_id, patient_id, site, y, + score columns)
  - read metadata.json for primary_score + resolution
  - clean cohort = slides whose slide_id is NOT in the QC exclusion set
  - aggregate to patient (max/sqrt(n) for slide-resolution; identity for patient)
  - compute screening_block + auroc/auprc + by_site on the CLEAN patient scores
  - inject flat keys (spec_at_sens90/95, npv_at_sens95, auroc, auprc) + by_site
    + a nested `screening_clean` block into metrics.json (idempotent).

Run from repo root in the argo env:  python ralph/backfill_screening.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from argo_deepmsi.eval.cohort import load_clean_exclusion
from argo_deepmsi.eval.metrics import aggregate_to_patient
from argo_deepmsi.eval.screening import screening_block

RESULTS = Path("results/scorers")
COHORT_CSV = Path("results/data/cohort_clean.csv")
SENS_FLOORS = (0.90, 0.95, 0.96, 0.98)


def _auroc(y, s):
    y = np.asarray(y)
    s = np.asarray(s)
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


def _auprc(y, s):
    y = np.asarray(y)
    s = np.asarray(s)
    return float(average_precision_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


def _patient_scores(
    sdf: pd.DataFrame, score_col: str, resolution: str, patient_aggregation: str
) -> pd.DataFrame:
    if resolution == "patient":
        keep = sdf.drop_duplicates("patient_id")
        return keep.rename(columns={score_col: "score"})[["patient_id", "y", "site", "score"]]
    return aggregate_to_patient(sdf, score_col=score_col, agg=patient_aggregation)


def backfill_one(d: Path, excluded: set[str]) -> str:
    ss = d / "slide_scores.csv"
    if not ss.exists():
        ss = d / "patient_scores.csv"
    mj = d / "metrics.json"
    md = d / "metadata.json"
    if not ss.exists():
        return f"{d.name}: SKIP (missing score CSV)"
    sdf = pd.read_csv(ss)
    meta = json.loads(md.read_text()) if md.exists() else {}
    primary = meta.get("primary_score")
    if primary is None or primary not in sdf.columns:
        # Broad priority list across the heterogeneous legacy scorers, then a
        # heuristic: first float column that isn't an id/label/count.
        PRIORITY = ("p_msih", "slide_score", "score", "smooth_mean", "smooth_topk",
                    "msi_mean", "tile_topk", "tile_mean", "raw_topk", "raw_max", "smooth_max")
        primary = next((c for c in PRIORITY if c in sdf.columns), None)
        if primary is None:
            NON_SCORE = {"y", "n_tiles", "n_cells_total", "n_tiles_processed"}
            for c in sdf.columns:
                if c in NON_SCORE or not pd.api.types.is_float_dtype(sdf[c]):
                    continue
                if c.startswith(("n_", "frac_")):
                    continue
                primary = c
                break
    if primary is None:
        return f"{d.name}: SKIP (no usable score column in {list(sdf.columns)[:8]})"
    resolution = meta.get("resolution", "slide")
    patient_aggregation = meta.get("patient_aggregation", "max_sqrtn")
    clean = sdf[~sdf["slide_id"].isin(excluded)].copy() if "slide_id" in sdf.columns else sdf.copy()
    pat = _patient_scores(clean, primary, resolution, patient_aggregation).dropna(subset=["score"])

    block = screening_block(pat["y"].to_numpy(), pat["score"].to_numpy(), SENS_FLOORS)
    by_site = {}
    for site, sub in pat.groupby("site"):
        by_site[str(site)] = {
            "n": int(len(sub)),
            "prevalence": float(sub["y"].mean()),
            "auroc": _auroc(sub["y"], sub["score"]),
            "spec_at_sens95": screening_block(sub["y"].to_numpy(), sub["score"].to_numpy(), (0.95,))["spec_at_sens95"]
            if sub["y"].nunique() == 2 else float("nan"),
        }

    m = json.loads(mj.read_text()) if mj.exists() else {"scorer": d.name}
    m["auroc"] = _auroc(pat["y"], pat["score"])
    m["auprc"] = _auprc(pat["y"], pat["score"])
    m["spec_at_sens90"] = block["spec_at_sens90"]
    m["spec_at_sens95"] = block["spec_at_sens95"]
    m["npv_at_sens95"] = block["npv_at_sens95"]
    m["operating_threshold"] = block["op_sens95"]["threshold"]
    m["by_site"] = by_site
    m["screening_clean"] = block
    m["screening_primary_variant"] = primary
    m["screening_n_patients_clean"] = int(len(pat))
    mj.write_text(json.dumps(m, indent=2))
    return (f"{d.name}: auroc={m['auroc']:.3f} spec@95={m['spec_at_sens95']:.3f} "
            f"npv@95={m['npv_at_sens95']:.3f} (n={len(pat)}, var={primary})")


def main() -> int:
    excluded = load_clean_exclusion(COHORT_CSV)
    print(f"Primary exclusion: {len(excluded)} feature-incomplete slides")
    if not RESULTS.exists():
        print("no results/scorers/ — nothing to backfill")
        return 0
    for d in sorted(RESULTS.glob("*/")):
        try:
            print(" ", backfill_one(d, excluded))
        except Exception as e:
            print(f"  {d.name}: ERROR {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
