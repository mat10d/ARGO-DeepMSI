"""Run the R3 fmMAP probe on the Q4 clean cohort.

Writes results/scorers/fmmap_probe/{slide_scores.csv, metadata.json, metrics.json,
few_shot_curve.csv}. CPU-only (supervised UMAP + LR).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from argo_deepmsi.scorers.fmmap_probe import FmmapProbe, _write_metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clean-csv", default="results/data/cohort_clean.csv", type=Path)
    a = p.parse_args()

    cc = pd.read_csv(a.clean_csv)
    clean_ids = {str(s) for s in cc.loc[cc["in_clean_set"] == 1, "slide_id"]}

    scorer = FmmapProbe()
    slide_df = scorer.compute_batch(
        pd.DataFrame(), clean_slide_ids=clean_ids, full_curve=True, write_outputs=True, retrain=True
    )
    scorer.write_metadata(scorer.score_path.parent)
    m = _write_metrics(scorer, slide_df)

    print(f"fmmap_probe: auroc={m['auroc']:.4f} spec@sens95={m['spec_at_sens95']:.3f} "
          f"npv@sens95={m['npv_at_sens95']:.3f} (n={m['n_patients']} patients).")
    print("per-site AUROC (delta vs champion):")
    for site, d in m["by_site"].items():
        dv = d.get("delta_vs_champion")
        print(f"  {site:<20s} {d['auroc']:.3f}  (Δ {dv:+.3f})" if dv is not None else f"  {site}: {d['auroc']:.3f}")
    print("few-shot:", [(p['K'], round(p['patient_auroc'], 3)) for p in m["few_shot_curve"]])


if __name__ == "__main__":
    main()
