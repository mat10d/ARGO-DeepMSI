"""Run the S5 Deep-Sets patient-bag aggregator on the Q4 clean cohort.

Writes results/scorers/setencoder_agg/{patient_scores.csv, metadata.json, metrics.json,
few_shot_curve.csv}. CPU-only (tiny patient bags).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from argo_deepmsi.scorers.setencoder_agg import SetEncoderAgg, _write_metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clean-csv", default="results/data/cohort_clean.csv", type=Path)
    a = p.parse_args()

    cc = pd.read_csv(a.clean_csv)
    clean_ids = {str(s) for s in cc.loc[cc["in_clean_set"] == 1, "slide_id"]}

    scorer = SetEncoderAgg()
    pat_df = scorer.compute_batch(
        pd.DataFrame(), clean_slide_ids=clean_ids, full_curve=True, write_outputs=True, retrain=True
    )
    scorer.write_metadata(scorer.score_path.parent)
    m = _write_metrics(scorer, pat_df)

    print(f"setencoder_agg: auroc={m['auroc']:.4f} spec@sens95={m['spec_at_sens95']:.3f} "
          f"npv@sens95={m['npv_at_sens95']:.3f} (n={m['n_patients']} patients).")
    print("few-shot:")
    for pt in m["few_shot_curve"]:
        print(f"  K={str(pt['K']):>3}  patient_auroc={pt['patient_auroc']:.4f}")


if __name__ == "__main__":
    main()
