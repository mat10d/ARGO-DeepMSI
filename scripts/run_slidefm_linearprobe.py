"""Run the S1 slide-FM linear probe on the Q4 clean cohort.

Writes results/scorers/slidefm_linearprobe/{slide_scores.csv, metadata.json,
metrics.json, few_shot_curve.csv, embedding_ranking.csv}. CPU-only; run on a
compute node (scripts/run_slidefm_linearprobe.sh) — not the head node.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from argo_deepmsi.scorers.slidefm_linearprobe import SlideFMLinearProbe, _write_metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clean-csv", default="results/data/cohort_clean.csv", type=Path)
    a = p.parse_args()

    cc = pd.read_csv(a.clean_csv)
    clean_ids = {str(s) for s in cc.loc[cc["in_clean_set"] == 1, "slide_id"]}

    scorer = SlideFMLinearProbe()
    slide_df = scorer.compute_batch(
        pd.DataFrame(), clean_slide_ids=clean_ids, full_curve=True, write_outputs=True
    )
    scorer.write_metadata(scorer.score_path.parent)
    m = _write_metrics(scorer, slide_df)

    print(f"slidefm_linearprobe: best={m['best_embedding']} "
          f"auroc={m['auroc']:.4f} spec@sens95={m['spec_at_sens95']:.3f} "
          f"npv@sens95={m['npv_at_sens95']:.3f} (n={m['n_patients']} patients).")
    best_disp = scorer._last["best_display"]
    print(f"few-shot ({best_disp}):")
    for pt in m["few_shot_curve"].get(best_disp, []):
        print(f"  K={str(pt['K']):>3}  patient_auroc={pt['patient_auroc']:.4f}")


if __name__ == "__main__":
    main()
