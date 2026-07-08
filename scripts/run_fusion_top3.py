"""Run the F1 stacked-fusion scorer on the Q4 clean cohort.

Writes results/scorers/fusion_top3/{patient_scores.csv, metadata.json, metrics.json}.
CPU-only (LR meta-learner over 3 cached base patient scores).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from argo_deepmsi.scorers.fusion_top3 import FusionTop3, _write_metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clean-csv", default="results/data/cohort_clean.csv", type=Path)
    a = p.parse_args()

    cc = pd.read_csv(a.clean_csv)
    clean_ids = {str(s) for s in cc.loc[cc["in_clean_set"] == 1, "slide_id"]}

    scorer = FusionTop3()
    pat = scorer.compute_batch(pd.DataFrame(), clean_slide_ids=clean_ids, write_outputs=True, retrain=True)
    scorer.write_metadata(scorer.score_path.parent)
    m = _write_metrics(scorer, pat, scorer._last["wide"])

    print(f"fusion_top3: AUROC {m['auroc']:.4f} (champion {m['champion_auroc']:.4f}, "
          f"Δ {m['fusion_minus_champion_auroc']:+.4f})")
    print(f"  components: {m['component_auroc']}")
    print(f"  spec@sens95 {m['spec_at_sens95']:.3f}, spec@sens96 {m['spec_at_sens96']:.3f}, "
          f"NPV@sens95 {m['npv_at_sens95']:.3f}")
    print(f"  MSIntuit target: spec {m['msintuit_target']['specificity']} @ sens {m['msintuit_target']['sensitivity']}")


if __name__ == "__main__":
    main()
