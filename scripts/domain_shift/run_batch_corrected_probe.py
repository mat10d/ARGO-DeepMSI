"""Run the A1 batch-correction sweep on the Q4 clean cohort.

Writes results/scorers/batch_corrected_probe/{slide_scores.csv, sweep.csv,
metadata.json, metrics.json}. CPU-only; run on a compute node. Reports OAUTHC first
and prints the per-method batch-sep vs MSI-AUROC trade table.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from argo_deepmsi.scorers.batch_corrected_probe import BatchCorrectedProbe, _write_metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clean-csv", default="results/data/cohort_clean.csv", type=Path)
    a = p.parse_args()

    cc = pd.read_csv(a.clean_csv)
    clean_ids = {str(s) for s in cc.loc[cc["in_clean_set"] == 1, "slide_id"]}

    scorer = BatchCorrectedProbe()
    slide_df = scorer.compute_batch(pd.DataFrame(), clean_slide_ids=clean_ids, write_outputs=True)
    scorer.write_metadata(scorer.score_path.parent)
    m = _write_metrics(scorer, slide_df)

    print(f"batch_corrected_probe: winner={m['best_method']}")
    print("  method                      oauthc_auroc  overall_auroc  residual_batch_auroc")
    for r in m["sweep"]:
        print(
            f"  {r['method']:<26} {r['oauthc_auroc']:.4f}        {r['overall_auroc']:.4f}"
            f"         {r['residual_batch_auroc']:.4f}"
        )
    print(
        f"  WINNER OAUTHC: auroc={m['oauthc_auroc']:.4f} spec@sens95={m['oauthc_spec_at_sens95']:.3f} "
        f"spec@sens96={m['oauthc_spec_at_sens96']:.3f} (n={m['oauthc_n_patients']} pt)"
    )
    print(
        f"  overall: auroc={m['auroc']:.4f} spec@sens95={m['spec_at_sens95']:.3f} "
        f"npv@sens95={m['npv_at_sens95']:.3f} (n={m['n_patients']} pt)"
    )


if __name__ == "__main__":
    main()
