"""Run the A3 OAUTHC-adaptive tile-MIL on the Q4 clean cohort.

Trains the base ABMIL ensemble + OAUTHC adaptation and writes
results/scorers/tilemil_oauthc_adapt/{slide_scores.csv, oauthc_kshot_curve.csv,
metadata.json, metrics.json}. CPU-only; run on a compute node. Reports OAUTHC first.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from argo_deepmsi.scorers.tilemil_oauthc_adapt import TileMILOAUTHCAdapt, _write_metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clean-csv", default="results/data/cohort_clean.csv", type=Path)
    a = p.parse_args()

    cc = pd.read_csv(a.clean_csv)
    clean_ids = {str(s) for s in cc.loc[cc["in_clean_set"] == 1, "slide_id"]}

    scorer = TileMILOAUTHCAdapt()
    slide_df = scorer.compute_batch(
        pd.DataFrame(), clean_slide_ids=clean_ids, full_curve=True, write_outputs=True, retrain=True
    )
    scorer.write_metadata(scorer.score_path.parent)
    m = _write_metrics(scorer, slide_df)

    print(
        f"tilemil_oauthc_adapt: OAUTHC auroc={m['oauthc_auroc']:.4f} "
        f"spec@sens95={m['oauthc_spec_at_sens95']:.3f} (n={m['oauthc_n_patients']} pt)"
    )
    print(
        f"  overall: auroc={m['auroc']:.4f} spec@sens95={m['spec_at_sens95']:.3f} "
        f"npv@sens95={m['npv_at_sens95']:.3f} (n={m['n_patients']} pt)"
    )
    print("  OAUTHC K-shot adaptation curve:")
    for pt in m["oauthc_kshot_curve"]:
        print(f"    K={str(pt['K']):>3}  OAUTHC auroc={pt['oauthc_patient_auroc']:.4f}  (draws={pt['n_draws']})")


if __name__ == "__main__":
    main()
