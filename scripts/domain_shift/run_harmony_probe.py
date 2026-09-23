"""Run the A0 Harmony CONCH-TITAN linear probe on the Q4 clean cohort.

Writes results/scorers/harmony_probe/{slide_scores.csv, metadata.json, metrics.json}.
CPU-only; run on a compute node (not the head node). Reports OAUTHC first.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from argo_deepmsi.scorers.harmony_probe import HarmonyProbe, _write_metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clean-csv", default="results/data/cohort_clean.csv", type=Path)
    a = p.parse_args()

    cc = pd.read_csv(a.clean_csv)
    clean_ids = {str(s) for s in cc.loc[cc["in_clean_set"] == 1, "slide_id"]}

    scorer = HarmonyProbe()
    slide_df = scorer.compute_batch(pd.DataFrame(), clean_slide_ids=clean_ids, write_outputs=True)
    scorer.write_metadata(scorer.score_path.parent)
    m = _write_metrics(scorer, slide_df)

    print(
        f"harmony_probe: OAUTHC auroc={m['oauthc_auroc']:.4f} "
        f"spec@sens95={m['oauthc_spec_at_sens95']:.3f} spec@sens96={m['oauthc_spec_at_sens96']:.3f} "
        f"(n={m['oauthc_n_patients']} pt)."
    )
    print(
        f"  overall: auroc={m['auroc']:.4f} spec@sens95={m['spec_at_sens95']:.3f} "
        f"npv@sens95={m['npv_at_sens95']:.3f} (n={m['n_patients']} pt)."
    )
    for site, d in sorted(m["by_site"].items()):
        print(f"  {site:<20} n={d['n']:>3} auroc={d['auroc']:.3f}")


if __name__ == "__main__":
    main()
