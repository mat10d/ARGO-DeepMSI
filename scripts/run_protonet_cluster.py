"""Run the S2 ProtoNet-cluster scorer on the Q4 clean cohort.

Requires scripts/build_protonet_features.py to have written the cluster features.
Writes results/scorers/protonet_cluster/{slide_scores.csv, metadata.json,
metrics.json, few_shot_curve.csv}. CPU-only; run on a compute node.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from argo_deepmsi.scorers.protonet_cluster import ProtoNetCluster, _write_metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clean-csv", default="results/data/cohort_clean.csv", type=Path)
    a = p.parse_args()

    cc = pd.read_csv(a.clean_csv)
    clean_ids = {str(s) for s in cc.loc[cc["in_clean_set"] == 1, "slide_id"]}

    scorer = ProtoNetCluster()
    slide_df = scorer.compute_batch(
        pd.DataFrame(), clean_slide_ids=clean_ids, full_curve=True, write_outputs=True
    )
    scorer.write_metadata(scorer.score_path.parent)
    m = _write_metrics(scorer, slide_df)

    print(f"protonet_cluster: auroc={m['auroc']:.4f} spec@sens95={m['spec_at_sens95']:.3f} "
          f"npv@sens95={m['npv_at_sens95']:.3f} (n={m['n_patients']} patients, "
          f"G={m['n_clusters']}, dim={m['wsi_dim']}).")
    print("few-shot:")
    for pt in m["few_shot_curve"]:
        print(f"  K={str(pt['K']):>3}  patient_auroc={pt['patient_auroc']:.4f}")


if __name__ == "__main__":
    main()
