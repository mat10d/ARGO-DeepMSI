"""Run the T1 selective/conformal abstention screener on the Q4 clean cohort.

Writes results/scorers/selective_abstention/{patient_scores.csv, metadata.json,
metrics.json}. CPU-only (wraps the cached champion scores; no training).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from argo_deepmsi.scorers.selective_abstention import SelectiveAbstention, _write_metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clean-csv", default="results/data/cohort_clean.csv", type=Path)
    a = p.parse_args()

    cc = pd.read_csv(a.clean_csv)
    clean_ids = {str(s) for s in cc.loc[cc["in_clean_set"] == 1, "slide_id"]}

    scorer = SelectiveAbstention()
    pat = scorer.compute_batch(pd.DataFrame(), clean_slide_ids=clean_ids, write_outputs=True)
    scorer.write_metadata(scorer.score_path.parent)
    m = _write_metrics(scorer, pat)

    c = m["conformal"]
    print(f"selective_abstention (rule-out): base AUROC {m['auroc']:.3f}, AURC(FOR) {m['aurc']:.3f}")
    print(f"conformal @ target FOR {c['target_risk']}: guaranteed rule-out coverage "
          f"{c['guaranteed_coverage']:.3f}, empirical covered FOR {c['empirical_covered_for']:.3f}")
    print("per-site rule-out coverage (abstention concentrates on weak/OOD sites):")
    for site, d in sorted(m["per_site_coverage"].items(), key=lambda kv: kv[1]["coverage"]):
        print(f"  {site:<20s} coverage {d['coverage']:.3f}  FOR {d['false_omission_rate']:.3f}  (n={d['n']})")


if __name__ == "__main__":
    main()
