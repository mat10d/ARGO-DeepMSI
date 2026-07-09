"""A5 — OAUTHC operating-point calibration on the best A-phase scorer (A0 Harmony).

Reads results/scorers/harmony_probe/slide_scores.csv, aggregates to patient (max/√n),
and prints + saves the OAUTHC global-vs-per-site operating-point table to
results/analysis/per_site_calibration/oauthc_operating_points.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from argo_deepmsi.eval.per_site_calibration import (
    patient_scores_from_slide_csv,
    per_site_operating_points,
)

OUT = Path("results/analysis/per_site_calibration")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scorer-csv", default="results/scorers/harmony_probe/slide_scores.csv", type=Path)
    p.add_argument("--site", default="OAUTHC")
    a = p.parse_args()

    pat = patient_scores_from_slide_csv(a.scorer_csv)
    tbl = per_site_operating_points(pat, site=a.site, target_sens=(0.95, 0.96))
    OUT.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(OUT / "oauthc_operating_points.csv", index=False)

    n_pat = int((pat["site"] == a.site).sum())
    n_pos = int(((pat["site"] == a.site) & (pat["y"] == 1)).sum())
    print(f"A5 per-site calibration on {a.site} (n={n_pat} pt, {n_pos} MSI-H), scorer={a.scorer_csv}")
    print(f"{'target_sens':>11} {'method':>20} {'thr':>7} {'sens':>6} {'spec':>6} {'npv':>6}")
    for _, r in tbl.iterrows():
        print(f"{r['target_sens']:>11} {r['method']:>20} {r['threshold']:>7.3f} "
              f"{r['sensitivity']:>6.3f} {r['specificity']:>6.3f} {r['npv']:>6.3f}")
    print(f"\nsaved -> {OUT / 'oauthc_operating_points.csv'}")


if __name__ == "__main__":
    main()
