"""No-regression gate: the comparable primary champion must not drop below the
recorded floor. The floor lives in results/data/cohort_manifest.json
(no_regression_floor); before that manifest exists it defaults to the current
known champion (0.710). Ratchets up: when a confirmed new champion beats the
floor, Q4/downstream tasks update the manifest.

Exit 0 if OK, non-zero (and a message) if the leaderboard's best clean AUROC has
regressed below the floor minus a small tolerance.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

DEFAULT_FLOOR = 0.710
TOL = 0.005  # allow tiny numerical wobble; a real regression is larger
MANIFEST = Path("results/data/cohort_manifest.json")
LEADERBOARD = Path("results/comparison/leaderboard.csv")


def _floor() -> float:
    if MANIFEST.exists():
        try:
            m = json.loads(MANIFEST.read_text())
            return float(m.get("no_regression_floor", DEFAULT_FLOOR))
        except Exception:
            return DEFAULT_FLOOR
    return DEFAULT_FLOOR


def main() -> int:
    if not LEADERBOARD.exists():
        print(f"[no_regression] leaderboard not found at {LEADERBOARD} — nothing to check yet.")
        return 0
    lb = pd.read_csv(LEADERBOARD)
    col = "patient_auroc_clean"
    if col not in lb.columns:
        print(f"[no_regression] '{col}' missing from leaderboard columns {list(lb.columns)}")
        return 1
    eligible = lb[lb["comparable_primary"].astype(bool)] if "comparable_primary" in lb else lb
    if "confirmatory_valid" in eligible:
        eligible = eligible[eligible["confirmatory_valid"].astype(bool)]
    if eligible.empty:
        print("[no_regression] no scorer has comparable primary-cohort coverage")
        return 1
    best = float(eligible[col].max())
    floor = _floor()
    if best < floor - TOL:
        print(
            f"[no_regression] FAIL: best primary AUROC {best:.4f} "
            f"< floor {floor:.4f} (tol {TOL})"
        )
        return 2
    print(f"[no_regression] OK: best primary AUROC {best:.4f} >= floor {floor:.4f} (tol {TOL})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
