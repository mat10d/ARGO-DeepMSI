#!/usr/bin/env bash
# The gate every iteration must pass. Exit 0 = "done" is legitimate.
# Run from repo root inside the argo conda env.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

echo "=== [1/4] contract + regime tests ==="
pytest tests/ -q -x

echo "=== [2/4] metric-block contract (no AUROC-only landings) ==="
python ralph/check_metrics_contract.py

echo "=== [3/4] regenerate leaderboard (never hand-edited) ==="
# The real entrypoint that emits results/comparison/leaderboard.csv.
# Uses cohort_clean.csv once Q-phase builds it; falls back to problem_slides.csv pre-Q.
if [ -f results/data/cohort_clean.csv ]; then
  python -m argo_deepmsi.eval.qc_comparison \
      --qc-csv results/data/problem_slides.csv \
      --slide-table results/data/slide_table_pyramidal.csv \
      --outdir results/comparison
else
  python -m argo_deepmsi.eval.qc_comparison --outdir results/comparison
fi

echo "=== [4/4] no-regression floor ==="
python ralph/no_regression.py

echo "=== verify.sh OK ==="
