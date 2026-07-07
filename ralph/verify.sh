#!/usr/bin/env bash
# The gate every iteration must pass. Exit 0 = "done" is legitimate.
# Self-activates the argo conda env so python/pytest resolve regardless of the
# caller's shell state.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# --- activate argo env (idempotent) ---
CONDA_ROOT="${CONDA_ROOT:-/lab/barcheese01/mdiberna/miniconda3}"
if [ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$CONDA_ROOT/etc/profile.d/conda.sh"
  conda activate argo
fi
echo "python: $(which python)"

echo "=== [1/4] contract + regime tests ==="
# Scope to the harness's own guards + any scorer-contract tests the loop adds.
# Deliberately NOT the whole tests/ dir, to avoid coupling the gate to pre-existing
# heavy/slow repo tests (test_argo_pipeline, test_lazyslide_api, ...).
GATE_TESTS=(tests/test_screening_metrics.py tests/test_no_external_data.py)
[ -f tests/test_scorer_contract.py ] && GATE_TESTS+=(tests/test_scorer_contract.py)
for t in tests/test_scorer_*.py; do [ -f "$t" ] && GATE_TESTS+=("$t"); done
pytest "${GATE_TESTS[@]}" -q -x -p no:cacheprovider

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
