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
# Always call tools through the env interpreter — `python -m pytest`, NOT bare
# `pytest` (a stale system pytest can shadow the env's on PATH and crash).
PY="${PY:-python}"
echo "python: $(which "$PY")"

echo "=== [1/4] contract + regime tests ==="
# Scope to the harness's own guards + any scorer-contract tests the loop adds.
# Deliberately NOT the whole tests/ dir, to avoid coupling the gate to pre-existing
# heavy/slow repo tests (test_argo_pipeline, test_lazyslide_api, ...).
GATE_TESTS=(tests/test_screening_metrics.py tests/test_no_external_data.py)
[ -f tests/test_scorer_contract.py ] && GATE_TESTS+=(tests/test_scorer_contract.py)
for t in tests/test_scorer_*.py; do [ -f "$t" ] && GATE_TESTS+=("$t"); done
"$PY" -m pytest "${GATE_TESTS[@]}" -q -x -p no:cacheprovider

echo "=== [2/4] backfill screening block into any metrics.json, then check contract ==="
# Idempotent: adds spec@sens/NPV/by_site to every cached scorer's metrics.json from
# its own slide_scores.csv (no model inference). Keeps legacy scorers gate-compliant.
"$PY" ralph/backfill_screening.py
"$PY" ralph/check_metrics_contract.py

echo "=== [3/4] leaderboard freshness (do NOT regenerate here) ==="
# Regenerating the leaderboard re-runs every scorer's compute_batch (reloads FM
# embeddings) and takes 15+ min — far too expensive to run every iteration. The
# ITERATION that lands/changes a scorer is responsible for running:
#   python -m argo_deepmsi.eval.qc_comparison --outdir results/comparison
# (see AGENT.md). This gate only asserts the leaderboard is present and FRESH —
# i.e. no scorer output is newer than leaderboard.csv. A stale leaderboard means
# the iteration changed scores without regenerating it → fail.
LB=results/comparison/leaderboard.csv
if [ ! -f "$LB" ]; then
  echo "[freshness] FAIL: $LB missing — iteration must regenerate it."
  exit 1
fi
newer="$(find results/scorers -name 'slide_scores.csv' -newer "$LB" 2>/dev/null || true)"
if [ -n "$newer" ]; then
  echo "[freshness] FAIL: scorer outputs newer than leaderboard — regenerate it:"
  echo "$newer" | sed 's/^/    /'
  echo "    python -m argo_deepmsi.eval.qc_comparison --outdir results/comparison"
  exit 1
fi
echo "[freshness] OK: $LB is newer than all scorer outputs."

echo "=== [4/4] no-regression floor ==="
"$PY" ralph/no_regression.py

echo "=== verify.sh OK ==="
