#!/bin/bash
#SBATCH --job-name=argo_domain_shift
#SBATCH --output=scripts/logs/domain_shift_%x_%A_%a.out
#SBATCH --error=scripts/logs/domain_shift_%x_%A_%a.err
#SBATCH --partition=20
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00

# Step 1-3 domain-shift analyses on existing slide embeddings (CPU).
#   sbatch scripts/domain_shift/domain_shift.sh embed
#   sbatch --array=0-6 scripts/domain_shift/domain_shift.sh dann   # one encoder per task
#   sbatch scripts/domain_shift/domain_shift.sh calibration        # after dann
#   sbatch scripts/domain_shift/domain_shift.sh image              # after image_stats.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
mkdir -p scripts/logs
STEP="$1"; shift
ENCODERS=(ctranspath_mean uni2_mean virchow2_mean conch_v1.5_titan phaet_mean mascaret_mean virchow2_prism)
if [[ "${STEP}" == "dann" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    ENC="${ENCODERS[${SLURM_ARRAY_TASK_ID}]}"
    uv run --frozen python -u -m argo_deepmsi.eval.domain_shift dann --encoders "${ENC}" \
        --out-dir "results/analysis/domain_shift/dann_${ENC}" "$@"
else
    uv run --frozen python -u -m argo_deepmsi.eval.domain_shift "${STEP}" "$@"
fi
