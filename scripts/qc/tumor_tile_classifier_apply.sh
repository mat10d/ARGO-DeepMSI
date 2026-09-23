#!/bin/bash
#SBATCH --job-name=mdiberna_argo_c5_1c_apply
#SBATCH --partition=20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=scripts/logs/%x_%j.out
#SBATCH --error=scripts/logs/%x_%j.err

set -euo pipefail
mkdir -p scripts/logs

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export ARGO_NCT_VARIANT="${ARGO_NCT_VARIANT:-nonorm}"

uv run --frozen python -u scripts/qc/tumor_tile_classifier_apply.py "$@"
