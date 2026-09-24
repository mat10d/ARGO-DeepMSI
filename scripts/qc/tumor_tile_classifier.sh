#!/bin/bash
#SBATCH --job-name=mdiberna_argo_c5_1c_train
#SBATCH --partition=nvidia-A6000-20
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=scripts/logs/%x_%j.out
#SBATCH --error=scripts/logs/%x_%j.err

set -euo pipefail
mkdir -p scripts/logs

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export HF_HOME="${HF_HOME:-$PWD/.huggingface_cache}"
export OMP_NUM_THREADS=4
# Variant: nonorm (default, raw H&E — matches our cohort) or norm (Macenko)
export ARGO_NCT_VARIANT="${ARGO_NCT_VARIANT:-nonorm}"

uv run --frozen --project envs/lazyslide python -u scripts/qc/tumor_tile_classifier.py "$@"
