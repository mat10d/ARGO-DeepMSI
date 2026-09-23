#!/bin/bash
#SBATCH --job-name=argo_ds_image_stats
#SBATCH --output=scripts/logs/ds_image_stats_%A_%a.out
#SBATCH --error=scripts/logs/ds_image_stats_%A_%a.err
#SBATCH --partition=20
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --array=0-7

# Step-1 image-level domain statistics (colour histograms, Macenko stain vectors,
# Inception tile features), sharded 8 ways on CPU. Summarise afterwards with
# `python -m argo_deepmsi.eval.domain_shift image`.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
mkdir -p scripts/logs
uv run --frozen python -u scripts/domain_shift/image_stats.py \
    --shard "${SLURM_ARRAY_TASK_ID}" --n-shards 8 "$@"
