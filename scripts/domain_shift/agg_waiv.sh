#!/bin/bash
#SBATCH --job-name=mdiberna_agg_waiv
#SBATCH --output=scripts/logs/agg_waiv_%j.out
#SBATCH --time=03:00:00
#SBATCH --partition=20
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
uv run --frozen argo aggregate phaet,mascaret --method mean \
  --slide-table results/data/slide_table_pyramidal.csv --device cpu
