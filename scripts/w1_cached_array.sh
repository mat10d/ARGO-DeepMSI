#!/usr/bin/env bash
#SBATCH --job-name=argo-w1-cached
#SBATCH --partition=nvidia-A6000-20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --array=1-6%3
#SBATCH --output=scripts/logs/w1_cached_%A_%a.out
#SBATCH --error=scripts/logs/w1_cached_%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

candidates=(W1-1 W1-2 W1-3 W1-4 W1-5 W1-6)
candidate="${candidates[$((SLURM_ARRAY_TASK_ID - 1))]}"

/lab/barcheese01/mdiberna/miniconda3/envs/argo/bin/python -u \
  scripts/w1_run_cached.py "${candidate}" --device cuda
