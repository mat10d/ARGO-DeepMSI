#!/usr/bin/env bash
#SBATCH --job-name=argo-w1-peft
#SBATCH --partition=nvidia-A6000-20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-1
#SBATCH --output=scripts/logs/w1_peft_%A_%a.out
#SBATCH --error=scripts/logs/w1_peft_%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

candidates=(W1-7 W1-9)
candidate="${candidates[${SLURM_ARRAY_TASK_ID}]}"

/lab/barcheese01/mdiberna/miniconda3/envs/argo/bin/python -u \
  scripts/w1_run_peft.py "${candidate}" --device cuda
