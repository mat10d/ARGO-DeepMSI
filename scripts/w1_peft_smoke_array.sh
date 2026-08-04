#!/usr/bin/env bash
#SBATCH --job-name=argo-w1-peft-smoke
#SBATCH --partition=nvidia-L40S-20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --output=scripts/logs/w1_peft_smoke_%A_%a.out
#SBATCH --error=scripts/logs/w1_peft_smoke_%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

modes=(bitfit_norm last_stage)
mode="${modes[${SLURM_ARRAY_TASK_ID}]}"

/lab/barcheese01/mdiberna/miniconda3/envs/argo/bin/python -u \
  scripts/w1_peft_smoke.py --mode "${mode}" --device cuda
