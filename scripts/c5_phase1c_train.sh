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

eval "$(conda shell.bash hook)"
conda activate argo
cd /lab/barcheese01/mdiberna/ARGO-DeepMSI

export HF_HOME="${HF_HOME:-/lab/barcheese01/mdiberna/.huggingface_cache}"
export OMP_NUM_THREADS=4
# Variant: nonorm (default, raw H&E — matches our cohort) or norm (Macenko)
export ARGO_NCT_VARIANT="${ARGO_NCT_VARIANT:-nonorm}"

python -u scripts/c5_phase1c_train.py "$@"
