#!/bin/bash
#SBATCH --job-name=argo_c6_a2
#SBATCH --output=scripts/logs/c6_a2_%j.out
#SBATCH --error=scripts/logs/c6_a2_%j.err
#SBATCH --partition=nvidia-A6000-20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3:00:00

# C6 / A2 — Histo-TransCLIP transductive smoothing of A1 VL tile scores.

set -euo pipefail

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURM_NODELIST"
echo "Start:   $(date)"

source /lab/barcheese01/mdiberna/miniconda3/etc/profile.d/conda.sh
conda activate argo

cd "${SLURM_SUBMIT_DIR:-/lab/barcheese01/mdiberna/ARGO-DeepMSI}"
mkdir -p scripts/logs
set -a; source .env; set +a

# Make c6_a1 module importable
export PYTHONPATH="$(pwd)/scripts:${PYTHONPATH:-}"

python -u scripts/c6_a2_transductive.py "$@"

echo "End: $(date)"
