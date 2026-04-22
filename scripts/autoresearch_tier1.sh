#!/bin/bash
#SBATCH --job-name=argo_tier1
#SBATCH --output=scripts/logs/tier1_%j.out
#SBATCH --error=scripts/logs/tier1_%j.err
#SBATCH --partition=20
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2:00:00

# B1 — autoresearch Tier 1 grid.
#   {5 embeddings} × {LR, SVM-rbf, RF, XGBoost} × {raw, PCA(100)} × GroupKFold(5)
# CPU-only. High CPU + high-dim kernel SVM makes 96G comfortable.
#
# Usage: sbatch scripts/autoresearch_tier1.sh

set -euo pipefail

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURM_NODELIST"
echo "Start:   $(date)"

source /lab/barcheese01/mdiberna/miniconda3/etc/profile.d/conda.sh
conda activate argo

cd "${SLURM_SUBMIT_DIR:-/lab/barcheese01/mdiberna/ARGO-DeepMSI}"
mkdir -p scripts/logs

python -u scripts/autoresearch_tier1.py

echo "End: $(date)"
