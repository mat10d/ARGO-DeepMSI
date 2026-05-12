#!/bin/bash
#SBATCH --job-name=argo_c6_a1
#SBATCH --output=scripts/logs/c6_a1_%j.out
#SBATCH --error=scripts/logs/c6_a1_%j.err
#SBATCH --partition=nvidia-A6000-20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00

# C6 / A1 — Vision-language zero-shot MSI scoring with TITAN's CONCH v1.5
# text encoder. Tile-level + slide-level cosine to MSI-H/MSS prompt prototypes.
#
# Usage: sbatch scripts/c6_a1_vl_zeroshot.sh [--limit 50]

set -euo pipefail

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURM_NODELIST"
echo "Start:   $(date)"

source /lab/barcheese01/mdiberna/miniconda3/etc/profile.d/conda.sh
conda activate argo

cd "${SLURM_SUBMIT_DIR:-/lab/barcheese01/mdiberna/ARGO-DeepMSI}"
mkdir -p scripts/logs

# Source HF_TOKEN from .env so TITAN download works on compute node
set -a; source .env; set +a

python -u scripts/c6_a1_vl_zeroshot.py "$@"

echo "End: $(date)"
