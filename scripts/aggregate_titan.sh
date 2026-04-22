#!/bin/bash
#SBATCH --job-name=argo_titan_conch
#SBATCH --output=scripts/logs/titan_%j.out
#SBATCH --error=scripts/logs/titan_%j.err
#SBATCH --partition=nvidia-A100-20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00

# TITAN slide encoder on conch_v1.5 patch features.
#
# Usage:
#   sbatch scripts/aggregate_titan.sh

set -euo pipefail

SLIDE_TABLE="results/data/slide_table_pyramidal.csv"
if [ ! -f "$SLIDE_TABLE" ]; then
    SLIDE_TABLE="results/data/slide_table.csv"
fi

echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURM_NODELIST"
echo "Slide table: $SLIDE_TABLE"
echo "Start:       $(date)"

source /lab/barcheese01/mdiberna/miniconda3/etc/profile.d/conda.sh
conda activate argo

cd "${SLURM_SUBMIT_DIR:-/lab/barcheese01/mdiberna/ARGO-DeepMSI}"
mkdir -p scripts/logs

python -m argo_deepmsi.cli aggregate conch_v1.5 \
    --slide-table "$SLIDE_TABLE" \
    --method titan \
    --device cuda

echo "End: $(date)"
