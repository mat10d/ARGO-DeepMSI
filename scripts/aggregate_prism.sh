#!/bin/bash
#SBATCH --job-name=argo_prism_virchow2
#SBATCH --output=scripts/logs/prism_%j.out
#SBATCH --error=scripts/logs/prism_%j.err
#SBATCH --partition=nvidia-A6000-20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00

# PRISM slide encoder on virchow2 patch features.
#
# Usage:
#   sbatch scripts/aggregate_prism.sh

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

python -m argo_deepmsi.cli aggregate virchow2 \
    --slide-table "$SLIDE_TABLE" \
    --method prism \
    --device cuda

echo "End: $(date)"
