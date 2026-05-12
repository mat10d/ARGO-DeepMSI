#!/bin/bash
#SBATCH --job-name=argo_c6_d
#SBATCH --output=scripts/logs/c6_d_%j.out
#SBATCH --error=scripts/logs/c6_d_%j.err
#SBATCH --partition=nvidia-A6000-20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00

# C6 / D — NuLite morphological feature extraction + LR training.

set -euo pipefail

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURM_NODELIST"
echo "Start:   $(date)"

source /lab/barcheese01/mdiberna/miniconda3/etc/profile.d/conda.sh
conda activate argo

cd "${SLURM_SUBMIT_DIR:-/lab/barcheese01/mdiberna/ARGO-DeepMSI}"
mkdir -p scripts/logs
set -a; source .env; set +a

python -u scripts/c6_d_morphology.py extract \
    --slide-table results/data/slide_table_pyramidal.csv \
    --tiles-per-slide 100 \
    --batch-size 8 \
    --cell-model nulite \
    --big-tile-px 256 \
    --outdir results/analysis/c6_d_morphology

python -u scripts/c6_d_morphology.py train \
    --features results/analysis/c6_d_morphology/features.csv \
    --clinical results/data/clinical_table.csv \
    --outdir results/analysis/c6_d_morphology

echo "End: $(date)"
