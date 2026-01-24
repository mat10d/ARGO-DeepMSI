#!/bin/bash
#SBATCH --job-name=argo_train
#SBATCH --output=scripts/logs/train_%A_%a.out
#SBATCH --time=2:00:00
#SBATCH --partition=short
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-9%5

# =============================================================================
# ARGO-DeepMSI: Classifier Training
# =============================================================================
# Train classifiers on slide-level embeddings.
# Customize EMBEDDINGS array below to match aggregated models.
#
# Usage:
#   sbatch scripts/train.sh
#
# Update --array=0-N%M where:
#   N = number of embedding types - 1
#   M = max concurrent jobs
# =============================================================================

# Embedding directories to train on (must exist in results/embeddings/)
EMBEDDINGS=(
    "plip_mean"
    "ctranspath_mean"
    "phikon_mean"
    "phikonv2_mean"
    "uni2_mean"
    "virchow2_mean"
    "h-optimus-0_mean"
    "gigapath_mean"
    "conch_mean"
    "hibou-b_mean"
)

# Get embedding for this array task
EMBEDDING=${EMBEDDINGS[$SLURM_ARRAY_TASK_ID]}

echo "========================================="
echo "ARGO-DeepMSI: Training"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Embedding: $EMBEDDING"
echo "Start time: $(date)"
echo "========================================="

# Load conda environment
source /lab/barcheese01/mdiberna/miniconda3/etc/profile.d/conda.sh
conda activate argo

# Run training
cd /lab/barcheese01/mdiberna/ARGO-DeepMSI
echo ""
echo "Training classifiers on $EMBEDDING..."
python -m argo_deepmsi.cli train \
    results/embeddings/$EMBEDDING \
    --clinical results/data/clinical_table.csv

echo ""
echo "========================================="
echo "✓ Training complete for $EMBEDDING"
echo "End time: $(date)"
echo "========================================="
