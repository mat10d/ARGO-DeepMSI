#!/bin/bash
#SBATCH --job-name=argo_aggregate
#SBATCH --output=scripts/logs/aggregate_%A_%a.out
#SBATCH --time=4:00:00
#SBATCH --partition=short
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-9%5

# =============================================================================
# ARGO-DeepMSI: Feature Aggregation
# =============================================================================
# Aggregate patch features to slide-level embeddings.
# Customize MODELS and METHODS arrays below.
#
# Usage:
#   sbatch scripts/aggregate.sh
#
# Update --array=0-N%M where:
#   N = number of combinations - 1
#   M = max concurrent jobs
# =============================================================================

# Models to aggregate (must match extracted models)
MODELS=(
    "plip"
    "ctranspath"
    "phikon"
    "phikonv2"
    "uni2"
    "virchow2"
    "h-optimus-0"
    "gigapath"
    "conch"
    "hibou-b"
)

# Aggregation method (mean, max, median, sum)
# For neural encoders (prism, titan, etc.), see README
METHOD="mean"

# Get model for this array task
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "========================================="
echo "ARGO-DeepMSI: Aggregation"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Model: $MODEL"
echo "Method: $METHOD"
echo "Start time: $(date)"
echo "========================================="

# Load conda environment
source /lab/barcheese01/mdiberna/miniconda3/etc/profile.d/conda.sh
conda activate argo

# Run aggregation
cd /lab/barcheese01/mdiberna/ARGO-DeepMSI
echo ""
echo "Aggregating $MODEL with $METHOD..."
python -m argo_deepmsi.cli aggregate \
    $MODEL \
    --slide-table results/data/slide_table.csv \
    --method $METHOD

echo ""
echo "========================================="
echo "✓ Aggregation complete for $MODEL"
echo "End time: $(date)"
echo "========================================="
