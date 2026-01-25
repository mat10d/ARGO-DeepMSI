#!/bin/bash
#SBATCH --job-name=argo_aggregate
#SBATCH --output=scripts/logs/aggregate_%A_%a.out
#SBATCH --time=4:00:00
#SBATCH --partition=short
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-14%5

# =============================================================================
# ARGO-DeepMSI: Feature Aggregation
# =============================================================================
# Aggregate patch features to slide-level embeddings using simple pooling.
# Edit MODELS array to match your extracted models.
#
# For neural slide encoders (prism, titan, etc.), use the CLI directly:
#   argo aggregate virchow2 --method prism
#   argo aggregate conch_v1.5 --method titan
#
# Usage:
#   sbatch scripts/aggregate.sh
#
# Update --array=0-N%M where:
#   N = number of models - 1
#   M = max concurrent jobs
# =============================================================================

# Models to aggregate (must match extracted models from extract.sh)
MODELS=(
    # ===== Recommended Gated Models (HuggingFace auth required) =====
    "uni2"              # UNI v2 (1024D) - latest version
    "virchow2"          # Virchow v2 (1280D) - latest version
    "conch_v1.5"        # CONCH v1.5 (512D) - latest version
    "h-optimus-1"       # H-Optimus 1 (768D) - newer version
    "gigapath"          # GigaPath (1536D)
    "hibou-b"           # Hibou-B (768D)
    "musk"              # MUSK (1024D)

    # ===== Non-Gated Models (no auth required) =====
    "chief"             # CHIEF (768D)
    "plip"              # PLIP vision-language (512D)
    "ctranspath"        # CTransPath (768D)
    "phikonv2"          # Phikon v2 (768D)

    # ===== Older Versions (superseded, keep commented) =====
    # "uni"             # UNI v1 (1024D) - use uni2 instead
    # "virchow"         # Virchow v1 (1280D) - use virchow2 instead
    # "conch"           # CONCH v1 (512D) - use conch_v1.5 instead

    # ===== Model Variants (different sizes) =====
    # "h0-mini"         # H-Optimus 0 Mini (384D) - smaller/faster
    # "hibou-l"         # Hibou-L (1024D) - larger variant

    # ===== Less Common Models =====
    # "gpfm"            # GPFM (768D)
    # "path_orchestra"  # PathOrchestra (768D)
    # "midnight"        # Midnight (768D)
)

# Aggregation method
# Options: mean, max, median, sum
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
