#!/bin/bash
#SBATCH --job-name=all_features
#SBATCH --partition=nvidia-2080ti-20
#SBATCH --output=/lab/barcheese01/mdiberna/ARGO-DeepMSI/logs/feature_extraction_all/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#
# Master script: Extract features using ALL available STAMP models
#
# This script runs feature extraction sequentially across all STAMP models.
# Each model processes 6 sites in parallel (via SLURM array job).
# Models are tested for accessibility before submission.
#
# Usage (as SLURM job - recommended):
#   sbatch scripts/3_feature_extraction_all.sh
#
# Usage (local submission - alternative):
#   bash scripts/3_feature_extraction_all.sh
#
# Features:
# - Prefers newer models (virchow2 over virchow, uni2 over uni, etc.)
# - Tests HF authentication before running gated models
# - Runs models sequentially (each model waits for previous to finish)
# - Each model processes all 6 sites in parallel
# - Logs results and skips inaccessible models
# - Can run as SLURM job (recommended) or local submission script
#

set -e  # Exit on error

BASE_DIR="/lab/barcheese01/mdiberna/ARGO-DeepMSI"
LOG_DIR="$BASE_DIR/logs/feature_extraction_all"
RESULT_LOG="$LOG_DIR/extraction_results_$(date +%Y%m%d_%H%M%S).log"

# Create log directory
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "ARGO-DeepMSI: Feature Extraction (All Models)"
echo "=========================================="
echo "Start time: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Log: $RESULT_LOG"
echo ""

# Load environment
echo "Loading environment..."
source ~/.bashrc

# Load environment variables from .env (for HF_TOKEN)
if [ -f "$BASE_DIR/.env" ]; then
    export $(grep -v '^#' "$BASE_DIR/.env" | xargs)
    echo "✓ Loaded .env file"
fi

# Set environment variables
export HF_HOME="$BASE_DIR/.huggingface_cache"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH

# Create cache directories
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"
echo "✓ Environment variables set"
echo ""

# List of models to process (prefer newer versions)
# Order: no-auth first, then gated models
MODELS=(
    # No authentication required
    "ctranspath"
    "plip"
    "dino-bloom"
    "chief-ctranspath"

    # Gated models (require HF authentication)
    "virchow2"        # Prefer over virchow
    "uni2"            # Prefer over uni
    "conch1_5"        # Prefer over conch (use underscore, STAMP may want dash)
    "gigapath"
    "h-optimus-0"
    "h-optimus-1"
    "mstar"
    "musk"
)

echo "Models to process (${#MODELS[@]} total):"
for model in "${MODELS[@]}"; do
    echo "  - $model"
done
echo ""

# Activate STAMP environment
echo "Activating STAMP environment..."
source "$BASE_DIR/STAMP/.venv/bin/activate"

# Track results
declare -a SUCCESSFUL_MODELS
declare -a FAILED_MODELS
declare -a SKIPPED_MODELS

# Test and run each model
for MODEL in "${MODELS[@]}"; do
    echo "=========================================="
    echo "Processing model: $MODEL"
    echo "Time: $(date)"
    echo "=========================================="

    # Test if model is accessible
    echo "Testing model accessibility..."
    python "$BASE_DIR/scripts/test_model_access.py" "$MODEL"

    TEST_EXIT_CODE=$?

    if [ $TEST_EXIT_CODE -eq 0 ]; then
        echo "✓ Model $MODEL is accessible"
        echo ""
        echo "Submitting SLURM job for $MODEL..."

        # Submit SLURM job and wait for completion
        JOB_OUTPUT=$(sbatch --wait "$BASE_DIR/scripts/3_feature_extraction.sh" "$MODEL")
        JOB_ID=$(echo "$JOB_OUTPUT" | grep -oP 'Submitted batch job \K\d+')

        echo "Submitted job: $JOB_ID"
        echo "Waiting for job $JOB_ID to complete..."

        # Wait for job to finish (sbatch --wait handles this)
        echo "Job $JOB_ID completed for model: $MODEL"

        # Check if job succeeded by looking at SLURM output files
        # Note: This is a simplified check - you may want more robust error checking
        if ls "$BASE_DIR/logs/feature_extraction/stamp_*_${JOB_ID}_*.out" 1> /dev/null 2>&1; then
            echo "✓ Feature extraction completed for $MODEL"
            SUCCESSFUL_MODELS+=("$MODEL")
            echo "[$(date)] SUCCESS: $MODEL (Job $JOB_ID)" >> "$RESULT_LOG"
        else
            echo "✗ Feature extraction may have failed for $MODEL (check logs)"
            FAILED_MODELS+=("$MODEL")
            echo "[$(date)] FAILED: $MODEL (Job $JOB_ID)" >> "$RESULT_LOG"
        fi

    elif [ $TEST_EXIT_CODE -eq 1 ]; then
        echo "✗ Model $MODEL is not accessible - SKIPPING"
        SKIPPED_MODELS+=("$MODEL")
        echo "[$(date)] SKIPPED: $MODEL (not accessible)" >> "$RESULT_LOG"

    else
        echo "✗ Error testing model $MODEL - SKIPPING"
        SKIPPED_MODELS+=("$MODEL")
        echo "[$(date)] SKIPPED: $MODEL (test error)" >> "$RESULT_LOG"
    fi

    echo ""
done

# Final summary
echo "=========================================="
echo "FEATURE EXTRACTION COMPLETE"
echo "=========================================="
echo "End time: $(date)"
echo ""
echo "Summary:"
echo "  Successful: ${#SUCCESSFUL_MODELS[@]}"
for model in "${SUCCESSFUL_MODELS[@]}"; do
    echo "    ✓ $model"
done
echo ""
echo "  Failed: ${#FAILED_MODELS[@]}"
for model in "${FAILED_MODELS[@]}"; do
    echo "    ✗ $model"
done
echo ""
echo "  Skipped: ${#SKIPPED_MODELS[@]}"
for model in "${SKIPPED_MODELS[@]}"; do
    echo "    ⊘ $model"
done
echo ""
echo "Full log: $RESULT_LOG"
echo "=========================================="

# Write final summary to log
{
    echo ""
    echo "=========================================="
    echo "FINAL SUMMARY"
    echo "=========================================="
    echo "Successful (${#SUCCESSFUL_MODELS[@]}): ${SUCCESSFUL_MODELS[*]}"
    echo "Failed (${#FAILED_MODELS[@]}): ${FAILED_MODELS[*]}"
    echo "Skipped (${#SKIPPED_MODELS[@]}): ${SKIPPED_MODELS[*]}"
    echo "=========================================="
} >> "$RESULT_LOG"
