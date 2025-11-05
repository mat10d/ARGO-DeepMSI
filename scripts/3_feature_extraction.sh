#!/bin/bash
#SBATCH --job-name=features_orchestrator
#SBATCH --partition=20
#SBATCH --output=/lab/barcheese01/mdiberna/ARGO-DeepMSI/logs/feature_extraction/orchestrator_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=72:00:00
#
# Stage 3: Feature Extraction via STAMP (Orchestrator)
#
# This orchestrator submits SLURM array jobs for feature extraction.
# Runs on CPU partition (no GPU needed for orchestration).
#
# Usage:
#   # Extract features for ALL models (default)
#   sbatch scripts/3_feature_extraction.sh
#
#   # Extract features for ONE specific model
#   sbatch scripts/3_feature_extraction.sh ctranspath
#   sbatch scripts/3_feature_extraction.sh virchow2
#
# Features:
# - Default: processes all available models sequentially
# - Optional: process just one model by passing model name as argument
# - Tests model accessibility before extraction
# - Submits worker jobs (3_feature_extraction_worker.sh) with GPU
# - Each model processes all 6 sites in parallel (SLURM array)
# - Logs results and skips inaccessible models
#

set -e  # Exit on error

BASE_DIR="/lab/barcheese01/mdiberna/ARGO-DeepMSI"

# Create log directories
mkdir -p "$BASE_DIR/logs/feature_extraction"
mkdir -p "$BASE_DIR/logs/feature_extraction/workers"

echo "=========================================="
echo "ARGO-DeepMSI: Stage 3 - Feature Extraction Orchestrator"
echo "=========================================="
echo "Start time: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo ""

# Check if specific model was requested
REQUESTED_MODEL=$1

if [ -n "$REQUESTED_MODEL" ]; then
    echo "Mode: Single model extraction"
    echo "Requested model: $REQUESTED_MODEL"
else
    echo "Mode: All models extraction (default)"
fi
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

# List of all available models (prefer newer versions)
ALL_MODELS=(
    # No authentication required
    "ctranspath"
    "plip"
    "dino-bloom"
    "chief-ctranspath"

    # Gated models (require HF authentication)
    "virchow2"        # Prefer over virchow
    "uni2"            # Prefer over uni
    "conch1_5"        # Prefer over conch
    "gigapath"
    "h-optimus-0"
    "h-optimus-1"
    "mstar"
    "musk"
)

# Determine which models to process
if [ -n "$REQUESTED_MODEL" ]; then
    # Single model requested
    MODELS=("$REQUESTED_MODEL")
    echo "Processing single model: $REQUESTED_MODEL"
else
    # All models
    MODELS=("${ALL_MODELS[@]}")
    echo "Processing all ${#MODELS[@]} models:"
    for model in "${MODELS[@]}"; do
        echo "  - $model"
    done
fi
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

        # Submit SLURM array job (worker) and wait for completion
        JOB_OUTPUT=$(sbatch --wait "$BASE_DIR/scripts/3_feature_extraction_worker.sh" "$MODEL")
        JOB_ID=$(echo "$JOB_OUTPUT" | grep -oP 'Submitted batch job \K\d+')

        echo "Submitted job: $JOB_ID"
        echo "Waiting for job $JOB_ID to complete..."

        # Wait for job to finish (sbatch --wait handles this)
        echo "Job $JOB_ID completed for model: $MODEL"

        # Check if job succeeded by looking at SLURM output files
        # Note: This is a simplified check - you may want more robust error checking
        if ls "$BASE_DIR/logs/feature_extraction/workers/extract_${JOB_ID}_*.out" 1> /dev/null 2>&1; then
            echo "✓ Feature extraction completed for $MODEL"
            SUCCESSFUL_MODELS+=("$MODEL")
        else
            echo "✗ Feature extraction may have failed for $MODEL (check logs)"
            FAILED_MODELS+=("$MODEL")
        fi

    elif [ $TEST_EXIT_CODE -eq 1 ]; then
        echo "✗ Model $MODEL is not accessible - SKIPPING"
        SKIPPED_MODELS+=("$MODEL")

    else
        echo "✗ Error testing model $MODEL - SKIPPING"
        SKIPPED_MODELS+=("$MODEL")
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
echo "Worker logs: $BASE_DIR/logs/feature_extraction/workers/"
echo "=========================================="
