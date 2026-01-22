#!/bin/bash
#SBATCH --job-name=argo-extract
#SBATCH --output=logs/slurm/extract_%A_%a.out
#SBATCH --error=logs/slurm/extract_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-11%4  # 12 models, max 4 concurrent jobs

# ============================================================================
# ARGO-DeepMSI: Multi-Model Feature Extraction
#
# Extracts features from all slides using multiple foundation models.
# Submits as a SLURM array job - one model per array task.
#
# Usage:
#   sbatch scripts/extract_all_models.sh <slide_table>
#
# Example:
#   sbatch scripts/extract_all_models.sh results/data/slide_table.csv
#
# To run specific models only, modify the MODELS array below.
# ============================================================================

set -e

# ============================================================================
# Configuration
# ============================================================================

# All supported models (modify this list as needed)
MODELS=(
    # No authentication required
    "resnet50"
    "ctranspath"
    "plip"
    # Gated models (require HF auth)
    "uni"
    "uni2"
    "virchow"
    "virchow2"
    "conch"
    "gigapath"
    "h-optimus-0"
    "h-optimus-1"
)

# Get current model from array index
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

# Slide table (first argument or default)
SLIDE_TABLE=${1:-"results/data/slide_table.csv"}

# Project directory
PROJECT_DIR="/home/user/ARGO-DeepMSI"

# ============================================================================
# Environment Setup
# ============================================================================

echo "=============================================="
echo "ARGO-DeepMSI Feature Extraction"
echo "=============================================="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Model: ${MODEL}"
echo "Slide Table: ${SLIDE_TABLE}"
echo "=============================================="

# Load environment
cd ${PROJECT_DIR}
source ~/.bashrc

# Activate conda environment
conda activate argo

# Set HuggingFace cache (modify path as needed)
export HF_HOME="${PROJECT_DIR}/.huggingface_cache"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"

# For offline compute nodes
# export HF_HUB_OFFLINE=1

# ============================================================================
# GPU Check
# ============================================================================

echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

python -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# ============================================================================
# Run Extraction
# ============================================================================

echo ""
echo "Starting feature extraction for model: ${MODEL}"
echo "=============================================="

# Create log directory
mkdir -p logs/slurm

# Run extraction
argo extract "${SLIDE_TABLE}" \
    --model "${MODEL}" \
    --device cuda \
    --amp \
    --output "results/features/${MODEL}"

echo ""
echo "=============================================="
echo "Extraction complete for model: ${MODEL}"
echo "Output: results/features/${MODEL}/"
echo "=============================================="
