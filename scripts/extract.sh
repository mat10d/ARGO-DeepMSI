#!/bin/bash
#SBATCH --job-name=argo_extract
#SBATCH --output=scripts/logs/extract_%A_%a.out
#SBATCH --time=48:00:00
#SBATCH --partition=nvidia-A6000-20
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --array=0-9%3

# =============================================================================
# ARGO-DeepMSI: Feature Extraction
# =============================================================================
# Customize the MODELS array below to extract features from specific models.
# Gated models require HF_TOKEN in .env file.
#
# Usage:
#   sbatch scripts/extract.sh
#
# Update --array=0-N%M where:
#   N = number of models - 1
#   M = max concurrent jobs
# =============================================================================

# Models to extract (edit this list as needed)
MODELS=(
    # Non-gated models (no HuggingFace auth required)
    "plip"
    "ctranspath"
    "phikon"
    "phikonv2"

    # Gated models (require HF_TOKEN in .env)
    "uni2"
    "virchow2"
    "h-optimus-0"
    "gigapath"
    "conch"
    "hibou-b"
)

# Get model for this array task
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "========================================="
echo "ARGO-DeepMSI: Feature Extraction"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Model: $MODEL"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "========================================="

# Load conda environment
source /lab/barcheese01/mdiberna/miniconda3/etc/profile.d/conda.sh
conda activate argo

# Set HuggingFace cache
export HF_HOME=/lab/barcheese01/mdiberna/ARGO-DeepMSI/.huggingface_cache

# Login to HuggingFace (only if HF_TOKEN is set in .env)
cd /lab/barcheese01/mdiberna/ARGO-DeepMSI
python -c "
from dotenv import load_dotenv
import os
from huggingface_hub import login
load_dotenv('.env')
token = os.getenv('HF_TOKEN')
if token:
    login(token=token, add_to_git_credential=False)
    print('✓ Logged in to HuggingFace')
else:
    print('⚠ No HF_TOKEN found (OK for non-gated models)')
" 2>/dev/null

# Run extraction
echo ""
echo "Extracting features with $MODEL..."
python -m argo_deepmsi.cli extract \
    results/data/slide_table.csv \
    --model $MODEL \
    --device cuda \
    --amp

echo ""
echo "========================================="
echo "✓ Extraction complete for $MODEL"
echo "End time: $(date)"
echo "========================================="
