#!/bin/bash
#SBATCH --job-name=argo_extract
#SBATCH --output=scripts/logs/extract_%A_%a.out
#SBATCH --time=168:00:00
#SBATCH --partition=nvidia-A6000-20
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --array=0-1

# =============================================================================
# ARGO-DeepMSI: Feature Extraction (Group-based Parallelization)
# =============================================================================
# Splits slides into 2 groups. Each group processes all its slides with
# ALL models in a single pass (per slide).
#
# Configuration:
# - Partition: nvidia-A6000-20 (80GB VRAM, 768G RAM available)
# - Memory: 128G per job (2 jobs × 128G = 256G total)
# - Time: 168 hours (7 days)
# - Array: 2 parallel jobs (0-1)
#
# Benefits:
# - Each slide preprocessed exactly once
# - All 11 models extracted per slide in one GPU session
# - 128G memory handles feature accumulation comfortably
# - A6000 80GB VRAM for large model support
#
# Usage:
#   sbatch scripts/extract.sh
#
# To change number of groups:
#   1. Update --array=0-N (where N = num_groups - 1)
#   2. Update NUM_GROUPS variable below
# =============================================================================

# Number of groups (must match --array parameter)
NUM_GROUPS=2

# Models to extract (all in one pass per slide)
MODELS=(
    # ===== Recommended Gated Models (HuggingFace auth required) =====
    "uni2"              # UNI v2 (1024D) - latest version
    "virchow2"          # Virchow v2 (1280D) - latest version
    "conch_v1.5"        # CONCH v1.5 (512D) - latest version
    "h-optimus-1"       # H-Optimus 1 (768D) - newer version
    "gigapath"          # GigaPath (1536D)
    "hibou-b"           # Hibou-B (768D)
    "musk"              # MUSK pathology foundation model

    # ===== Non-Gated Models (no auth required) =====
    "chief"             # CHIEF (768D)
    "ctranspath"        # CTransPath (768D)
    "phikonv2"          # Phikon v2 (768D)
    "plip"              # PLIP vision-language model

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

# Group ID for this task
GROUP_ID=$SLURM_ARRAY_TASK_ID

# Path to slide table
SLIDE_TABLE="results/data/slide_table.csv"

# Calculate group boundaries
TOTAL_SLIDES=$(tail -n +2 "$SLIDE_TABLE" | wc -l)
SLIDES_PER_GROUP=$(((TOTAL_SLIDES + NUM_GROUPS - 1) / NUM_GROUPS))  # Round up

# Calculate which lines to extract (accounting for header)
START_LINE=$((GROUP_ID * SLIDES_PER_GROUP + 2))  # +2 to skip header
END_LINE=$(((GROUP_ID + 1) * SLIDES_PER_GROUP + 1))

# Don't exceed total slides
if [ $END_LINE -gt $((TOTAL_SLIDES + 1)) ]; then
    END_LINE=$((TOTAL_SLIDES + 1))
fi

NUM_SLIDES=$((END_LINE - START_LINE + 1))

if [ $NUM_SLIDES -le 0 ]; then
    echo "No slides assigned to group $((GROUP_ID + 1))/$NUM_GROUPS. Exiting."
    exit 0
fi

echo "========================================="
echo "ARGO-DeepMSI: Feature Extraction"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Group: $((GROUP_ID + 1))/$NUM_GROUPS"
echo "Slides: $NUM_SLIDES (lines $START_LINE-$END_LINE of $TOTAL_SLIDES total)"
echo "Models: ${#MODELS[@]} models per slide"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "========================================="

# Create group table in logs directory (for record-keeping)
TEMP_TABLE="scripts/logs/slide_group_${GROUP_ID}_job${SLURM_JOB_ID}.csv"

# Extract header
head -n 1 "$SLIDE_TABLE" > "$TEMP_TABLE"

# Extract this group's slides
tail -n +$START_LINE "$SLIDE_TABLE" | head -n $NUM_SLIDES >> "$TEMP_TABLE"

echo ""
echo "Created group table: $TEMP_TABLE"
echo "Contains $(tail -n +2 "$TEMP_TABLE" | wc -l) slides"
echo ""

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

# Build model arguments
MODEL_ARGS=""
for model in "${MODELS[@]}"; do
    MODEL_ARGS="$MODEL_ARGS --model $model"
done

# Run extraction with ALL models on this group
echo ""
echo "Extracting ${#MODELS[@]} models from $NUM_SLIDES slides..."
echo "Models: ${MODELS[*]}"
echo ""
python -m argo_deepmsi.cli extract \
    "$TEMP_TABLE" \
    $MODEL_ARGS \
    --device cuda \
    --amp

echo ""
echo "========================================="
echo "✓ Extraction complete for group $((GROUP_ID + 1))/$NUM_GROUPS"
echo "Slide list saved: $TEMP_TABLE"
echo "End time: $(date)"
echo "========================================="
