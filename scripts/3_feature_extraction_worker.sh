#!/bin/bash
#SBATCH --job-name=extract
#SBATCH --partition=nvidia-2080ti-20
#SBATCH --output=/lab/barcheese01/mdiberna/ARGO-DeepMSI/logs/feature_extraction/workers/%x_%A_%a.out
#SBATCH --array=0-5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00

#######################################################################################
# Stage 3: Feature Extraction via STAMP
#
# Usage:
#   sbatch scripts/3_feature_extraction.sh <MODEL>
#
# Examples:
#   sbatch scripts/3_feature_extraction.sh ctranspath
#   sbatch scripts/3_feature_extraction.sh h-optimus-0
#   sbatch scripts/3_feature_extraction.sh virchow2
#
# This script:
# - Runs STAMP preprocessing for feature extraction
# - Works across all models (parameterized)
# - Processes 6 sites in parallel via SLURM array (0-5)
# - Stores features in: results/stage3_features/{MODEL}/{SITE}/
# - Uses configs from: configs/{MODEL}/config_{SITE}.yaml
#######################################################################################

# Define sites array (6 sites)
sites=("OAUTHC" "LUTH" "LASUTH" "UITH" "retrospective_msk" "retrospective_oau")

# Get model from command-line argument
MODEL=$1

if [ -z "$MODEL" ]; then
    echo "ERROR: Model name required"
    echo "Usage: sbatch scripts/3_feature_extraction.sh <MODEL>"
    echo "Example: sbatch scripts/3_feature_extraction.sh ctranspath"
    exit 1
fi

# Get the current site based on array index
SITE=${sites[$SLURM_ARRAY_TASK_ID]}

echo "================================="
echo "STAMP FEATURE EXTRACTION"
echo "Model: $MODEL"
echo "Site: $SITE"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Time: $(date)"
echo "================================="

# Define base directory
BASE_DIR="/lab/barcheese01/mdiberna/ARGO-DeepMSI"

# Load environment variables from .env (for HF_TOKEN)
if [ -f "$BASE_DIR/.env" ]; then
    export $(grep -v '^#' "$BASE_DIR/.env" | xargs)
fi

# Set environment variables
export HF_HOME="$BASE_DIR/.huggingface_cache"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH

# Create cache directories
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

# Activate STAMP environment
echo "Activating STAMP environment..."
source "$BASE_DIR/STAMP/.venv/bin/activate"

# Print GPU information
echo "==== GPU INFO ===="
nvidia-smi
echo "=================="

# Check PyTorch GPU access
echo "==== PYTORCH GPU CHECK ===="
python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('CUDA device count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('CUDA current device:', torch.cuda.current_device())
    print('CUDA device name:', torch.cuda.get_device_name(0))
"
echo "==========================="

# Check Hugging Face authentication for gated models
GATED_MODELS=("h-optimus-0" "h-optimus-1" "virchow2" "uni2" "conch1_5" "gigapath" "mstar" "musk")
if [[ " ${GATED_MODELS[@]} " =~ " ${MODEL} " ]]; then
    echo "==== HUGGING FACE CHECK ===="
    python -c "
import os
from huggingface_hub import HfApi
try:
    token = os.environ.get('HF_TOKEN')
    if not token:
        print('✗ HF_TOKEN not found in environment')
        print('  Add HF_TOKEN to .env file')
        exit(1)
    api = HfApi(token=token)
    user = api.whoami(token=token)
    print(f'✓ Logged in as: {user[\"name\"]}')
except Exception as e:
    print(f'✗ HF authentication failed: {e}')
    print('  Check your HF_TOKEN in .env file')
    exit(1)
"
    exit_code=$?
    echo "==========================="

    if [ $exit_code -ne 0 ]; then
        echo "ERROR: HF authentication required for $MODEL"
        exit 1
    fi
fi

# Define paths
TEMPLATE_FILE="$BASE_DIR/configs/templates/preprocessing_site.yaml.template"
CONFIG_FILE="$BASE_DIR/.temp_configs/$MODEL/config_${SITE}.yaml"

# Check if template exists
if [ ! -f "$TEMPLATE_FILE" ]; then
    echo "ERROR: Template file not found: $TEMPLATE_FILE"
    exit 1
fi

# Generate config from template
echo "Generating config from template..."
python "$BASE_DIR/scripts/generate_config.py" \
    --template "$TEMPLATE_FILE" \
    --output "$CONFIG_FILE" \
    --model "$MODEL" \
    --site "$SITE" \
    --device "cuda:0" \
    --base-dir "$BASE_DIR"

if [ $? -ne 0 ]; then
    echo "ERROR: Config generation failed"
    exit 1
fi

echo "Using generated config: $CONFIG_FILE"

# Change to project directory
cd "$BASE_DIR"

# Create log directory if it doesn't exist
mkdir -p "$BASE_DIR/logs/feature_extraction"

# Run STAMP preprocessing
echo "==== STARTING PREPROCESSING ===="
echo "Command: stamp --config $CONFIG_FILE preprocess"
echo "================================="

stamp --config "$CONFIG_FILE" preprocess

exit_code=$?

echo "================================="
echo "PREPROCESSING COMPLETED"
echo "Model: $MODEL"
echo "Site: $SITE"
echo "Exit code: $exit_code"
echo "Time: $(date)"
echo "================================="

if [ $exit_code -eq 0 ]; then
    echo "✓ Feature extraction completed successfully for $MODEL on $SITE"
else
    echo "✗ Feature extraction failed for $MODEL on $SITE (exit code: $exit_code)"
fi

exit $exit_code
