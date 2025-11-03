#!/bin/bash
#SBATCH --job-name=stamp_hoptimus0_preprocess
#SBATCH --partition=nvidia-2080ti-20             
#SBATCH --output=out/stamp_hoptimus0_preprocess_%A_%a.out
#SBATCH --array=0-5  # 6 sites with h-optimus-0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G          
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00

# Define sites array
sites=("OAUTHC" "LUTH")
#"LASUTH" "UITH" "retrospective_msk" "retrospective_oau")

# Get the current site based on array index
SITE=${sites[$SLURM_ARRAY_TASK_ID]}
EXTRACTOR="h-optimus-0"

echo "Processing site: $SITE with extractor: $EXTRACTOR"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"

# Set environment variables
export HF_HOME="/lab/barcheese01/mdiberna/ARGO-DeepMSI/.huggingface_cache"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"

# Set CUDA environment (needed for GPU processing)
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH

# Create cache directories
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

# Activate STAMP environment
echo "Activating STAMP environment..."
source /lab/barcheese01/mdiberna/ARGO-DeepMSI/STAMP/.venv/bin/activate

# Print GPU information
echo "==== GPU INFO ===="
nvidia-smi
echo "================="

# Run a quick Python script to check PyTorch GPU access
echo "==== PYTORCH GPU CHECK ===="
python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('CUDA device count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('CUDA current device:', torch.cuda.current_device())
    print('CUDA device name:', torch.cuda.get_device_name(0))
"
echo "=========================="

# Check Hugging Face authentication for H-optimus models
echo "==== HUGGING FACE CHECK ===="
python -c "
from huggingface_hub import HfApi
try:
    api = HfApi()
    user = api.whoami()
    print(f'✓ Logged in as: {user[\"name\"]}')
except Exception as e:
    print(f'✗ HF authentication failed: {e}')
    print('Please run: hf auth login')
    exit(1)
"
echo "=========================="

# Define base directory and config path
BASE_DIR="/lab/barcheese01/mdiberna/ARGO-DeepMSI"
CONFIG_FILE="$BASE_DIR/configs/h-optimus-0/config_${SITE}.yaml"

echo "Using config file: $CONFIG_FILE"

# Change to project directory for running STAMP
cd "$BASE_DIR"

# Run preprocessing
echo "==== STARTING PREPROCESSING ===="
echo "Site: $SITE"
echo "Extractor: h-optimus-0"
echo "Config: $CONFIG_FILE"
echo "Time: $(date)"
echo "================================="

stamp --config "$CONFIG_FILE" preprocess

exit_code=$?

echo "================================="
echo "PREPROCESSING COMPLETED"
echo "Site: $SITE"
echo "Extractor: h-optimus-0"
echo "Exit code: $exit_code"
echo "Time: $(date)"
echo "================================="

if [ $exit_code -eq 0 ]; then
    echo "✓ Preprocessing completed successfully for $SITE with h-optimus-0"
else
    echo "✗ Preprocessing failed for $SITE with h-optimus-0 (exit code: $exit_code)"
fi

exit $exit_code