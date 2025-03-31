#!/bin/bash
#SBATCH --job-name=stamp_crossval
#SBATCH --partition=nvidia-2080ti-20             
#SBATCH --output=out/stamp_crossval_%A_%a.out
#SBATCH --array=0-5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G          
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

# Define sites array
sites=("OAUTHC" "LUTH" "LASUTH" "UITH" "retrospective_msk" "retrospective_oau")

# Get the current site based on array index
SITE=${sites[$SLURM_ARRAY_TASK_ID]}
echo "Processing site: $SITE"

# Load conda environment
source ~/.bashrc
conda activate stamp-env

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

# Define base directory and config path
BASE_DIR="/lab/barcheese01/mdiberna/ARGO-DeepMSI"
CONFIG_FILE="$BASE_DIR/configs/config_$SITE.yaml"

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file for $SITE not found at $CONFIG_FILE"
    exit 1
fi

# Step 2: Cross-validation
echo "Running cross-validation for $SITE..."
stamp --config "$CONFIG_FILE" crossval
