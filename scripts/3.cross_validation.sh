#!/bin/bash
#SBATCH --job-name=stamp_crossval
#SBATCH --partition=nvidia-A100-20          
#SBATCH --output=out/stamp_crossval_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G          
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

# Define target directory where all features will be consolidated
TARGET_DIR="/lab/barcheese01/mdiberna/ARGO-DeepMSI/data/all/features/xiyuewang-ctranspath-7c998680-02627079/"

# Create target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Copy all feature data from individual repos to the consolidated directory
for dir in ../data/*/features/xiyuewang-ctranspath-7c998680-02627079/; do
    # Skip the target directory itself to avoid copying a directory into itself
    if [[ "$dir" != "../data/all/features/xiyuewang-ctranspath-7c998680-02627079/" ]]; then
        echo "Copying files from $dir to $TARGET_DIR"
        rsync -av "$dir"/* "$TARGET_DIR"
    fi
done
echo "All files have been consolidated to $TARGET_DIR"

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
CONFIG_FILE="$BASE_DIR/configs/config_all.yaml"  # Use a unified config for consolidated data

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found at $CONFIG_FILE"
    exit 1
fi

# Run cross-validation
echo "Running cross-validation..."
stamp --config "$CONFIG_FILE" crossval