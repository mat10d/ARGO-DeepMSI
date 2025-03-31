#!/bin/bash
#SBATCH --job-name=stamp_preprocess
#SBATCH --partition=nvidia-2080ti-20             
#SBATCH --output=out/stamp_preprocess_%A_%a.out
#SBATCH --array=0-5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G          
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

# Define base directory and config path
BASE_DIR="/lab/barcheese01/mdiberna/ARGO-DeepMSI"
CONFIG_FILE="$BASE_DIR/configs/config_$SITE.yaml"

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file for $SITE not found at $CONFIG_FILE"
    exit 1
fi

# Step 1: Feature extraction
echo "Running preprocessing for $SITE..."
stamp --config "$CONFIG_FILE" preprocess
