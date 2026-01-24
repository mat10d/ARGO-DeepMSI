#!/bin/bash
#SBATCH --job-name=argo_extract_gated
#SBATCH --output=scripts/logs/extract_gated_%A_%a.out
#SBATCH --time=48:00:00
#SBATCH --partition=nvidia-A6000-20
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --array=0-5%3

# Gated models (require HuggingFace authentication)
MODELS=(
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
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Model: $MODEL"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "========================================="

# Load conda
source /lab/barcheese01/mdiberna/miniconda3/etc/profile.d/conda.sh

# Activate environment
conda activate argo

# Set HuggingFace cache and load token
export HF_HOME=/lab/barcheese01/mdiberna/ARGO-DeepMSI/.huggingface_cache

# Login to HuggingFace (using token from .env)
cd /lab/barcheese01/mdiberna/ARGO-DeepMSI
python -c "
from dotenv import load_dotenv
import os
from huggingface_hub import login
load_dotenv('.env')
token = os.getenv('HF_TOKEN')
if token:
    login(token=token, add_to_git_credential=False)
    print('Logged in to HuggingFace')
else:
    print('WARNING: No HF_TOKEN found')
"

# Run extraction on full slide table
echo "Starting feature extraction with $MODEL..."
echo "Time: $(date)"

python -m argo_deepmsi.cli extract \
    results/data/slide_table.csv \
    --model $MODEL \
    --device cuda \
    --amp

echo "Extraction complete for $MODEL!"
echo "Time: $(date)"
