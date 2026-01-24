#!/bin/bash
#SBATCH --job-name=argo_extract_test
#SBATCH --output=scripts/logs/extract_uith_%j.out
#SBATCH --time=02:00:00
#SBATCH --partition=nvidia-t4-20
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Load conda
source  /lab/barcheese01/mdiberna/miniconda3/etc/profile.d/conda.sh

# Activate environment
conda activate argo

# Set HuggingFace cache
export HF_HOME=/lab/barcheese01/mdiberna/ARGO-DeepMSI/.huggingface_cache

# Run extraction on UITH test set
echo "Starting feature extraction on UITH slides..."
echo "Time: $(date)"

python -m argo_deepmsi.cli extract \
    results/data/slide_table_UITH_test.csv \
    --model plip \
    --device cuda \
    --overwrite

echo "Extraction complete!"
echo "Time: $(date)"
