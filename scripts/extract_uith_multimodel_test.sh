#!/bin/bash
#SBATCH --job-name=argo_multi_test
#SBATCH --output=scripts/logs/extract_multi_%j.out
#SBATCH --time=04:00:00
#SBATCH --partition=nvidia-t4-20
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

echo "Starting multi-model feature extraction on UITH slides..."
echo "Time: $(date)"

# Activate environment
source /lab/barcheese01/mdiberna/miniconda3/etc/profile.d/conda.sh
conda activate argo

# Set HuggingFace cache
export HF_HOME=/lab/barcheese01/mdiberna/ARGO-DeepMSI/.huggingface_cache

# Test with two non-gated models: plip and ctranspath
python -m argo_deepmsi.cli extract \
    results/data/slide_table_UITH_test.csv \
    --model plip \
    --model ctranspath \
    --device cuda \
    --overwrite

echo "Extraction complete!"
echo "Time: $(date)"
