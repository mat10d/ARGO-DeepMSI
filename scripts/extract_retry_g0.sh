#!/bin/bash
#SBATCH --job-name=argo_extract_retry_g0
#SBATCH --output=scripts/logs/extract_retry_g0_%j.out
#SBATCH --time=168:00:00
#SBATCH --partition=nvidia-A6000-20
#SBATCH --gres=gpu:1
#SBATCH --mem=384G
#SBATCH --cpus-per-task=16

# Retry of group 0 after OOM on HP1353_21_1.svs (78k×75k non-pyramidal TIFF).
# Filtered slide list in scripts/logs/slide_group_0_retry.csv excludes that slide.
# Memory bumped to 384G for headroom vs. other potentially-large slides.
# Incremental: skips slides whose zarrs already have all 11 models.

MODELS=(
    "uni2" "virchow2" "conch_v1.5" "h-optimus-1" "gigapath" "hibou-b"
    "musk" "chief" "ctranspath" "phikonv2" "plip"
)

TEMP_TABLE="scripts/logs/slide_group_0_retry.csv"

echo "========================================="
echo "ARGO-DeepMSI: Feature Extraction (retry g0)"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Slides: $(tail -n +2 $TEMP_TABLE | wc -l)"
echo "Models: ${#MODELS[@]}"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "========================================="

source /lab/barcheese01/mdiberna/miniconda3/etc/profile.d/conda.sh
conda activate argo

export HF_HOME=/lab/barcheese01/mdiberna/ARGO-DeepMSI/.huggingface_cache

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
" 2>/dev/null

MODEL_ARGS=""
for model in "${MODELS[@]}"; do
    MODEL_ARGS="$MODEL_ARGS --model $model"
done

echo "Extracting ${#MODELS[@]} models..."
python -m argo_deepmsi.cli extract \
    "$TEMP_TABLE" \
    $MODEL_ARGS \
    --device cuda \
    --amp

echo "========================================="
echo "✓ Retry extraction complete"
echo "End: $(date)"
echo "========================================="
