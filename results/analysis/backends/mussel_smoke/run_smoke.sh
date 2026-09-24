#!/bin/bash
#SBATCH --job-name=mdiberna_argo_mussel_smoke
#SBATCH --partition=nvidia-A6000-20
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Mussel smoke at default settings. Submit from the repo root.
# Usage: sbatch results/analysis/backends/mussel_smoke/run_smoke.sh <slide_path> <OPTIMUS|TITAN>
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
SLIDE="$1"; MODE="$2"
STEM=$(basename "$SLIDE"); STEM="${STEM%.*}"
OUT=results/analysis/backends/mussel_smoke/$STEM
mkdir -p "$OUT" .tmp_build
export TMPDIR=$PWD/.tmp_build HF_HOME=$PWD/.huggingface_cache HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 2; done ) > "$OUT/gpu_mem_${MODE}.log" &
MON=$!
START=$(date +%s)
if [ "$MODE" = OPTIMUS ]; then
  /usr/bin/time -v uv run --frozen --project envs/mussel tessellate_extract_features \
    slide_path="$PWD/$SLIDE" output_h5_path="$PWD/$OUT/OPTIMUS.features.h5" \
    output_pt_path="$PWD/$OUT/OPTIMUS.features.pt" model_type=OPTIMUS
else
  /usr/bin/time -v uv run --frozen --project envs/mussel tessellate_extract_features \
    slide_path="$PWD/$SLIDE" output_dir="$PWD/$OUT/titan" \
    model_type=CONCH1_5 slide_model_type=TITAN_SLIDE
fi
END=$(date +%s)
kill $MON || true
echo "WALL_SECONDS=$((END-START)) PEAK_GPU_MIB=$(sort -n "$OUT/gpu_mem_${MODE}.log" | tail -1)" | tee "$OUT/timing_${MODE}.txt"
