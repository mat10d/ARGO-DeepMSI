#!/bin/bash
#SBATCH --job-name=argo_q3_smokeoff
#SBATCH --output=scripts/logs/q3_smokeoff_%j.out
#SBATCH --error=scripts/logs/q3_smokeoff_%j.err
#SBATCH --partition=nvidia-A6000-20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --requeue

# Q3-tumor-filter SMOKE-OFF — GrandQC tissue seg (GPU) vs cached CTransPath TUM
# head on a fixed 20-slide probe. Submit via ralph/gpu_gate.sh (GPU cap 3).
# Reduce/harvest with scripts/tumor_tiles_apply.py + eval.cohort --tumor-tiles-dir.

set -euo pipefail

echo "Job ${SLURM_JOB_ID}  Node ${SLURM_NODELIST}  Start $(date)"

source /lab/barcheese01/mdiberna/miniconda3/etc/profile.d/conda.sh
conda activate argo

cd "${SLURM_SUBMIT_DIR:-/lab/barcheese01/mdiberna/ARGO-DeepMSI}"
mkdir -p scripts/logs
export HF_HOME="${HF_HOME:-/lab/barcheese01/mdiberna/.huggingface_cache}"
[ -f .env ] && { set -a; source .env; set +a; }

python -u scripts/tumor_filter_smokeoff.py \
    --cohort results/data/cohort_clean.csv \
    --slide-table results/data/slide_table_pyramidal.csv \
    --n-probe 20 --seed 42 --floor 0.01 --device cuda

echo "End: $(date)"
