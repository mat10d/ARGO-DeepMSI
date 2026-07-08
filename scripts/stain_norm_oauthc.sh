#!/bin/bash
#SBATCH --job-name=mdiberna_argo_a2_stainnorm
#SBATCH --partition=nvidia-A6000-20
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --array=0-2
#SBATCH --output=scripts/logs/%x_%A_%a.out
#SBATCH --error=scripts/logs/%x_%A_%a.err

# =============================================================================
# A2 — Macenko stain-norm OAUTHC-prospective + re-extract CONCH-TITAN
# =============================================================================
# 3-shard GPU array (respects the 3-GPU cap). Each shard processes OAUTHC clean
# slides i::3. Submit through the gate:
#   bash ralph/gpu_gate.sh sbatch scripts/stain_norm_oauthc.sh
# HARVEST: python scripts/stain_norm_oauthc_merge.py  (concat shards + build embedding)
# =============================================================================
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/lab/barcheese01/mdiberna/ARGO-DeepMSI}"
mkdir -p scripts/logs

source /lab/barcheese01/mdiberna/miniconda3/etc/profile.d/conda.sh
conda activate argo

# Frozen FM weights live in the project HF cache; use it offline (no gated auth).
export HF_HOME=/lab/barcheese01/mdiberna/ARGO-DeepMSI/.huggingface_cache
export HF_HUB_OFFLINE=1

NSHARDS="${NSHARDS:-3}"
echo "Job ${SLURM_ARRAY_JOB_ID} shard ${SLURM_ARRAY_TASK_ID}/${NSHARDS} on ${SLURM_NODELIST}  $(date)"

python scripts/stain_norm_oauthc.py \
    --shard "${SLURM_ARRAY_TASK_ID}" \
    --nshards "${NSHARDS}"

echo "End shard ${SLURM_ARRAY_TASK_ID}: $(date)"
