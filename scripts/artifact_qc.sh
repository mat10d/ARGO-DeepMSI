#!/bin/bash
#SBATCH --job-name=argo_q2_aqc
#SBATCH --output=scripts/logs/q2_aqc_%A_%a.out
#SBATCH --error=scripts/logs/q2_aqc_%A_%a.err
#SBATCH --partition=nvidia-A6000-20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --array=0-2

# Q2-artifact-qc — GrandQC artifact segmentation over the clean cohort, sharded
# 3 ways (one GPU per array task). Submit via ralph/gpu_gate.sh so the 3 tasks
# never exceed the GPU cap. Reduce with `python -m argo_deepmsi.eval.cohort
# --artifact-qc-dir results/data/artifact_qc` afterwards.

set -euo pipefail

echo "Array task ${SLURM_ARRAY_TASK_ID}/2  Job ${SLURM_JOB_ID}  Node ${SLURM_NODELIST}  Start $(date)"

source /lab/barcheese01/mdiberna/miniconda3/etc/profile.d/conda.sh
conda activate argo

cd "${SLURM_SUBMIT_DIR:-/lab/barcheese01/mdiberna/ARGO-DeepMSI}"
mkdir -p scripts/logs
export HF_HOME="${HF_HOME:-/lab/barcheese01/mdiberna/.huggingface_cache}"
[ -f .env ] && { set -a; source .env; set +a; }

python -u scripts/artifact_qc.py \
    --cohort results/data/cohort_clean.csv \
    --slide-table results/data/slide_table_pyramidal.csv \
    --outdir results/data/artifact_qc \
    --variant 7x \
    --shard "${SLURM_ARRAY_TASK_ID}" \
    --nshards 3

echo "End: $(date)"
