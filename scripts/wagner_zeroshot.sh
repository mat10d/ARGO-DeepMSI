#!/bin/bash
#SBATCH --job-name=argo_wagner
#SBATCH --output=scripts/logs/wagner_%j.out
#SBATCH --error=scripts/logs/wagner_%j.err
#SBATCH --partition=nvidia-A6000-20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00

# A3 — Zero-shot Wagner et al. (HistoBistro, Cancer Cell 2023) MSI classifier
# evaluation on the Nigerian cohort using our CTransPath features.
#
# Usage: sbatch scripts/wagner_zeroshot.sh

set -euo pipefail

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURM_NODELIST"
echo "Start:   $(date)"

source /lab/barcheese01/mdiberna/miniconda3/etc/profile.d/conda.sh
conda activate argo

cd "${SLURM_SUBMIT_DIR:-/lab/barcheese01/mdiberna/ARGO-DeepMSI}"
mkdir -p scripts/logs

python -u scripts/wagner_zeroshot.py

echo "End: $(date)"
