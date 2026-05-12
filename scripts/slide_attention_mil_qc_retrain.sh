#!/bin/bash
#SBATCH --job-name=argo_sam_qc
#SBATCH --output=scripts/logs/sam_qc_%j.out
#SBATCH --error=scripts/logs/sam_qc_%j.err
#SBATCH --partition=20  # CPU partition (model trains on CPU)
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

# Retrain slide_attention_mil on the QC-clean cohort.
# Output: results/scorers/slide_attention_mil/patient_scores_clean.csv

set -euo pipefail

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURM_NODELIST"
echo "Start:   $(date)"

source /lab/barcheese01/mdiberna/miniconda3/etc/profile.d/conda.sh
conda activate argo

cd "${SLURM_SUBMIT_DIR:-/lab/barcheese01/mdiberna/ARGO-DeepMSI}"
mkdir -p scripts/logs

python -u scripts/slide_attention_mil_qc_retrain.py

echo "End: $(date)"
