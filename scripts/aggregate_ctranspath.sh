#!/bin/bash
#SBATCH --job-name=argo_agg_ctranspath
#SBATCH --output=scripts/logs/agg_ctranspath_%j.out
#SBATCH --error=scripts/logs/agg_ctranspath_%j.err
#SBATCH --partition=20
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00

set -euo pipefail
source /lab/barcheese01/mdiberna/miniconda3/etc/profile.d/conda.sh
conda activate argo
cd "${SLURM_SUBMIT_DIR:-/lab/barcheese01/mdiberna/ARGO-DeepMSI}"
python -m argo_deepmsi.cli aggregate ctranspath \
    --slide-table results/data/slide_table_pyramidal.csv \
    --method mean
