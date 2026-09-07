#!/bin/bash
#SBATCH --job-name=mdiberna_erranatomy_c
#SBATCH --output=scripts/logs/erranatomy_c_%j.out
#SBATCH --time=02:00:00
#SBATCH --partition=20
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Stage C runs Wagner on CACHED CTransPath features (not images), so it is CPU-only
# and does not contend with GPU extraction jobs.
set -euo pipefail
source /lab/barcheese01/mdiberna/miniconda3/etc/profile.d/conda.sh
conda activate argo
cd /lab/barcheese01/mdiberna/ARGO-DeepMSI
python -c "from argo_deepmsi.eval.error_anatomy import run_stage_c; \
  run_stage_c('results/analysis/error_anatomy/A_error_ledger.csv', \
              'results/analysis/error_anatomy', device='cpu')"
