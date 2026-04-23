#!/bin/bash
#SBATCH --job-name=mdiberna_argo_multislide
#SBATCH --partition=20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=scripts/logs/%x_%j.out
#SBATCH --error=scripts/logs/%x_%j.err

set -euo pipefail
mkdir -p scripts/logs

eval "$(conda shell.bash hook)"
conda activate argo
cd /lab/barcheese01/mdiberna/ARGO-DeepMSI

export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2

python -u scripts/multislide_analysis.py "$@"
