#!/bin/bash
#SBATCH --job-name=mdiberna_argo_c5_phase1b
#SBATCH --partition=20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=scripts/logs/%x_%j.out
#SBATCH --error=scripts/logs/%x_%j.err

set -euo pipefail
mkdir -p scripts/logs

eval "$(conda shell.bash hook)"
conda activate argo
cd /lab/barcheese01/mdiberna/ARGO-DeepMSI

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

python -u scripts/c5_phase1b.py "$@"
