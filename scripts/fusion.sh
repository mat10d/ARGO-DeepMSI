#!/bin/bash
#SBATCH --job-name=mdiberna_argo_fusion
#SBATCH --partition=20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=scripts/logs/%x_%j.out
#SBATCH --error=scripts/logs/%x_%j.err

set -euo pipefail
mkdir -p scripts/logs

eval "$(conda shell.bash hook)"
conda activate argo
cd /lab/barcheese01/mdiberna/ARGO-DeepMSI

# Cap BLAS/OpenMP threads — on the head node this script can otherwise spawn
# 128 threads via scikit-learn's internal BLAS and thrash the shared box.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export BLIS_NUM_THREADS=${BLIS_NUM_THREADS:-4}

python -u scripts/fusion.py "$@"
