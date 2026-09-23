#!/bin/bash
#SBATCH --job-name=mdiberna_argo_umap
#SBATCH --partition=20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=scripts/logs/%x_%j.out
#SBATCH --error=scripts/logs/%x_%j.err

set -euo pipefail
mkdir -p scripts/logs

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export NUMBA_NUM_THREADS=${NUMBA_NUM_THREADS:-4}

uv run --frozen python -u scripts/domain_shift/umap_embeddings.py "$@"
