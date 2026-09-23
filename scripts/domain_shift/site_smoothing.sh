#!/bin/bash
#SBATCH --job-name=argo_site_smoothing
#SBATCH --output=scripts/logs/site_smoothing_%x_%j.out
#SBATCH --error=scripts/logs/site_smoothing_%x_%j.err
#SBATCH --partition=nvidia-A6000-20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00

# Label-free site smoothing. GPU step re-scores Wagner under input moment matching:
#   sbatch scripts/domain_shift/site_smoothing.sh wagner
# CPU step (override partition/gres): shift-invariant bag descriptors + nested LR:
#   sbatch -p 20 --gres=none scripts/domain_shift/site_smoothing.sh bags --encoders mascaret phaet

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p scripts/logs
uv run --frozen python -u scripts/domain_shift/site_smoothing.py "$@"
