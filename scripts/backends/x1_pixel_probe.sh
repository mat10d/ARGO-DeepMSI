#!/bin/bash
#SBATCH --job-name=mdiberna_argo_x1_probe
#SBATCH --output=logs/%x_%j.out
#SBATCH --partition=nvidia-A6000-20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00

# X1: replay each backend's read + preprocessing on identical tile boxes and factorise the
# H-optimus-0 feature difference (reader / resize / runtime). Needs the Mussel (a) and
# LazySlide 224 (c) stores. See scripts/backends/x1_pixel_probe.py.
#
# Usage: sbatch scripts/backends/x1_pixel_probe.sh [--k 64]

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs .tmp_build
export HF_HOME="$PWD/.huggingface_cache" HF_HUB_OFFLINE=1 TMPDIR="$PWD/.tmp_build"
PROBE=scripts/backends/x1_pixel_probe.py

uv run --frozen --project envs/lazyslide python "$PROBE" ls-read "$@"
uv run --frozen --project envs/mussel python "$PROBE" m-read "$@"
uv run --frozen --project envs/lazyslide python "$PROBE" ls-cross
uv run --frozen python "$PROBE" report
