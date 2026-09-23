#!/bin/bash
#SBATCH --job-name=mdiberna_waiv_race
#SBATCH --output=scripts/logs/waiv_race_%j.out
#SBATCH --time=01:00:00
#SBATCH --partition=20
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
uv run --frozen python scripts/domain_shift/waiv_race.py
