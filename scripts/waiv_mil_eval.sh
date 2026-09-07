#!/bin/bash
#SBATCH --job-name=mdiberna_waivmil_eval
#SBATCH --output=scripts/logs/waivmil_eval_%j.out
#SBATCH --time=08:00:00
#SBATCH --partition=20
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
set -euo pipefail
source /lab/barcheese01/mdiberna/miniconda3/etc/profile.d/conda.sh
conda activate argo
cd /lab/barcheese01/mdiberna/ARGO-DeepMSI
python scripts/waiv_mil.py eval "$1"
