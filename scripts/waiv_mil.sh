#!/bin/bash
#SBATCH --job-name=mdiberna_waiv_mil
#SBATCH --output=scripts/logs/waiv_mil_%j.out
#SBATCH --time=03:00:00
#SBATCH --partition=20
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -euo pipefail
source /lab/barcheese01/mdiberna/miniconda3/etc/profile.d/conda.sh
conda activate argo
cd /lab/barcheese01/mdiberna/ARGO-DeepMSI
python scripts/waiv_mil.py
