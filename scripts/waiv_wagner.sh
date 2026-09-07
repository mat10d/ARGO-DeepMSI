#!/bin/bash
#SBATCH --job-name=mdiberna_waiv_wagner
#SBATCH --output=scripts/logs/waiv_wagner_%j.out
#SBATCH --time=08:00:00
#SBATCH --partition=nvidia-A6000-20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -euo pipefail
source /lab/barcheese01/mdiberna/miniconda3/etc/profile.d/conda.sh
conda activate argo
cd /lab/barcheese01/mdiberna/ARGO-DeepMSI
python scripts/waiv_wagner.py "$1" --tile-cap 2048 --epochs 20
