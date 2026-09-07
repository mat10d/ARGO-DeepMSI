#!/bin/bash
#SBATCH --job-name=mdiberna_vl_tumor
#SBATCH --output=scripts/logs/vl_tumor_%j.out
#SBATCH --time=03:00:00
#SBATCH --partition=20
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
set -euo pipefail
source /lab/barcheese01/mdiberna/miniconda3/etc/profile.d/conda.sh
conda activate argo
cd /lab/barcheese01/mdiberna/ARGO-DeepMSI
python scripts/vl_tumor_forensic.py
