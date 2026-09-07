#!/bin/bash
#SBATCH --job-name=mdiberna_agg_waiv
#SBATCH --output=scripts/logs/agg_waiv_%j.out
#SBATCH --time=03:00:00
#SBATCH --partition=20
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
set -euo pipefail
source /lab/barcheese01/mdiberna/miniconda3/etc/profile.d/conda.sh
conda activate argo
cd /lab/barcheese01/mdiberna/ARGO-DeepMSI
argo aggregate phaet,mascaret --method mean \
  --slide-table results/data/slide_table_pyramidal.csv --device cpu
