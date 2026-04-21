#!/bin/bash
#SBATCH --job-name=argo_pyramidal
#SBATCH --output=scripts/logs/convert_pyramidal_%j.out
#SBATCH --time=12:00:00
#SBATCH --partition=20
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# Wrap scripts/convert_pyramidal.sh in a SLURM job so the vips conversions
# (which can read 78k×75k non-pyramidal TIFFs into RAM) don't hit the head
# node. Writes results/data/slide_table_pyramidal.csv.

set -euo pipefail

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Start:  $(date)"

source /lab/barcheese01/mdiberna/miniconda3/etc/profile.d/conda.sh
conda activate argo

cd /lab/barcheese01/mdiberna/ARGO-DeepMSI
bash scripts/convert_pyramidal.sh results/data/slide_table.csv

echo "End: $(date)"
