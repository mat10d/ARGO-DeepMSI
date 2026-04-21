#!/bin/bash
#SBATCH --job-name=argo_pyramidal
#SBATCH --output=scripts/logs/pyramidal_%j.out
#SBATCH --time=12:00:00
#SBATCH --partition=20
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# SLURM wrapper for `argo pyramidal`. Converts non-pyramidal slides in the
# input slide table to tiled pyramidal TIFFs via libvips, and writes an
# updated slide table (`<input>_pyramidal.csv`) pointing at the converted
# files.
#
# Usage:
#   sbatch scripts/pyramidal.sh [slide_table.csv]
#
# Default input: results/data/slide_table.csv

set -euo pipefail

SLIDE_TABLE="${1:-results/data/slide_table.csv}"

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Input:  $SLIDE_TABLE"
echo "Start:  $(date)"

source /lab/barcheese01/mdiberna/miniconda3/etc/profile.d/conda.sh
conda activate argo
cd /lab/barcheese01/mdiberna/ARGO-DeepMSI

argo pyramidal "$SLIDE_TABLE"

echo "End: $(date)"
