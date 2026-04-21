#!/bin/bash
#SBATCH --job-name=argo_extract_dask
#SBATCH --output=scripts/logs/dask/driver_%j.out
#SBATCH --time=7-00:00:00
#SBATCH --partition=20
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2

# SLURM wrapper for scripts/extract_dask.py. Runs the Dask driver itself
# as a small CPU SLURM job so it isn't sitting on the head node; the
# driver spawns GPU workers via dask-jobqueue.
#
# Usage:
#   sbatch scripts/extract_dask.sh <slide_table.csv> [extra args for extract_dask.py]
#
# Examples:
#   # QC pass
#   sbatch scripts/extract_dask.sh results/data/slide_table_pyramidal.csv \
#       --models grandqc-artifact grandqc-tissue --memory "64 GB" --max-workers 5
#
#   # Foundation models (uses DEFAULT_MODELS)
#   sbatch scripts/extract_dask.sh results/data/slide_table_qc.csv --max-workers 3

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: sbatch $0 <slide_table.csv> [extra args]"
    exit 2
fi

SLIDE_TABLE="$1"
shift

echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURM_NODELIST"
echo "Slide table: $SLIDE_TABLE"
echo "Extra args:  $*"
echo "Start:       $(date)"

source /lab/barcheese01/mdiberna/miniconda3/etc/profile.d/conda.sh
conda activate argo

cd /lab/barcheese01/mdiberna/ARGO-DeepMSI
mkdir -p scripts/logs/dask

python scripts/extract_dask.py --slide-table "$SLIDE_TABLE" "$@"

echo "End: $(date)"
