#!/bin/bash
#SBATCH --job-name=argo_extract_mussel
#SBATCH --output=scripts/logs/mussel/extract_%A_%a.out
#SBATCH --time=1-00:00:00
#SBATCH --partition=nvidia-A6000-20
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# SLURM array wrapper for `argo extract --backend mussel`. Each array task processes
# a contiguous shard of slide-table rows; outputs land next to each slide in
# <slide_stem>.mussel/<model>.features.{h5,pt} + <model>.provenance.json, and
# complete outputs are skipped on resubmission.
#
# Usage:
#   sbatch --array=0-7 scripts/extract_mussel.sh <slide_table.csv> [config.toml] [extra args]
#
# Environment overrides:
#   CONFIG           backend config (default: configs/backends/mussel-hoptimus0.toml)
#   SLIDES_PER_TASK  rows per array task (default: 1)
#
# Examples:
#   sbatch --array=0-99 scripts/extract_mussel.sh results/data/slide_table_pyramidal.csv
#   SLIDES_PER_TASK=10 sbatch --array=0-80 scripts/extract_mussel.sh \
#       results/data/slide_table_pyramidal.csv configs/backends/mussel-titan.toml

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: sbatch --array=0-N $0 <slide_table.csv> [config.toml] [extra args]"
    exit 2
fi

SLIDE_TABLE="$1"
shift
CONFIG="${CONFIG:-configs/backends/mussel-hoptimus0.toml}"
if [ $# -gt 0 ] && [[ "$1" == *.toml ]]; then
    CONFIG="$1"
    shift
fi
SLIDES_PER_TASK="${SLIDES_PER_TASK:-1}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
START=$(( TASK_ID * SLIDES_PER_TASK ))
END=$(( START + SLIDES_PER_TASK - 1 ))
INDICES=$(seq -s, "$START" "$END")

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p scripts/logs/mussel
export HF_HOME="${HF_HOME:-$PWD/.huggingface_cache}"

echo "Job ID:      ${SLURM_JOB_ID:-local} (array task $TASK_ID)"
echo "Node:        ${SLURM_NODELIST:-$(hostname)}"
echo "Slide table: $SLIDE_TABLE"
echo "Config:      $CONFIG"
echo "Rows:        $INDICES"
echo "Extra args:  $*"
echo "Start:       $(date)"

uv run --frozen argo extract --backend mussel --config "$CONFIG" "$SLIDE_TABLE" \
    --indices "$INDICES" "$@"

echo "End: $(date)"
