#!/bin/bash
#SBATCH --job-name=argo_aggregate
#SBATCH --output=scripts/logs/aggregate_%A_%a.out
#SBATCH --time=4:00:00
#SBATCH --partition=20
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# =============================================================================
# ARGO-DeepMSI: Feature Aggregation (auto-discovery)
# =============================================================================
# Auto-discovers which models have been extracted by scanning zarr files.
# No hardcoded model list — works with whatever Phase 1 or Phase 2 produced.
#
# Usage:
#   sbatch scripts/aggregate.sh
# =============================================================================

METHOD="mean"
SLIDE_TABLE="results/data/slide_table_pyramidal.csv"

# Fall back to non-pyramidal if pyramidal doesn't exist
if [ ! -f "$SLIDE_TABLE" ]; then
    SLIDE_TABLE="results/data/slide_table.csv"
fi

echo "========================================="
echo "ARGO-DeepMSI: Aggregation (auto-discovery)"
echo "========================================="
echo "Slide table: $SLIDE_TABLE"
echo "Method: $METHOD"
echo "Start time: $(date)"

# Auto-discover models by scanning the first zarr's tables/ directory
FIRST_ZARR=$(python3 -c "
import pandas as pd
from pathlib import Path
df = pd.read_csv('$SLIDE_TABLE')
for _, row in df.iterrows():
    z = Path(row['FILENAME']).with_suffix('.zarr')
    if z.exists() and (z / 'tables').exists():
        print(z)
        break
")

if [ -z "$FIRST_ZARR" ]; then
    echo "ERROR: No zarr files found. Run extraction first."
    exit 1
fi

# Get all model names from the first zarr
MODELS=()
for table_dir in "$FIRST_ZARR"/tables/*_tiles; do
    if [ -d "$table_dir" ]; then
        model_name=$(basename "$table_dir" | sed 's/_tiles$//')
        MODELS+=("$model_name")
    fi
done

echo "Discovered ${#MODELS[@]} models: ${MODELS[*]}"
echo "========================================="

# Load conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate argo

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Aggregate each model
for model in "${MODELS[@]}"; do
    echo ""
    echo "Aggregating $model with $METHOD..."
    python -m argo_deepmsi.cli aggregate \
        "$model" \
        --slide-table "$SLIDE_TABLE" \
        --method "$METHOD"
done

echo ""
echo "========================================="
echo "✓ Aggregation complete for ${#MODELS[@]} models"
echo "End time: $(date)"
echo "========================================="
