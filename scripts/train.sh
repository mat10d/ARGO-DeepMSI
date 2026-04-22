#!/bin/bash
#SBATCH --job-name=argo_train
#SBATCH --output=scripts/logs/train_%A_%a.out
#SBATCH --time=2:00:00
#SBATCH --partition=20
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# =============================================================================
# ARGO-DeepMSI: Classifier Training (auto-discovery)
# =============================================================================
# Auto-discovers which embeddings exist in results/embeddings/ and trains
# classifiers on each. No hardcoded list — works with whatever was aggregated.
#
# Usage:
#   sbatch scripts/train.sh
# =============================================================================

CLINICAL_TABLE="results/data/clinical_table.csv"
EMBEDDINGS_DIR="results/embeddings"

echo "========================================="
echo "ARGO-DeepMSI: Training (auto-discovery)"
echo "========================================="
echo "Clinical table: $CLINICAL_TABLE"
echo "Start time: $(date)"

# Auto-discover embedding directories
EMBEDDINGS=()
for emb_dir in "$EMBEDDINGS_DIR"/*/; do
    if [ -f "${emb_dir}embeddings.npy" ] || [ -f "${emb_dir}embeddings.h5ad" ]; then
        EMBEDDINGS+=("$emb_dir")
    fi
done

if [ ${#EMBEDDINGS[@]} -eq 0 ]; then
    echo "ERROR: No embedding directories found in $EMBEDDINGS_DIR. Run aggregation first."
    exit 1
fi

echo "Discovered ${#EMBEDDINGS[@]} embedding sets:"
for emb in "${EMBEDDINGS[@]}"; do
    echo "  - $(basename "$emb")"
done
echo "========================================="

# Load conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate argo

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Train on each embedding set
for emb_dir in "${EMBEDDINGS[@]}"; do
    emb_name=$(basename "$emb_dir")
    echo ""
    echo "Training on $emb_name..."
    python -m argo_deepmsi.cli train \
        "$emb_dir" \
        --clinical "$CLINICAL_TABLE"
done

echo ""
echo "========================================="
echo "✓ Training complete for ${#EMBEDDINGS[@]} embedding sets"
echo "End time: $(date)"
echo "========================================="

# Summary
echo ""
echo "Results summary:"
for f in results/models/*/classifier_comparison.csv; do
    if [ -f "$f" ]; then
        model=$(basename $(dirname "$f"))
        best=$(python3 -c "
import pandas as pd
d=pd.read_csv('$f')
r=d.sort_values('auroc_mean',ascending=False).iloc[0]
print(f\"{r['classifier']:25s} auroc={r['auroc_mean']:.3f}+/-{r['auroc_std']:.3f}\")
" 2>/dev/null)
        printf "  %-25s  %s\n" "$model" "$best"
    fi
done
