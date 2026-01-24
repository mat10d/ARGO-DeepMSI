#!/bin/bash
#SBATCH --job-name=argo_agg_train_test
#SBATCH --output=scripts/logs/agg_train_test_%j.out
#SBATCH --time=01:00:00
#SBATCH --partition=nvidia-t4-20
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

echo "=========================================="
echo "ARGO-DeepMSI: Aggregation + Training Test"
echo "=========================================="
echo "Start time: $(date)"
echo ""

# Activate environment
source /lab/barcheese01/mdiberna/miniconda3/etc/profile.d/conda.sh
conda activate argo

# Set working directory
cd /lab/barcheese01/mdiberna/ARGO-DeepMSI

echo "Environment activated: argo"
echo "Working directory: $(pwd)"
echo ""

# ============================================================================
# Phase 1: Aggregate features (mean pooling)
# ============================================================================

echo "=========================================="
echo "Phase 1: Aggregating Features"
echo "=========================================="
echo ""

echo "Testing simple pooling aggregation..."
echo "Models: plip, ctranspath"
echo "Method: mean"
echo ""

python -m argo_deepmsi.cli aggregate \
    plip,ctranspath \
    --slide-table results/data/slide_table_UITH_test.csv \
    --method mean

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Aggregation completed successfully"
    echo ""

    # Check output
    echo "Checking output files..."
    ls -lh results/embeddings/plip_mean/
    ls -lh results/embeddings/ctranspath_mean/
    echo ""
else
    echo ""
    echo "✗ Aggregation failed!"
    echo "Stopping here."
    exit 1
fi

# ============================================================================
# Phase 2: Train classifiers on plip embeddings
# ============================================================================

echo "=========================================="
echo "Phase 2: Training Classifiers (plip)"
echo "=========================================="
echo ""

python -m argo_deepmsi.cli train \
    results/embeddings/plip_mean \
    --clinical results/data/clinical_table.csv \
    --splits 5

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Training on plip completed successfully"
    echo ""

    # Check output
    echo "Checking output files..."
    ls -lh results/models/plip_mean/
    echo ""

    echo "Classifier results:"
    cat results/models/plip_mean/classifier_comparison.csv
    echo ""
else
    echo ""
    echo "✗ Training on plip failed!"
fi

# ============================================================================
# Phase 3: Train classifiers on ctranspath embeddings
# ============================================================================

echo "=========================================="
echo "Phase 3: Training Classifiers (ctranspath)"
echo "=========================================="
echo ""

python -m argo_deepmsi.cli train \
    results/embeddings/ctranspath_mean \
    --clinical results/data/clinical_table.csv \
    --splits 5

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Training on ctranspath completed successfully"
    echo ""

    # Check output
    echo "Checking output files..."
    ls -lh results/models/ctranspath_mean/
    echo ""

    echo "Classifier results:"
    cat results/models/ctranspath_mean/classifier_comparison.csv
    echo ""
else
    echo ""
    echo "✗ Training on ctranspath failed!"
fi

# ============================================================================
# Summary
# ============================================================================

echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo ""

if [ -f results/models/plip_mean/classifier_comparison.csv ]; then
    echo "PLIP Results:"
    cat results/models/plip_mean/classifier_comparison.csv
    echo ""
fi

if [ -f results/models/ctranspath_mean/classifier_comparison.csv ]; then
    echo "CTransPath Results:"
    cat results/models/ctranspath_mean/classifier_comparison.csv
    echo ""
fi

echo "End time: $(date)"
echo "=========================================="
