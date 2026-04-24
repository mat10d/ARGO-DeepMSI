#!/bin/bash
#SBATCH --job-name=mdiberna_argo_c5_phase2
#SBATCH --partition=20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=scripts/logs/%x_%j.out
#SBATCH --error=scripts/logs/%x_%j.err

# =============================================================================
# ARGO-DeepMSI: C5 Phase 2 — SlideAttentionMSI
# =============================================================================
# Trains a lightweight attention model (~5K params) over patient slide bags.
# CPU only — no GPU needed. Runs full ablation in ~30min.
#
# Usage:
#   sbatch scripts/c5_phase2.sh
# =============================================================================

echo "========================================="
echo "ARGO-DeepMSI: Phase 2 — SlideAttentionMSI"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "========================================="

# Load conda environment
source /lab/barcheese01/mdiberna/miniconda3/etc/profile.d/conda.sh
conda activate argo

cd /lab/barcheese01/mdiberna/ARGO-DeepMSI

python scripts/c5_phase2_slide_attention.py

echo ""
echo "========================================="
echo "✓ Phase 2 complete"
echo "End time: $(date)"
echo "========================================="
