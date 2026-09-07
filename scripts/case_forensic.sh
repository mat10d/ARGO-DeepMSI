#!/bin/bash
#SBATCH --job-name=mdiberna_case_forensic
#SBATCH --output=scripts/logs/case_forensic_%j.out
#SBATCH --time=00:30:00
#SBATCH --partition=20
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Per-case failure forensic (CPU: cached VL tiles + small Wagner forward).
# Usage: sbatch scripts/case_forensic.sh <slide_id> <patient_id>
set -euo pipefail
source /lab/barcheese01/mdiberna/miniconda3/etc/profile.d/conda.sh
conda activate argo
cd /lab/barcheese01/mdiberna/ARGO-DeepMSI
python scripts/case_forensic.py --slide-id "$1" --patient-id "${2:-}"
