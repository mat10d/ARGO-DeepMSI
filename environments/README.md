# Environment Setup Guide

## Overview

ARGO-DeepMSI uses **multiple separate environments** for different pipeline stages:

| Environment | Manager | Python | Purpose | Stages |
|-------------|---------|--------|---------|--------|
| **ARGO** | conda | 3.11 | Data processing, validation, analysis | 1, 2, 4, 7, 8 |
| **STAMP** | uv/venv | 3.11+ | Feature extraction, MIL training | 3, 6 |
| **HistoBistro** | conda | 3.10 | Baseline testing | 5 |

---

## 1. ARGO Environment (Main)

**Purpose:** Data ingestion, QC, feature validation, statistics, visualization

### Installation

```bash
# Create environment from file
conda env create -f environments/argo.yml

# Activate
conda activate argo

# Install argo-deepmsi package (editable mode)
pip install -e .
```

### Verify Installation

```bash
conda activate argo
python -c "from argo_deepmsi import data_ingestion; print('✓ ARGO environment ready')"
```

### Used For

- Stage 1: Data ingestion (REDCap, Halo data)
- Stage 2: Quality control
- Stage 4: Feature validation
- Stage 7: Statistics
- Stage 8: Visualization

---

## 2. STAMP Environment (Feature Extraction)

**Purpose:** Feature extraction and MIL training using STAMP framework

### Installation

The STAMP environment should **already exist** at `STAMP/.venv/` (installed via `uv`).

If it doesn't exist, install it:

**IMPORTANT:** Install STAMP on a **compute node**, not the head/login node. The head node doesn't have CUDA development tools needed for GPU packages.

```bash
# Step 1: Request an interactive GPU node
srun --partition=nvidia-2080ti-20 \
     --gres=gpu:1 \
     --mem=64G \
     --cpus-per-task=8 \
     --time=2:00:00 \
     --pty bash

# Step 2: Once on compute node, install STAMP
cd /lab/barcheese01/mdiberna/ARGO-DeepMSI/STAMP

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install STAMP with GPU support
MAX_JOBS=4 uv sync --extra build --extra gpu

# Step 3: Activate and verify
source .venv/bin/activate
stamp --version

# Step 4: Exit compute node
exit
```

**Why compute node?** Packages like `causal-conv1d` (used by Gigapath) require CUDA compilation tools (`nvcc`) which are only available on compute nodes.

### Configure Hugging Face Cache

**IMPORTANT:** STAMP models are large and should use a shared cache:

```bash
# This is already set in .env file
export HF_HOME="/lab/barcheese01/mdiberna/ARGO-DeepMSI/.huggingface_cache"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"

# Create cache directories
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"
```

### Authenticate with Hugging Face

Many STAMP models (virchow2, uni2, h-optimus-0, etc.) are **gated** and require authentication:

```bash
# Activate STAMP environment first
source STAMP/.venv/bin/activate

# Login to Hugging Face (do this ONCE)
hf auth login

# You'll be prompted for your HF token
# Get token from: https://huggingface.co/settings/tokens
```

### Verify Installation

```bash
source STAMP/.venv/bin/activate
stamp --version
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Request Access to Gated Models

Before using gated models, request access on Hugging Face:

- **Virchow2**: https://huggingface.co/paige-ai/Virchow2
- **UNI2**: https://huggingface.co/MahmoodLab/UNI2-h
- **H-optimus-0**: https://huggingface.co/bioptimus/H-optimus-0
- **H-optimus-1**: https://huggingface.co/bioptimus/H-optimus-1
- **CONCHv1.5**: https://huggingface.co/MahmoodLab/conchv1_5
- **Gigapath**: https://huggingface.co/prov-gigapath/prov-gigapath

After requesting access and being approved, run `hf auth login` to authenticate.

### Used For

- Stage 3: Feature extraction (preprocessing)
- Stage 6: MIL training (cross-validation)

---

## 3. HistoBistro Environment (Baseline Testing)

**Purpose:** Pre-trained model inference for baseline validation

### Installation

```bash
# Create environment
conda env create -f HistoBistro/environment_simple.yaml

# Activate
conda activate histobistro
```

### Verify Installation

```bash
conda activate histobistro
python -c "import torch; print('✓ HistoBistro environment ready')"
```

### Used For

- Stage 5: Baseline testing with pre-trained HistoBistro model

---

## Quick Reference

### Environment Activation

```bash
# For data ingestion, validation, stats, viz
conda activate argo

# For feature extraction and training
source STAMP/.venv/bin/activate

# For baseline testing
conda activate histobistro
```

### Check Current Environment

```bash
# Conda environments
conda env list

# Active Python environment
which python
python --version
```

---

## Complete Setup (First Time)

Run these steps **once** when setting up the project:

```bash
# 1. Clone repository (if not already done)
cd /lab/barcheese01/mdiberna/ARGO-DeepMSI

# 2. Create ARGO environment
conda env create -f environments/argo.yml
conda activate argo
pip install -e .

# 3. Verify STAMP environment exists
source STAMP/.venv/bin/activate
stamp --version

# If STAMP not installed, install it:
# cd STAMP && uv sync --extra build --extra gpu && cd ..

# 4. Configure HF cache (already in .env)
export HF_HOME="/lab/barcheese01/mdiberna/ARGO-DeepMSI/.huggingface_cache"
mkdir -p "$HF_HOME"

# 5. Authenticate with Hugging Face
hf auth login
# Enter token from: https://huggingface.co/settings/tokens

# 6. Create HistoBistro environment (if using baseline testing)
conda env create -f HistoBistro/environment_simple.yaml

# 7. Verify setup
conda activate argo
python -c "from argo_deepmsi import data_ingestion; print('✓ ARGO OK')"

source STAMP/.venv/bin/activate
python scripts/test_model_access.py ctranspath
# Should print: ✓ Model ctranspath is accessible and ready to use

conda activate histobistro
python -c "import torch; print('✓ HistoBistro OK')"
```

---

## Environment Variables

Key environment variables (already set in `.env` and scripts):

```bash
# Hugging Face cache (for STAMP models)
export HF_HOME="/lab/barcheese01/mdiberna/ARGO-DeepMSI/.huggingface_cache"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"

# CUDA (for GPU support)
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
```

These are automatically set by SLURM scripts - no manual action needed.

---

## Troubleshooting

### STAMP installation fails with "nvcc not found" or causal-conv1d build error

**Problem:** You're trying to install on the head/login node which doesn't have CUDA development tools.

**Solution:** Install on a compute node using an interactive job (see Installation section above).

### STAMP environment not found

```bash
# Request compute node first
srun --partition=nvidia-2080ti-20 --gres=gpu:1 --mem=64G --time=2:00:00 --pty bash

# Then install
cd STAMP
MAX_JOBS=4 uv sync --extra build --extra gpu
source .venv/bin/activate
exit
```

### HF authentication fails

```bash
# Make sure you're in STAMP environment
source STAMP/.venv/bin/activate

# Login with your token
hf auth login

# Test access
python scripts/test_model_access.py virchow2
```

### GPU not available

```bash
# Check CUDA
nvidia-smi

# Check PyTorch
python -c "import torch; print(torch.cuda.is_available())"

# If false, reinstall PyTorch with CUDA support
```

### Models too large / disk space issues

Models are cached in `$HF_HOME`. Check disk usage:

```bash
du -sh /lab/barcheese01/mdiberna/ARGO-DeepMSI/.huggingface_cache
```

To clear cache (if needed):

```bash
rm -rf /lab/barcheese01/mdiberna/ARGO-DeepMSI/.huggingface_cache/*
# You'll need to re-download models
```

---

## Pipeline Stage → Environment Mapping

| Stage | Script | Environment | Activation |
|-------|--------|-------------|------------|
| 1. Data Ingestion | `1_data_ingestion.py` | ARGO | `conda activate argo` |
| 2. Quality Control | `2_quality_control.py` | ARGO | `conda activate argo` |
| 3. Feature Extraction | `3_feature_extraction.sh` | STAMP | `source STAMP/.venv/bin/activate` |
| 4. Feature Validation | `4_feature_validation.py` | ARGO | `conda activate argo` |
| 5. Baseline Testing | `5_baseline_testing.py` | HistoBistro | `conda activate histobistro` |
| 6. MIL Training | `6_mil_training.sh` | STAMP | `source STAMP/.venv/bin/activate` |
| 7. Statistics | `7_statistics.py` | ARGO | `conda activate argo` |
| 8. Visualization | `8_visualization.py` | ARGO | `conda activate argo` |

**Note:** SLURM scripts (`.sh` files) automatically activate the correct environment.
