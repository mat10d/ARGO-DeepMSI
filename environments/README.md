# ARGO-DeepMSI Environments

This directory contains environment configurations for the three separate environments used in the pipeline.

## Environment Overview

### 1. ARGO Environment (Main Pipeline)
- **File**: `argo.yml`
- **Python**: 3.9 (consider upgrading to 3.10)
- **Manager**: conda
- **Purpose**: Data ingestion, QC, feature validation, baseline testing, statistics, visualization
- **Used in**: Stages 1, 2, 4, 5, 7, 8

### 2. STAMP Environment (Feature Extraction & Training)
- **Location**: `../STAMP/.venv/`
- **Python**: 3.12
- **Manager**: uv (per STAMP requirements)
- **Purpose**: Feature extraction and MIL training
- **Used in**: Stages 3, 6

### 3. HistoBistro Environment (Baseline Validation)
- **File**: `../HistoBistro/environment_simple.yaml`
- **Python**: 3.10.9
- **Manager**: conda
- **Purpose**: Pre-trained model inference for baseline validation
- **Used in**: Stage 5 (baseline testing)

## Quick Setup

```bash
# From the ARGO-DeepMSI root directory

# 1. Create ARGO environment
conda env create -f environments/argo.yml
conda activate argo

# 2. Setup STAMP environment
cd STAMP
rm -r ~/.triton  # Clear triton cache (important!)
uv sync --extra build --extra gpu
source .venv/bin/activate
cd ..

# 3. Setup HistoBistro environment
conda env create -f HistoBistro/environment_simple.yaml
conda activate histobistro
```

## Environment Usage by Stage

| Stage | Environment | Activation Command |
|-------|-------------|-------------------|
| 1. Data Ingestion | ARGO | `conda activate argo` |
| 2. Quality Control | ARGO | `conda activate argo` |
| 3. Feature Extraction | STAMP | `source STAMP/.venv/bin/activate` |
| 4. Feature Validation | ARGO | `conda activate argo` |
| 5. Baseline Testing | HistoBistro | `conda activate histobistro` |
| 6. MIL Training | STAMP | `source STAMP/.venv/bin/activate` |
| 7. Statistics | ARGO | `conda activate argo` |
| 8. Visualization | ARGO | `conda activate argo` |

## Environment Variables

Set these in your `~/.bashrc` or `.env` file:

```bash
# Hugging Face cache (required for STAMP)
export HF_HOME=/lab/barcheese01/mdiberna/ARGO-DeepMSI/.huggingface_cache

# REDCap credentials (create .env file in root)
export REDCAP_API_URL=https://redcap.oauife.edu.ng/api/
export REDCAP_API_TOKEN=your_token_here
```

## Automated Setup Script

Use the automated setup script (recommended):

```bash
bash environments/setup.sh
```

This will:
1. Check for existing environments
2. Create all three environments
3. Verify installations
4. Test environment switching

## Troubleshooting

### STAMP Installation Issues

If you encounter errors during STAMP installation:

```bash
# Clear caches
rm -r ~/.triton
uv cache clean flash_attn
uv cache clean mamba-ssm
uv cache clean causal_conv1d

# Reinstall
cd STAMP
uv sync --extra build
uv sync --extra build --extra gpu
```

### CUDA Issues

If PyTorch doesn't detect GPU:

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# If False, check CUDA installation
nvidia-smi
echo $CUDA_HOME
```

### HistoBistro Environment

If HistoBistro environment fails:

```bash
# Try the simpler environment file
conda env create -f HistoBistro/environment_simple.yaml

# Or manually install key packages
conda create -n histobistro python=3.10.9
conda activate histobistro
conda install pytorch=2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pytorch-lightning=2.0.1 -c conda-forge
```

## Updating Environments

To update environments as dependencies change:

```bash
# ARGO environment
conda env update -f environments/argo.yml --prune

# STAMP environment
cd STAMP
git pull
uv sync --extra build --extra gpu
cd ..

# HistoBistro environment
cd HistoBistro
git pull
conda env update -f environment_simple.yaml --prune
cd ..
```

## Testing Environments

Test each environment after setup:

```bash
# Test ARGO
conda activate argo
python -c "import pandas, numpy, sklearn, matplotlib, seaborn, requests; print('ARGO OK')"

# Test STAMP
source STAMP/.venv/bin/activate
python -c "import torch, transformers, timm, lightning; print('STAMP OK'); print(f'CUDA: {torch.cuda.is_available()}')"

# Test HistoBistro
conda activate histobistro
python -c "import torch, pytorch_lightning; print('HistoBistro OK')"
```
