# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ARGO-DeepMSI is a multi-stage pipeline for microsatellite instability (MSI) prediction from whole slide images of colorectal cancer. The pipeline integrates REDCap data collection, multi-site processing, and cross-validation using multiple specialized environments.

## Environment Architecture

This project uses **multiple separate environments** for different stages of the pipeline:

### 1. ARGO Environment (Conda)
- **Purpose**: Data collection, processing, validation, and post-processing analysis
- **Python version**: 3.9
- **Activation**: `conda activate argo`
- **Setup**: `conda env create -f argo_env.yml`
- **Used in**: Scripts 0, 2, 5, 6

### 2. STAMP Environment (UV/venv)
- **Purpose**: Feature extraction and model training using STAMP framework
- **Python version**: 3.11
- **Location**: `/lab/barcheese01/mdiberna/ARGO-DeepMSI/STAMP/.venv`
- **Activation**: `source /lab/barcheese01/mdiberna/ARGO-DeepMSI/STAMP/.venv/bin/activate`
- **Setup**: Run `stamp_v2_setup.sh` which clones STAMP repo and installs via `uv sync --all-extras`
- **Used in**: Scripts 1, 3, 4
- **Critical**: Requires `export HF_HOME=/lab/barcheese01/mdiberna/ARGO-DeepMSI/.huggingface_cache` before use
- **Hugging Face models**: Requires `hf auth login` for gated models (H-optimus)

### 3. TRIDENT Environment (Conda)
- **Purpose**: Alternative feature extraction using TRIDENT
- **Python version**: 3.10
- **Setup**: `conda create -n trident-env python=3.10 && pip install git+https://github.com/mahmoodlab/trident.git`

### 4. HistoBistro Environment (Conda)
- **Purpose**: Model comparison and benchmarking
- **Location**: `HistoBistro/` subdirectory (cloned from GitHub)
- **Activation**: `conda activate histobistro`
- **Setup**: `conda env create --file HistoBistro/environment_simple.yaml`
- **Used in**: Script 7

## Essential Commands

### Development Workflow

```bash
# Step 0: Data collection from REDCap
conda activate argo
python scripts/0.prepare.py

# Step 1: Feature extraction via STAMP (SLURM job array for 6 sites)
# Must be in STAMP environment for preprocessing
sbatch scripts/1.preprocess.sh  # Array job 0-5, uses config files per site

# Step 2: Validate preprocessing
conda activate argo
python scripts/2.preprocess_eval.py

# Step 3: Consolidate features and run cross-validation
sbatch scripts/3.cross_validation.sh

# Step 4: Generate statistical analysis
sbatch scripts/4.statistics.sh

# Optional: HistoBistro comparison
conda activate argo
python scripts/6.prepare_histobistro.py
sbatch scripts/7.histobistro.sh
```

### STAMP-Specific Commands

```bash
# Initialize STAMP config for a site
stamp --config configs/<encoder>/<site>.yaml init

# Run preprocessing manually (usually done via SLURM)
stamp --config configs/<encoder>/<site>.yaml preprocess

# Cross-validation
stamp --config configs/<encoder>/config_all.yaml crossval

# Generate statistics
stamp --config configs/<encoder>/config_all.yaml statistics

# Generate heatmaps
stamp --config configs/<encoder>/config_all.yaml heatmaps
```

## Architecture and Data Flow

### Multi-Site Processing Pattern

The pipeline processes slides from 6 independent sites, then consolidates for cross-validation:
- **Site-specific repos**: `data/{OAUTHC, LUTH, LASUTH, UITH, retrospective_msk, retrospective_oau}/`
- **Consolidated repo**: `data/all/`

Each site has its own YAML config in `configs/<encoder>/config_<SITE>.yaml`.

### Data Pipeline Flow

```
REDCap API → 0.prepare.py → tables/0/{clinical,slide}_table.csv
                                ↓
            1.preprocess.sh (per-site) → data/{SITE}/features/
                                ↓
            2.preprocess_eval.py → tables/2/{SITE}_{clinical,slide}_table.csv
                                ↓
            3.cross_validation.sh → consolidates to data/all/features/
                                   → data/all/results/crossval/split-{0,1,2}/
                                ↓
            4.statistics.sh → data/all/results/statistics/
```

### Directory Structure Conventions

- `data/{SITE}/raw/`: Original SVS whole slide images
- `data/{SITE}/features/{encoder}/`: Extracted features per encoder (e.g., `xiyuewang-ctranspath-7c998680-02627079/`)
- `data/{SITE}/results/`: Model training outputs and checkpoints
- `data/{SITE}/.cache/`: STAMP preprocessing cache
- `tables/0/`: Initial clinical and slide tables from REDCap
- `tables/2/`: Post-preprocessing validated tables (split by site and consolidated)
- `configs/{encoder}/`: STAMP YAML configs per encoder type (ctranspath, h-optimus-0, h-optimus-1)

## STAMP Configuration Structure

STAMP configs are YAML files with these key sections:

```yaml
crossval:
  output_dir: "path/to/results/crossval"
  clini_table: "path/to/clinical_table.csv"
  feature_dir: "path/to/features/{encoder}/"
  slide_table: "path/to/slide_table.csv"
  ground_truth_label: "isMSIH"  # Column name in clinical table
  patient_label: "PATIENT"
  filename_label: "FILENAME"
  categories: ["MSI-H", "MSS"]
  n_splits: 3

statistics:
  output_dir: "path/to/results/statistics"
  ground_truth_label: "isMSIH"
  true_class: "MSI-H"
  pred_csvs:
    - "path/to/crossval/split-0/patient-preds.csv"
    - "path/to/crossval/split-1/patient-preds.csv"
    - "path/to/crossval/split-2/patient-preds.csv"

heatmaps:
  output_dir: "path/to/results/heatmaps"
  feature_dir: "path/to/features/{encoder}/"
  wsi_dir: "path/to/raw/"
  checkpoint_path: "path/to/results/training/model.ckpt"
  topk: 10
  bottomk: 10
```

## SLURM Job Configuration Patterns

Scripts use SLURM for HPC execution:

- **Preprocessing** (1.preprocess.sh): Array job `--array=0-5` for 6 sites, requires GPU (`--gres=gpu:1`), 12h runtime
- **Cross-validation** (3.cross_validation.sh): Single job, GPU required, consolidates features via `rsync` before running
- **Statistics** (4.statistics.sh): CPU-only, short runtime (1h), no GPU needed
- **HistoBistro** (7.histobistro.sh): Uses `histobistro` conda environment, runs from `HistoBistro/` subdirectory

All SLURM scripts:
1. Source `~/.bashrc`
2. Activate appropriate conda/venv environment
3. Print GPU info via `nvidia-smi`
4. Test PyTorch CUDA availability
5. Verify config file exists before running

## Critical Environment Variables

```bash
# Required for STAMP to find Hugging Face models
export HF_HOME="/lab/barcheese01/mdiberna/ARGO-DeepMSI/.huggingface_cache"

# Required for CUDA compilation (if needed)
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
```

## REDCap Integration

The project fetches patient data from REDCap API. Required setup:

1. Create `.env` file in project root:
   ```
   REDCAP_API_TOKEN=your_token_here
   REDCAP_API_URL=https://redcap.oauife.edu.ng/api/
   ```

2. Script `0.prepare.py` handles:
   - Fetching patient records via POST to REDCap API
   - Extracting MSI status from prospective (cmo_msi_status) vs retrospective (msi_status_mmr) fields
   - Mapping batch_number to differentiate prospective (batch != 1,2) vs retrospective (batch = 1,2)
   - Loading Halo Link CSV exports (`halo_link_*.csv`)
   - Generating clinical and slide tables with MSI labels

## Feature Extractors

STAMP supports multiple feature extractors (located in `STAMP/src/stamp/encoding/encoder/`):
- CTransPath (ctranspath)
- H-optimus-0, H-optimus-1 (requires Hugging Face authentication)
- CHIEF, COBRA, EAGLE, GigaPath, Madeleine, PRISM, TITAN

Configs are organized by encoder in `configs/{encoder}/`.

## Cross-Validation Strategy

- **n_splits**: 3 (defined in YAML configs)
- **Output structure**: `data/all/results/crossval/split-{0,1,2}/patient-preds.csv`
- **Statistics aggregation**: Combines predictions from all splits to generate ROC curves, AUROC/AUPRC with 95% CI

## Notes for Development

- The script numbering (0, 1, 2, 3, 4, 6, 7) directly corresponds to workflow steps
- Step 5 exists for embedding visualizations (h-optimus-0 specific)
- Always verify which environment a script requires before running
- STAMP is installed as a local clone in `STAMP/` subdirectory, not via pip globally
- HistoBistro is also a local clone in `HistoBistro/` subdirectory
- AWS credentials should be configured system-wide using `aws configure` (see AWS.md for data transfer instructions)
- When adding new sites, create a new config in `configs/<encoder>/config_<NEWSITE>.yaml` and add to the sites array in `1.preprocess.sh`
