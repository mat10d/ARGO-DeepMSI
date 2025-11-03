# ARGO-DeepMSI: Multi-Model MSI Prediction Pipeline

A clean, modular pipeline for microsatellite instability (MSI) prediction from whole slide images using multiple foundation models.

## Overview

ARGO-DeepMSI is an end-to-end pipeline for predicting MSI status from H&E-stained colorectal cancer slides. The pipeline:
- Processes slides from multiple sites (6 cohorts)
- Extracts features using 18+ foundation models (via STAMP)
- Validates data quality with pre-trained models (HistoBistro baseline)
- Trains MIL classifiers with k-fold cross-validation
- Generates comprehensive statistics and visualizations

**Branch**: Currently on `overhaul` - clean refactor with modular `utils/` structure

---

## Quick Start

### 1. Clone Repository

```bash
git clone <your-repo-url>
cd ARGO-DeepMSI
git checkout overhaul  # Development branch with clean structure
```

### 2. Clone External Dependencies

**STAMP** (Feature extraction + MIL training):
```bash
git clone https://github.com/KatherLab/STAMP.git
```

**HistoBistro** (Baseline validation):
```bash
git clone https://github.com/peng-lab/HistoBistro.git
```

**Note**: We keep STAMP and HistoBistro as-is (never modify) to stay up-to-date with upstream changes.

### 3. Set Up Environments

Run the automated setup script:
```bash
bash environments/setup.sh
```

Or manually:
```bash
# ARGO environment (main pipeline)
conda env create -f environments/argo.yml
conda activate argo

# STAMP environment (feature extraction)
cd STAMP
rm -rf ~/.triton  # Clear cache
uv sync --extra build --extra gpu
cd ..

# HistoBistro environment (baseline)
conda env create -f HistoBistro/environment_simple.yaml
```

### 4. Configure Credentials

Create `.env` file:
```bash
# Hugging Face cache
export HF_HOME=/path/to/ARGO-DeepMSI/.huggingface_cache

# REDCap API (update with your credentials)
export REDCAP_API_URL=https://redcap.oauife.edu.ng/api/
export REDCAP_API_TOKEN=your_token_here
```

Login to Hugging Face (required for gated models):
```bash
huggingface-cli login
```

### 5. Run Pipeline

```bash
# Stage 1: Data Ingestion
conda activate argo
python scripts/1_data_ingestion.py

# Stage 2: Quality Control
python scripts/2_quality_control.py

# Stage 3: Feature Extraction
source STAMP/.venv/bin/activate
sbatch scripts/slurm/3_extract_features.sh ctranspath

# Stage 4: Feature Validation
conda activate argo
python scripts/4_feature_validation.py

# Stage 5: Baseline Testing
conda activate histobistro
python scripts/5_baseline_testing.py

# Stage 6: MIL Training
source STAMP/.venv/bin/activate
sbatch scripts/slurm/6_mil_training.sh ctranspath

# Stage 7-8: Statistics & Visualization
conda activate argo
python scripts/7_statistics.py
python scripts/8_visualization.py
```

---

## Pipeline Stages

### Stage 1: Data Ingestion
- **Input**: SVS files, REDCap metadata
- **Output**: `tables/clinical_table.csv`, `tables/slide_table.csv`
- **Environment**: ARGO

### Stage 2: Quality Control
- **Input**: WSI files, slide table
- **Output**: `results/qc/qc_report.csv`
- **Environment**: ARGO
- **Purpose**: Filter low-quality slides before feature extraction

### Stage 3: Feature Extraction
- **Input**: QC-passed slides
- **Output**: `data/{SITE}/features/{MODEL}/`
- **Environment**: STAMP
- **Models**: 18+ foundation models (CTransPath, Virchow2, UNI2, CONCH, etc.)

### Stage 4: Feature Validation
- **Input**: Extracted features
- **Output**: `results/feature_validation/extraction_report.csv`
- **Environment**: ARGO
- **Purpose**: Identify which slides passed/failed extraction and why

### Stage 5: Baseline Testing
- **Input**: CTransPath features
- **Output**: `results/baseline/predictions.csv`
- **Environment**: HistoBistro
- **Purpose**: Validate data quality with pre-trained model (published 0.99 NPV)

### Stage 6: MIL Training
- **Input**: Features from any model
- **Output**: `data/all/results/crossval/split-{0,1,2}/`
- **Environment**: STAMP
- **Purpose**: Train MIL classifiers with 3-fold cross-validation

### Stage 7: Statistics
- **Input**: Cross-validation predictions
- **Output**: `results/statistics/metrics.csv`
- **Environment**: ARGO
- **Purpose**: Calculate AUROC, AUPRC, sensitivity, NPV with 95% CI

### Stage 8: Visualization
- **Input**: Statistics, trained models
- **Output**: `results/figures/`
- **Environment**: ARGO
- **Purpose**: ROC curves, heatmaps, embeddings, top tiles

---

## Directory Structure

```
ARGO-DeepMSI/
├── README.md                    # This file
├── TODO.md                      # Development roadmap
├── CLAUDE.md                    # AI assistant instructions
│
├── utils/                       # Core Python utilities
│   ├── data_ingestion.py
│   ├── quality_control.py
│   ├── feature_extraction.py
│   ├── feature_validation.py
│   ├── baseline_testing.py
│   ├── training.py
│   ├── statistics.py
│   ├── visualization.py
│   ├── config_utils.py
│   └── io_utils.py
│
├── scripts/                     # Executable scripts
│   ├── 1_data_ingestion.py
│   ├── 2_quality_control.py
│   ├── 3_feature_extraction.py
│   ├── 4_feature_validation.py
│   ├── 5_baseline_testing.py
│   ├── 6_mil_training.py
│   ├── 7_statistics.py
│   ├── 8_visualization.py
│   └── slurm/                   # SLURM batch scripts
│
├── environments/                # Environment configs
│   ├── argo.yml
│   ├── setup.sh
│   └── README.md
│
├── configs/                     # STAMP model configs
│   ├── ctranspath/
│   ├── virchow2/
│   └── ...
│
├── data/                        # Processed data (gitignored)
│   ├── OAUTHC/
│   ├── LUTH/
│   ├── LASUTH/
│   ├── UITH/
│   ├── retrospective_msk/
│   ├── retrospective_oau/
│   └── all/
│
├── tables/                      # Clinical/slide metadata
├── results/                     # Pipeline outputs
├── logs/                        # Execution logs
│
├── STAMP/                       # External: git clone
└── HistoBistro/                 # External: git clone
```

---

## Available Feature Extractors (STAMP v2.3)

### Priority Tier 1 (Clinical-grade)
- **Virchow2** - Microsoft 2024, best general performance
- **UNI2** - Mahmood Lab 2024, widely validated
- **CONCHv1.5** - Vision-language model
- **Gigapath** - Large-scale pathology FM

### Priority Tier 2 (Specialized)
- **MUSK** - Multi-scale understanding
- **mSTAR** - Multi-task pre-training
- **DinoBloom** - Self-supervised DINO
- **PLIP** - Pathology-language integration

### Currently Available
- **CTransPath** ✓ (baseline features exist)
- **H-optimus-0** ✓
- **H-optimus-1** ✓

**Total**: 18 extractors available in STAMP v2.3

---

## Environment Activation Quick Reference

| Stage | Environment | Command |
|-------|-------------|---------|
| 1. Data Ingestion | ARGO | `conda activate argo` |
| 2. Quality Control | ARGO | `conda activate argo` |
| 3. Feature Extraction | STAMP | `source STAMP/.venv/bin/activate` |
| 4. Feature Validation | ARGO | `conda activate argo` |
| 5. Baseline Testing | HistoBistro | `conda activate histobistro` |
| 6. MIL Training | STAMP | `source STAMP/.venv/bin/activate` |
| 7. Statistics | ARGO | `conda activate argo` |
| 8. Visualization | ARGO | `conda activate argo` |

---

## Key Design Principles

1. **Modular Structure**: Core logic in `utils/`, orchestration in `scripts/`
2. **External Dependencies**: STAMP and HistoBistro cloned locally, never modified
3. **Clean Environments**: Three separate environments for different stages
4. **Reproducibility**: Config-driven, no hardcoded paths
5. **Documentation First**: Update README as changes are made

---

## Development Workflow

### Current Branch: `overhaul`

This branch contains the cleaned, refactored pipeline. **Do not merge to `main` until**:
- ✅ All 8 stages implemented
- ✅ End-to-end test passes
- ✅ Feature validation works
- ✅ HistoBistro baseline validates data (close to 0.99 NPV)
- ✅ Documentation complete

### Making Changes

1. All custom code goes in `utils/` or `scripts/`
2. Never modify `STAMP/` or `HistoBistro/` directories
3. Update this README when adding features
4. Test changes on small subset before full run

### Updating External Dependencies

```bash
# Update STAMP
cd STAMP && git pull && uv sync --extra build --extra gpu && cd ..

# Update HistoBistro
cd HistoBistro && git pull && cd ..
```

---

## Troubleshooting

### STAMP Installation Issues

```bash
# Clear caches and reinstall
rm -rf ~/.triton
cd STAMP
uv cache clean flash_attn mamba-ssm causal_conv1d
uv sync --extra build
uv sync --extra build --extra gpu
```

### GPU Not Detected

```bash
# Check CUDA
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Set CUDA path if needed
export CUDA_HOME=/usr/local/cuda-12.6
```

### Feature Extraction Fails

1. Check Stage 4 (Feature Validation) output
2. Review logs in `logs/feature_extraction/`
3. Verify slide quality in Stage 2 (QC)
4. Check STAMP preprocessing cache: `data/{SITE}/.cache/`

---

## Citation

If you use this pipeline, please cite:

```bibtex
@article{your_paper,
  title={ARGO-DeepMSI: Multi-Model MSI Prediction from Histopathology},
  author={Your Name et al.},
  journal={Journal Name},
  year={2025}
}
```

Also cite STAMP and HistoBistro:
- STAMP: https://github.com/KatherLab/STAMP
- HistoBistro: https://github.com/peng-lab/HistoBistro

---

## License

[Specify your license]

---

## Contact

For questions or issues:
- Open an issue on GitHub
- Contact: [your email]

---

**Last Updated**: 2025-11-03 (overhaul branch)
