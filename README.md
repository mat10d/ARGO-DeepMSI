# ARGO-DeepMSI: MSI Prediction Pipeline

Multi-model pipeline for microsatellite instability (MSI) prediction from whole slide images of colorectal cancer.

## Pipeline Overview

8-stage pipeline from raw slides to biomarker prediction:

1. **Data Ingestion** - REDCap + Halo metadata → clinical/slide tables
2. **Quality Control** - Filter low-quality slides (placeholder)
3. **Feature Extraction** - Extract features using STAMP models
4. **Feature Validation** - Verify extraction success
5. **Baseline Testing** - Validate with pre-trained model (HistoBistro)
6. **MIL Training** - Train models with cross-validation
7. **Statistics** - Calculate metrics (AUROC, AUPRC, CI)
8. **Visualization** - Generate heatmaps and figures

---

## Quick Start

### 1. Setup Environments

See **[environments/README.md](environments/README.md)** for complete installation instructions.

```bash
# ARGO environment (data processing)
conda env create -f environments/argo.yml
conda activate argo
pip install -e .

# STAMP environment (feature extraction) - MUST install on compute node!
srun --partition=nvidia-2080ti-20 --gres=gpu:1 --mem=64G --time=2:00:00 --pty bash
cd STAMP && MAX_JOBS=4 uv sync --extra build --extra gpu && exit

# Authenticate with Hugging Face (for gated models)
source STAMP/.venv/bin/activate
hf auth login
```

### 2. Run Pipeline

```bash
# Stage 1: Data Ingestion
conda activate argo
python scripts/1_data_ingestion.py

# Stage 2: Quality Control (placeholder)
python scripts/2_quality_control.py

# Stage 3: Feature Extraction (ALL models)
bash scripts/3_feature_extraction_all.sh

# Or single model:
# sbatch scripts/3_feature_extraction.sh ctranspath

# Stage 4: Feature Validation
conda activate argo
python scripts/4_feature_validation.py

# Stage 5: Baseline Testing (optional)
conda activate histobistro
python scripts/5_baseline_testing.py

# Stage 6: MIL Training
source STAMP/.venv/bin/activate
sbatch scripts/6_mil_training.sh ctranspath

# Stage 7: Statistics
conda activate argo
python scripts/7_statistics.py

# Stage 8: Visualization
python scripts/8_visualization.py
```

---

## Script Reference

| Stage | Script | Environment | Description |
|-------|--------|-------------|-------------|
| 1 | `1_data_ingestion.py` | ARGO | Fetch REDCap + Halo data |
| 2 | `2_quality_control.py` | ARGO | Slide QC (placeholder) |
| 3 | `3_feature_extraction_all.sh` | STAMP | Extract features (all models) |
| 3 | `3_feature_extraction.sh` | STAMP | Extract features (single model) |
| 4 | `4_feature_validation.py` | ARGO | Verify feature extraction |
| 5 | `5_baseline_testing.py` | HistoBistro | Pre-trained model validation |
| 6 | `6_mil_training.sh` | STAMP | Cross-validation training |
| 7 | `7_statistics.py` | ARGO | Performance metrics |
| 8 | `8_visualization.py` | ARGO | Heatmaps and figures |

**Helper Scripts:**
- `scripts/test_model_access.py` - Test if model is accessible
- `scripts/generate_config.py` - Generate STAMP configs from templates

---

## Feature Extraction Models

Stage 3 processes slides with 12 STAMP models (newer versions preferred):

**No authentication:**
- ctranspath
- plip
- dinobloom
- chief-ctranspath

**Gated (requires HF authentication):**
- virchow2 (vs virchow)
- uni2 (vs uni)
- conch1_5 (vs conch)
- gigapath
- h-optimus-0
- h-optimus-1
- mstar
- musk

---

## Output Structure

```
results/
├── stage1_data_ingestion/       # Clinical and slide tables
├── stage2_qc/                   # QC reports
├── stage3_features/             # Extracted features
│   ├── ctranspath/
│   │   ├── OAUTHC/
│   │   ├── LUTH/
│   │   └── ...
│   ├── virchow2/
│   └── ...
├── stage4_feature_validation/   # Feature QC reports
├── stage5_baseline/             # HistoBistro results
├── stage6_training/             # MIL training outputs
├── stage7_statistics/           # Performance metrics
└── stage8_visualization/        # Figures and heatmaps
```

---

## Documentation

- **[environments/README.md](environments/README.md)** - Environment setup and installation
- **[QUICKSTART_FEATURE_EXTRACTION.md](QUICKSTART_FEATURE_EXTRACTION.md)** - Feature extraction guide
- **[FEATURE_EXTRACTION_GUIDE.md](FEATURE_EXTRACTION_GUIDE.md)** - Technical details
- **[CLAUDE.md](CLAUDE.md)** - AI assistant instructions
- **[TODO.md](TODO.md)** - Development roadmap

---

## Key Design Principles

1. **Modular Package**: Heavy logic in `argo_deepmsi/` package, thin scripts in `scripts/`
2. **Template-Based Configs**: STAMP configs generated from templates (not hardcoded per model)
3. **External Dependencies**: STAMP and HistoBistro cloned locally, never modified
4. **Multi-Environment**: Three separate environments (ARGO, STAMP, HistoBistro)

---

## Troubleshooting

**STAMP installation fails:**
- Must install on compute node (not head node) - see [environments/README.md](environments/README.md)

**Feature extraction fails:**
- Run `python scripts/test_model_access.py <model>` to verify model is accessible
- Check Hugging Face authentication: `huggingface-cli whoami`

**GPU not detected:**
- Check: `nvidia-smi` and `python -c "import torch; print(torch.cuda.is_available())"`

For more troubleshooting, see [environments/README.md](environments/README.md).

---

## Citation

If using this pipeline, please cite:

```bibtex
@article{argo_deepmsi,
  title={ARGO-DeepMSI: Multi-Model MSI Prediction from Histopathology},
  author={Your Name et al.},
  year={2025}
}
```

Also cite:
- **STAMP**: https://github.com/KatherLab/STAMP
- **HistoBistro**: https://github.com/peng-lab/HistoBistro
