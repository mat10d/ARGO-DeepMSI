# ARGO-DeepMSI

MSI prediction from whole slide images using [LazySlide](https://github.com/rendeirolab/LazySlide).

## Overview

Simplified pipeline for microsatellite instability (MSI) prediction from colorectal cancer histopathology:

1. **Data Ingestion** - REDCap + Halo metadata → clinical/slide tables
2. **Feature Extraction** - Extract features using foundation models (UNI2, Virchow2, etc.)
3. **Visualization** - UMAP/t-SNE embeddings, slide visualization
4. **Training** - Simple classifiers (LogReg, RF, SVM) or lightweight MLP

## Installation

```bash
# 1. Create conda environment with Python, uv, and PyTorch+CUDA
conda create -n argo -c pytorch -c nvidia -c conda-forge \
  python=3.11 uv pip pytorch pytorch-cuda=12.1 -y

# 2. Activate and install dependencies
conda activate argo
uv pip install -e .

# 3. Configure credentials
cp .env.template .env
# Edit .env with your HuggingFace token and (optionally) REDCap credentials
```

## Quick Start

### CLI Interface

```bash
# List available models
argo models

# Data ingestion (from REDCap)
argo ingest

# Extract features from slides
argo extract slide_table.csv --model uni2 --model virchow2

# Aggregate patch features to slide embeddings
argo aggregate results/features/uni2 uni2 --method mean

# Visualize embeddings
argo visualize --embeddings results/embeddings/uni2

# Train classifiers
argo train results/embeddings/uni2 clinical_table.csv

# Full pipeline
argo run slide_table.csv clinical_table.csv --model uni2
```

### Python API

```python
import lazyslide as zs
from wsidata import open_wsi
from argo_deepmsi import feature_extraction, visualization, training

# Load slide
wsi = open_wsi("path/to/slide.svs")

# Process slide
zs.pp.find_tissues(wsi)
zs.pp.tile_tissues(wsi, tile_px=256, mpp=0.5)

# Extract features with any model
zs.tl.feature_extraction(wsi, model="uni2", amp=True)

# Access features
features = wsi["uni2_tiles"]
```

## Supported Models

### Patch-Level Extractors

| Model | Auth Required | Description |
|-------|--------------|-------------|
| resnet50 | No | ImageNet pretrained ResNet50 |
| ctranspath | No | CTransPath pathology foundation model |
| plip | No | PLIP vision-language model |
| uni / uni2 | Yes | UNI pathology foundation models |
| virchow / virchow2 | Yes | Virchow models (631M params) |
| conch | Yes | CONCH vision-language model |
| gigapath | Yes | GigaPath foundation model |
| h-optimus-0/1 | Yes | H-Optimus models |

### Slide-Level Aggregators

| Model | Description |
|-------|-------------|
| prism | PRISM slide-level aggregator |
| threads | THREADS slide-level model |

## Project Structure

```
ARGO-DeepMSI/
├── argo_deepmsi/           # Core Python package
│   ├── cli.py              # Typer-based CLI entry point
│   ├── data_ingestion.py   # REDCap + Halo data loading
│   ├── feature_extraction.py # LazySlide feature extraction (27 models)
│   ├── visualization.py    # UMAP, t-SNE, slide visualization
│   ├── training.py         # Classifiers (LogReg, RF, SVM, MLP)
│   └── io_utils.py         # Path management & logging
├── scripts/                # HPC execution scripts
│   ├── run_all_models.py   # Multi-model extraction orchestrator
│   └── aggregate_and_visualize.py # Aggregation pipeline
├── .env.template           # Environment variables template
├── pyproject.toml          # Package config & dependencies
├── CLAUDE.md               # Claude Code instructions
└── README.md

# Auto-generated (gitignored):
├── .huggingface_cache/     # Downloaded models (~10-50GB)
├── results/                # All outputs
├── logs/                   # Runtime logs
└── old/                    # Archived code & data
```

## Output Structure

All outputs are auto-generated in `results/` (gitignored):

```
results/                    # Created automatically on first run
├── data/                   # Clinical and slide tables
│   ├── clinical_table.csv
│   └── slide_table.csv
├── features/               # Patch-level features (.h5ad)
│   ├── uni2/
│   ├── virchow2/
│   └── ...
├── embeddings/             # Slide-level embeddings
│   ├── uni2_mean/
│   │   ├── embeddings.npy
│   │   └── metadata.csv
│   ├── uni2_prism/
│   └── ...
├── visualizations/         # UMAP/t-SNE plots
└── models/                 # Trained classifiers
```

Logs are auto-generated in `logs/` (gitignored).

## HPC / Multi-Model Processing

For running across all models on an HPC cluster:

```bash
# List available model groups
python scripts/run_all_models.py --list-models

# Run all recommended models sequentially (interactive GPU session)
python scripts/run_all_models.py results/data/slide_table.csv --group recommended

# Submit SLURM array job (parallel, one model per GPU)
python scripts/run_all_models.py results/data/slide_table.csv --slurm --max-concurrent 4

# Run only non-gated models (no HF auth needed)
python scripts/run_all_models.py results/data/slide_table.csv --no-auth-only

# Test with small subset
python scripts/run_all_models.py results/data/slide_table.csv --models uni2 virchow2 --max-slides 10

# After extraction, aggregate and visualize all models
python scripts/aggregate_and_visualize.py --clinical results/data/clinical_table.csv
```

### Model Groups

| Group | Models |
|-------|--------|
| `no_auth` | resnet50, ctranspath, plip |
| `recommended` | uni2, virchow2, h-optimus-0, gigapath, ctranspath |
| `gated` | uni, uni2, virchow, virchow2, conch, gigapath, h-optimus-0/1 |
| `all_patch` | All patch-level extractors |

## Environment Variables

```bash
cp .env.template .env
# Edit .env with your credentials
```

⚠️ **Security**: Never commit `.env` to git!

## References

- **LazySlide**: https://github.com/rendeirolab/LazySlide
- **LazySlide Paper**: https://doi.org/10.1101/2025.05.28.656548
