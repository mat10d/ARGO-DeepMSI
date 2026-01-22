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
# Create environment
conda env create -f environments/argo.yml
conda activate argo

# Or with pip directly
pip install -e .

# For gated models (UNI, Virchow, etc.), authenticate with Hugging Face
huggingface-cli login
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
from argo_deepmsi import feature_extraction, visualization, training

# Load slide
wsi = zs.WSI("path/to/slide.svs")

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
├── argo_deepmsi/           # Core package
│   ├── cli.py              # CLI entry point
│   ├── data_ingestion.py   # REDCap + Halo data loading
│   ├── feature_extraction.py # LazySlide feature extraction
│   ├── visualization.py    # UMAP, t-SNE, slide viz
│   ├── training.py         # Simple classifiers + MLP
│   └── io_utils.py         # Path management
├── environments/
│   └── argo.yml            # Conda environment
├── pyproject.toml          # Package config
└── README.md
```

## Output Structure

```
results/
├── data/                   # Clinical and slide tables
│   ├── clinical_table.csv
│   └── slide_table.csv
├── features/               # Patch-level features (.h5ad)
│   ├── uni2/
│   ├── virchow2/
│   └── ...
├── embeddings/             # Slide-level embeddings
│   ├── uni2/
│   │   ├── embeddings.npy
│   │   └── metadata.csv
│   └── ...
├── visualizations/         # Plots and figures
└── models/                 # Trained classifiers
```

## Environment Variables

```bash
# Required for gated HuggingFace models
export HF_HOME="/path/to/.huggingface_cache"

# For offline use on compute nodes
export HF_HUB_OFFLINE=1

# REDCap credentials (in .env file)
REDCAP_API_TOKEN=your_token
REDCAP_API_URL=https://redcap.example.com/api/
```

## References

- **LazySlide**: https://github.com/rendeirolab/LazySlide
- **LazySlide Paper**: https://doi.org/10.1101/2025.05.28.656548
