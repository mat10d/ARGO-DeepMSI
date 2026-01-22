# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

ARGO-DeepMSI is a simplified pipeline for MSI prediction from whole slide images using [LazySlide](https://github.com/rendeirolab/LazySlide). The architecture uses a single Python environment and a unified CLI interface.

## Architecture

```
argo_deepmsi/
├── cli.py              # Single entry point (typer-based CLI)
├── data_ingestion.py   # REDCap + Halo data loading
├── feature_extraction.py # LazySlide feature extraction
├── visualization.py    # UMAP, t-SNE, slide visualization
├── training.py         # Classifiers (LogReg, RF, SVM, MLP)
└── io_utils.py         # Path management
```

## Essential Commands

```bash
# Install
pip install -e .

# CLI commands
argo --help              # Show all commands
argo models              # List available feature extraction models
argo ingest              # Data ingestion from REDCap
argo extract <table>     # Extract features from slides
argo aggregate <dir>     # Aggregate patch → slide embeddings
argo visualize           # Generate visualizations
argo train <embeddings>  # Train classifiers
argo run <slides> <clinical>  # Full pipeline
```

## Key Dependencies

- **lazyslide**: Core library for WSI processing and feature extraction
- **typer/rich**: CLI interface
- **pandas/numpy**: Data handling
- **scikit-learn**: Simple classifiers
- **torch**: MLP training (optional)

## LazySlide Usage Pattern

```python
import lazyslide as zs

# Load slide
wsi = zs.WSI("path/to/slide.svs")

# Preprocessing
zs.pp.find_tissues(wsi)
zs.pp.tile_tissues(wsi, tile_px=256, mpp=0.5)

# Feature extraction (any supported model)
zs.tl.feature_extraction(wsi, model="uni2", amp=True)

# Access features as AnnData
features = wsi["uni2_tiles"]
```

## Supported Models

**Patch-level (no auth):** resnet50, ctranspath, plip

**Patch-level (HF auth required):** uni, uni2, virchow, virchow2, conch, gigapath, h-optimus-0, h-optimus-1

**Slide-level aggregators:** prism, threads

## Output Structure

```
results/
├── data/           # clinical_table.csv, slide_table.csv
├── features/       # .h5ad files per model (patch-level)
├── embeddings/     # .npy + metadata.csv per model (slide-level)
├── visualizations/ # Plots and figures
└── models/         # Trained classifiers
```

## Environment Variables

```bash
export HF_HOME="/path/to/.huggingface_cache"  # HuggingFace cache location
export HF_HUB_OFFLINE=1                       # For offline compute nodes
```

## REDCap Integration

Create `.env` file in project root:
```
REDCAP_API_TOKEN=your_token
REDCAP_API_URL=https://redcap.example.com/api/
```

The `data_ingestion.py` module handles:
- Fetching patient records via POST to REDCap API
- Extracting MSI status from prospective (cmo_msi_status) vs retrospective (msi_status_mmr) fields
- Loading Halo Link CSV exports (`halo_link_*.csv`)
- Generating clinical and slide tables with MSI labels

## Development Notes

- All paths are managed via `io_utils.py` - use helper functions, not hardcoded paths
- Feature extraction saves as AnnData (.h5ad) files for scverse compatibility
- Embeddings are saved as numpy arrays with metadata CSV
- CLI is built with typer, uses rich for nice terminal output
- Training module supports sklearn classifiers and a simple PyTorch MLP
