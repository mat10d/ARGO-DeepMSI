# ARGO-DeepMSI

MSI prediction from whole slide images using [LazySlide](https://github.com/rendeirolab/LazySlide).

## Installation

```bash
# Create conda environment with Python, uv, and PyTorch+CUDA
conda create -n argo -c pytorch -c nvidia -c conda-forge \
  python=3.11 uv pip pytorch pytorch-cuda=12.1 -y

# Activate and install dependencies
conda activate argo
uv pip install -e .

# Configure credentials (HuggingFace token for gated models)
cp .env.template .env
# Edit .env with your HF_TOKEN
```

## Running the Pipeline

### 1. Data Ingestion

```bash
# Generate clinical_table.csv and slide_table.csv from REDCap
argo ingest
```

Outputs:
- `results/data/clinical_table.csv` - Patient MSI labels
- `results/data/slide_table.csv` - Slide paths and metadata

### 2. Feature Extraction

Extract features from slides and save to zarr format:

```bash
# Non-gated models (no auth required)
sbatch scripts/extract_all_models.sh

# Gated models (requires HF_TOKEN in .env)
sbatch scripts/extract_gated_models.sh
```

Monitor jobs:
```bash
squeue -u $USER
tail -f scripts/logs/extract_*.out
```

Outputs: `data/SITE/slide.zarr/tables/{model}_tiles/` for each slide and model

### 3. Aggregation

Aggregate patch features to slide-level embeddings:

```bash
# Simple pooling (mean, max, median, sum)
argo aggregate plip,ctranspath --method mean

# Neural slide encoders (requires specific base models)
argo aggregate virchow --method prism
argo aggregate conch_v1.5 --method titan
```

Outputs: `results/embeddings/{model}_{method}/`
- `embeddings.npy` - Slide embeddings matrix
- `metadata.csv` - Slide metadata (patient_id, site, etc.)

### 4. Training

Train classifiers on slide embeddings:

```bash
argo train results/embeddings/plip_mean
```

Outputs: `results/models/{embedding_type}/`
- `classifier_comparison.csv` - Performance metrics (AUROC, accuracy)
- `training_data.csv` - Patient-slide-label mappings

## Available Models

Run `argo models` to see all available models and aggregation methods.

**Non-gated** (no auth): plip, ctranspath, phikon, phikonv2, resnet50

**Gated** (HF auth required): uni2, virchow2, h-optimus-0, gigapath, conch, hibou-b

**Aggregation methods**:
- Simple: mean, max, median, sum
- Neural: prism (virchow), titan (conch_v1.5), chief, madeleine

## References

- **LazySlide**: https://github.com/rendeirolab/LazySlide
- **LazySlide Paper**: https://doi.org/10.1101/2025.05.28.656548
