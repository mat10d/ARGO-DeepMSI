# ARGO-DeepMSI

MSI prediction from whole slide images using [LazySlide](https://github.com/rendeirolab/LazySlide).

## Installation

```bash
# Create conda environment
conda create -n argo -c pytorch -c nvidia -c conda-forge \
  python=3.11 uv pip pytorch pytorch-cuda=12.1 -y

# Install dependencies
conda activate argo
uv pip install -e .

# Configure HuggingFace token (for gated models)
cp .env.template .env
# Edit .env and add your HF_TOKEN
```

## Running the Pipeline

### 1. Data Ingestion

```bash
argo ingest
```

Creates:
- `results/data/clinical_table.csv` - Patient MSI labels
- `results/data/slide_table.csv` - Slide paths and metadata

### 2. Feature Extraction

Edit `scripts/extract.sh` to select which models to run, then submit:

```bash
sbatch scripts/extract.sh
```

Monitor progress:
```bash
squeue -u $USER
tail -f scripts/logs/extract_*.out
```

Creates: `data/SITE/slide.zarr/tables/{model}_tiles/` for each slide

### 3. Aggregation

Edit `scripts/aggregate.sh` to match your extracted models, then submit:

```bash
sbatch scripts/aggregate.sh
```

Creates: `results/embeddings/{model}_{method}/`
- `embeddings.npy` - Slide embedding matrix
- `metadata.csv` - Slide metadata

### 4. Training

Edit `scripts/train.sh` to match your embeddings, then submit:

```bash
sbatch scripts/train.sh
```

Creates: `results/models/{embedding_type}/`
- `classifier_comparison.csv` - Performance metrics
- `training_data.csv` - Patient-slide-label mappings

## Customizing Scripts

All scripts have a `MODELS` or `EMBEDDINGS` array at the top that you can edit:

```bash
# scripts/extract.sh
MODELS=(
    "plip"
    "uni2"
    # Add or remove models here
)
```

Update the SLURM `--array` parameter to match:
- For N models: `--array=0-$((N-1))%M`
- M = max concurrent jobs

## Available Models

**Non-gated** (no auth): plip, ctranspath, phikon, phikonv2, resnet50

**Gated** (requires HF_TOKEN): uni2, virchow2, h-optimus-0, gigapath, conch, hibou-b

**Aggregation methods**: mean, max, median, sum

For neural aggregators (prism, titan), use the CLI:
```bash
argo aggregate virchow --method prism
```

## References

- **LazySlide**: https://github.com/rendeirolab/LazySlide
- **Paper**: https://doi.org/10.1101/2025.05.28.656548
