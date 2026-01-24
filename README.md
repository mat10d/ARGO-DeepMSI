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

Extract features from ALL models in 3 parallel groups:

```bash
sbatch scripts/extract.sh
```

This splits slides into 3 groups (~270 slides each):
- **Group 1**: slides 1-270 with all 12 models
- **Group 2**: slides 271-540 with all 12 models
- **Group 3**: slides 541-808 with all 12 models

Each slide is preprocessed once, all models extracted in one pass.

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

### Feature Extraction

Edit the `MODELS` array in `scripts/extract.sh` to select which models to extract:

```bash
# scripts/extract.sh
MODELS=(
    "uni2"
    "virchow2"
    "plip"
    # Add or remove models here
)
```

To change the number of parallel groups, update both:
1. `#SBATCH --array=0-N` (where N = num_groups - 1)
2. `NUM_GROUPS=N` variable in the script

For example, to use 5 groups instead of 3:
```bash
#SBATCH --array=0-4
NUM_GROUPS=5
```

### Aggregation and Training

`scripts/aggregate.sh` and `scripts/train.sh` use model-based arrays:
- For N models/embeddings: `--array=0-$((N-1))%M`
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
