# Feature Extraction Guide

## Overview

The ARGO-DeepMSI pipeline uses template-based configuration for flexible, model-agnostic feature extraction and training.

## Architecture

### Template-Based Configs (NEW ✅)

Instead of maintaining separate configs for every model × site combination, we use **templates** with runtime generation:

```
configs/
├── templates/
│   ├── preprocessing_site.yaml.template    # Feature extraction (per site)
│   └── training_all.yaml.template          # MIL training (all sites)
└── [old model-specific dirs kept for reference]
```

**Benefits:**
- ✅ Single template per stage (not N models × M sites configs)
- ✅ Model and site set dynamically at runtime
- ✅ Easy to add new models (no new config files needed)
- ✅ Consistent paths across all models

## Stage 3: Feature Extraction

### Quick Start

Extract features for a specific model across all sites:

```bash
# Single model, all sites (parallel via SLURM array)
sbatch scripts/3_feature_extraction.sh ctranspath

# Or for other models:
sbatch scripts/3_feature_extraction.sh h-optimus-0
sbatch scripts/3_feature_extraction.sh virchow2
sbatch scripts/3_feature_extraction.sh uni2
```

### How It Works

1. **SLURM array job** processes 6 sites in parallel (indices 0-5)
2. **For each site**, the script:
   - Generates config from template: `configs/templates/preprocessing_site.yaml.template`
   - Replaces placeholders: `${MODEL}`, `${SITE}`, `${BASE_DIR}`, `${DEVICE}`
   - Saves to: `.temp_configs/{MODEL}/config_{SITE}.yaml`
   - Runs: `stamp --config .temp_configs/{MODEL}/config_{SITE}.yaml preprocess`
3. **Features saved to**: `results/stage3_features/{MODEL}/{SITE}/`

### Available Models

STAMP supports 18+ models. Common ones:
- `ctranspath` - CTransPath (no HF auth required)
- `h-optimus-0` - H-optimus-0 (requires HF auth)
- `h-optimus-1` - H-optimus-1 (requires HF auth)
- `virchow2` - Virchow v2 (requires HF auth)
- `uni2` - UNI v2 (requires HF auth)
- `conch1_5` - CONCH v1.5 (requires HF auth)
- `gigapath` - Gigapath (requires HF auth)
- `plip` - PLIP
- `dinobloom` - DinoBloom

For gated models, authenticate first:
```bash
huggingface-cli login
```

### Sites Processed

Array job processes these 6 sites:
1. OAUTHC
2. LUTH
3. LASUTH
4. UITH
5. retrospective_msk
6. retrospective_oau

## Manual Config Generation

Generate configs outside of SLURM:

```bash
# Generate preprocessing config for a site
python scripts/generate_config.py \
    --template configs/templates/preprocessing_site.yaml.template \
    --output my_config.yaml \
    --model ctranspath \
    --site OAUTHC \
    --device cuda:0

# Generate training config
python scripts/generate_config.py \
    --template configs/templates/training_all.yaml.template \
    --output my_training_config.yaml \
    --model ctranspath \
    --site all \
    --n-splits 3
```

## Template Placeholders

### Preprocessing Template

| Placeholder | Description | Example |
|-------------|-------------|---------|
| `${BASE_DIR}` | Project root | `/lab/barcheese01/mdiberna/ARGO-DeepMSI` |
| `${MODEL}` | Feature extractor | `ctranspath`, `virchow2` |
| `${SITE}` | Site name | `OAUTHC`, `LUTH` |
| `${DEVICE}` | CUDA device | `cuda:0` |

### Training Template

| Placeholder | Description | Example |
|-------------|-------------|---------|
| `${BASE_DIR}` | Project root | `/lab/barcheese01/mdiberna/ARGO-DeepMSI` |
| `${MODEL}` | Feature extractor | `ctranspath` |
| `${N_SPLITS}` | CV folds | `3` |
| `${DEVICE}` | CUDA device | `cuda:0` |

## Output Structure

```
results/
├── stage3_features/          # Feature extraction outputs
│   ├── ctranspath/
│   │   ├── OAUTHC/          # Features for OAUTHC site
│   │   ├── LUTH/
│   │   ├── LASUTH/
│   │   ├── UITH/
│   │   ├── retrospective_msk/
│   │   ├── retrospective_oau/
│   │   └── all/             # Consolidated (for training)
│   ├── h-optimus-0/
│   │   └── [same structure]
│   └── virchow2/
│       └── [same structure]
├── stage6_training/          # MIL training outputs
│   └── {MODEL}/
│       └── all/
│           └── crossval/
├── stage7_statistics/        # Performance metrics
│   └── {MODEL}/
│       └── all/
└── stage8_visualization/     # Heatmaps
    └── {MODEL}/
        └── all/
            └── heatmaps/
```

## Adding New Models

To add a new model (e.g., `mstar`):

1. **No config creation needed!** Templates handle all models.
2. Run feature extraction:
   ```bash
   sbatch scripts/3_feature_extraction.sh mstar
   ```
3. Done! Features will be in `results/stage3_features/mstar/{SITE}/`

## Troubleshooting

### Config Generation Failed

Check template exists:
```bash
ls -la configs/templates/preprocessing_site.yaml.template
```

### STAMP Not Found

Activate STAMP environment:
```bash
source /lab/barcheese01/mdiberna/ARGO-DeepMSI/STAMP/.venv/bin/activate
stamp --version
```

### HF Authentication Error

For gated models (h-optimus-0, h-optimus-1, virchow2, etc.):
```bash
export HF_HOME="/lab/barcheese01/mdiberna/ARGO-DeepMSI/.huggingface_cache"
huggingface-cli login
```

### GPU Not Available

Check SLURM allocation:
```bash
nvidia-smi
```

Verify PyTorch sees GPU:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Migration from Old Structure

Old configs in `configs/{MODEL}/` are kept for reference but no longer used. The template system replaces them.

**Old:** `configs/ctranspath/config_OAUTHC.yaml` (static, hardcoded paths)
**New:** Generate at runtime from `configs/templates/preprocessing_site.yaml.template`

## Next Steps

After feature extraction completes:
1. **Stage 4**: Feature validation - verify all features extracted successfully
2. **Stage 5**: Baseline testing - test with HistoBistro pre-trained model
3. **Stage 6**: MIL training - train models with cross-validation
4. **Stage 7**: Statistics - calculate AUROC/AUPRC metrics
5. **Stage 8**: Visualization - generate heatmaps
