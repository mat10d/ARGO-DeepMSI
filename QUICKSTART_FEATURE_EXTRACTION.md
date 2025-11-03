# Quick Start: Feature Extraction

## Prerequisites

### 0. Install STAMP Environment (First Time Only)

**IMPORTANT:** If STAMP is not installed, you must install it on a **compute node**, not the head node:

```bash
# Request interactive GPU node
srun --partition=nvidia-2080ti-20 --gres=gpu:1 --mem=64G --time=2:00:00 --pty bash

# Install STAMP
cd /lab/barcheese01/mdiberna/ARGO-DeepMSI/STAMP
MAX_JOBS=4 uv sync --extra build --extra gpu

# Verify
source .venv/bin/activate
stamp --version

# Exit compute node
exit
```

See `ENVIRONMENT_SETUP.md` for more details.

### 1. Activate STAMP Environment

```bash
cd /lab/barcheese01/mdiberna/ARGO-DeepMSI
source STAMP/.venv/bin/activate
```

### 2. Authenticate with Hugging Face (One-Time Setup)

```bash
# Login to Hugging Face
huggingface-cli login

# Paste your token from: https://huggingface.co/settings/tokens
```

### 3. Request Access to Gated Models (One-Time Setup)

Visit these pages and click "Request Access":
- **Virchow2**: https://huggingface.co/paige-ai/Virchow2
- **UNI2**: https://huggingface.co/MahmoodLab/UNI2-h
- **H-optimus-0**: https://huggingface.co/bioptimus/H-optimus-0
- **H-optimus-1**: https://huggingface.co/bioptimus/H-optimus-1
- **CONCHv1.5**: https://huggingface.co/MahmoodLab/conchv1_5
- **Gigapath**: https://huggingface.co/prov-gigapath/prov-gigapath

Access is usually granted within hours.

---

## Running Feature Extraction

### Option 1: Extract Features for ALL Models (Recommended)

This runs feature extraction **sequentially** across all available STAMP models:

```bash
bash scripts/3_feature_extraction_all.sh
```

**What it does:**
- Tests each model for accessibility
- Runs models in order (no-auth models first, then gated models)
- Each model processes **6 sites in parallel** (SLURM array)
- Skips inaccessible models
- Logs everything to `logs/feature_extraction_all/extraction_results_<timestamp>.log`

**Models processed (12 total):**

No authentication required:
- ctranspath
- plip
- dinobloom
- chief-ctranspath

Gated models (newer versions preferred):
- virchow2
- uni2
- conch1_5
- gigapath
- h-optimus-0
- h-optimus-1
- mstar
- musk

**Estimated time:**
- ~6-12 hours per model (depends on dataset size)
- Total: ~3-5 days for all models

---

### Option 2: Extract Features for ONE Model

Run a single model across all sites:

```bash
# No HF auth required
sbatch scripts/3_feature_extraction.sh ctranspath
sbatch scripts/3_feature_extraction.sh plip

# Gated models (requires HF auth)
sbatch scripts/3_feature_extraction.sh virchow2
sbatch scripts/3_feature_extraction.sh uni2
sbatch scripts/3_feature_extraction.sh h-optimus-0
```

---

## Test Model Access Before Running

Test if a model is accessible:

```bash
source STAMP/.venv/bin/activate
python scripts/test_model_access.py ctranspath
python scripts/test_model_access.py virchow2
```

**Output:**
```
✓ STAMP available
✓ Hugging Face authenticated as: YourUsername
Testing model: virchow2
============================================================
Loading virchow2 extractor...
✓ Model virchow2 loaded successfully

✓ Model virchow2 is accessible and ready to use
```

---

## Monitoring Progress

### Check SLURM Jobs

```bash
# View running jobs
squeue -u $USER

# View job details
squeue -u $USER -o "%.18i %.9P %.30j %.8u %.2t %.10M %.6D %R"

# Cancel a job
scancel <JOB_ID>
```

### Check Logs

```bash
# Feature extraction logs (per job)
ls -ltr logs/feature_extraction/

# View latest log
tail -f logs/feature_extraction/stamp_*.out

# All models log (from 3_feature_extraction_all.sh)
tail -f logs/feature_extraction_all/extraction_results_*.log
```

### Check Output Features

```bash
# List extracted features
ls -lh results/stage3_features/

# Example: Check ctranspath features for OAUTHC
ls -lh results/stage3_features/ctranspath/OAUTHC/

# Count feature files
find results/stage3_features/ctranspath/OAUTHC/ -name "*.h5" | wc -l
```

---

## Expected Output Structure

After feature extraction completes:

```
results/stage3_features/
├── ctranspath/
│   ├── OAUTHC/              # ~200-500 .h5 files
│   ├── LUTH/
│   ├── LASUTH/
│   ├── UITH/
│   ├── retrospective_msk/
│   └── retrospective_oau/
├── virchow2/
│   └── [same structure]
├── h-optimus-0/
│   └── [same structure]
└── ...
```

Each `.h5` file contains features for one slide.

---

## Troubleshooting

### "Model not accessible" Error

**For gated models:**
1. Check HF authentication: `huggingface-cli whoami`
2. Verify access granted: Visit model page on HF
3. Re-login if needed: `huggingface-cli login`

**Test access:**
```bash
python scripts/test_model_access.py virchow2
```

### "STAMP not found" Error

Activate STAMP environment:
```bash
source STAMP/.venv/bin/activate
stamp --version
```

### "CUDA out of memory" Error

Reduce batch size or max_workers in template:
```bash
# Edit: configs/templates/preprocessing_site.yaml.template
# Change: max_workers: 16 → max_workers: 8
```

### Job Stuck or Failed

Check SLURM log:
```bash
# Find job ID
squeue -u $USER

# Check output
cat logs/feature_extraction/stamp_*_<JOB_ID>_*.out
```

Cancel and restart:
```bash
scancel <JOB_ID>
sbatch scripts/3_feature_extraction.sh ctranspath
```

### Disk Space Issues

Models are cached in `.huggingface_cache/`:
```bash
# Check cache size
du -sh .huggingface_cache/

# Free up space (re-downloads on next run)
rm -rf .huggingface_cache/hub/*
```

---

## Next Steps

After feature extraction completes:

1. **Stage 4: Feature Validation**
   ```bash
   conda activate argo
   python scripts/4_feature_validation.py
   ```

2. **Stage 5: Baseline Testing** (optional)
   ```bash
   conda activate histobistro
   python scripts/5_baseline_testing.py
   ```

3. **Stage 6: MIL Training**
   ```bash
   source STAMP/.venv/bin/activate
   sbatch scripts/6_mil_training.sh ctranspath
   ```

---

## Summary Commands

```bash
# Complete workflow
cd /lab/barcheese01/mdiberna/ARGO-DeepMSI

# 1. Setup (one-time)
source STAMP/.venv/bin/activate
huggingface-cli login
python scripts/test_model_access.py ctranspath  # Test

# 2. Run all models
bash scripts/3_feature_extraction_all.sh

# 3. Monitor
tail -f logs/feature_extraction_all/extraction_results_*.log

# 4. Check results
ls -lh results/stage3_features/
```
