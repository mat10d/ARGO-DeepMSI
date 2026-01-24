# HPC Scripts for ARGO-DeepMSI

## Feature Extraction Scripts

### Test Extraction (UITH subset)
```bash
sbatch scripts/extract_uith_test.sh
```
- Runs on 12 UITH slides only
- Uses PLIP model (no auth required)
- Good for testing the pipeline

### Non-Gated Models (No HuggingFace Auth)
```bash
sbatch scripts/extract_all_models.sh
```
- Runs 4 models in parallel: plip, ctranspath, phikon, phikonv2
- No HuggingFace authentication required
- Each model runs on its own GPU node
- `--array=0-3%4` runs max 4 jobs concurrently

### Gated Models (Requires HuggingFace Auth)
```bash
sbatch scripts/extract_gated_models.sh
```
- Runs 6 recommended gated models: uni2, virchow2, h-optimus-0, gigapath, conch, hibou-b
- Requires HF_TOKEN in `.env` file
- Uses larger GPU nodes (A6000) with more memory
- `--array=0-5%3` runs max 3 jobs concurrently

## Monitoring Jobs

Check job status:
```bash
squeue -u $USER
```

Check specific job output:
```bash
tail -f scripts/logs/extract_<JOBID>_<TASKID>.out
```

Cancel all jobs:
```bash
scancel -u $USER
```

Cancel specific job:
```bash
scancel <JOBID>
```

## Customizing Models

Edit the `MODELS` array in the script to run different models:

```bash
MODELS=(
    "uni2"
    "virchow2"
    "your_model_here"
)
```

Update the `--array` parameter to match the number of models:
- For N models: `--array=0-$((N-1))%X` where X is max concurrent jobs

## Output Structure

Features are saved to:
```
results/features/<model_name>/<slide_id>.h5ad
```

Logs are saved to:
```
scripts/logs/extract_<JOBID>_<TASKID>.out
```
(stderr and stdout combined in .out file)

## GPU Partitions Available

- `nvidia-t4-20` - T4 GPUs (good for testing, smaller models)
- `nvidia-A6000-20` - A6000 GPUs (recommended for large models)
- `nvidia-A100-20` - A100 GPUs (fastest, limited availability)
- `nvidia-L40S-20` - L40S GPUs (good balance)

## Aggregation and Visualization

After feature extraction completes, run:
```bash
python scripts/aggregate_and_visualize.py --clinical results/data/clinical_table.csv
```
