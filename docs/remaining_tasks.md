# Remaining Tasks

Three things to fix before the next full-cohort run.

---

## 1. Switch to Dask extraction (`extract_dask.py`)

**Problem:** `extract.sh` runs 2 static SLURM groups of ~400 slides each.
If one group finishes early, that GPU idles. If one slide OOMs, it kills
the entire group (400 slides lost until retry). Walltime: 7 days.

**Solution:** `scripts/extract_dask.py` submits one Dask worker per slide,
auto-scales 1–N GPUs, and isolates failures to individual slides. A single
OOM doesn't kill anything else. Already built, now with the zarr-reopen
fix applied.

```bash
pip install -e ".[dask]"

python scripts/extract_dask.py \
    --slide-table results/data/slide_table.csv \
    --partition nvidia-A6000-20 \
    --memory "256 GB" \
    --max-workers 3 \
    --walltime 24:00:00
```

Key differences from `extract.sh`:
- One slide per worker — no group splitting, no temp CSVs
- `cluster.adapt(min=1, max=N)` — auto-scales based on queue availability
- Per-slide failure isolation — exceptions are caught per-future, not per-group
- Dashboard URL printed on start for live monitoring
- Same incremental logic — skips already-extracted models per slide
- Uses `open_wsi(svs_path, store=parent)` — no fastslide KeyError

`--max-workers 3` stays within the A6000-20 partition's 768G cap (256G × 3).
Increase if the partition allows more.

---

## 2. Convert non-pyramidal slides before extraction

**Problem:** `HP1353_21_1.svs` (78K×75K, 1 pyramid level) OOMs during
`find_tissues` at 256G because LazySlide tries to load the full image.
Any non-pyramidal slide will hit this. We currently exclude it — but we
shouldn't lose data.

**Solution:** `scripts/convert_pyramidal.sh` scans for non-pyramidal slides
and converts them to pyramidal TIFFs with `vips`. Run once before extraction.

```bash
bash scripts/convert_pyramidal.sh results/data/slide_table.csv
```

What it does:
1. Reads each slide path from the slide table
2. Checks pyramid levels via `openslide-show-properties` (or Python fallback)
3. Non-pyramidal slides (n_levels <= 1) get converted:
   `vips tiffsave input.svs output.tiff --pyramid --tile --compression jpeg --Q 90`
4. Updates the slide table FILENAME to point to the converted TIFF
5. Preserves the original SVS (renamed to `.svs.original`)

Requires `libvips` (`conda install -c conda-forge libvips` or `apt install libvips-tools`).

---

## 3. Run QC before feature extraction

**Problem:** QC models (`grandqc-artifact`, `grandqc-tissue`) are built and
tested (`argo qc` CLI + `filter_slides_by_qc()`) but commented out in the
production flow. We extract expensive foundation models on every slide
regardless of quality.

**Solution:** Two-pass flow, already supported:

```bash
# Pass 1: Extract QC models only (fast, lightweight)
python scripts/extract_dask.py \
    --slide-table results/data/slide_table.csv \
    --models grandqc-artifact grandqc-tissue \
    --max-workers 5

# Pass 2: Filter by QC scores
argo qc results/data/slide_table.csv \
    --model grandqc-artifact \
    --reduce mean --threshold 0.5 \
    --output results/data/slide_table_qc.csv

# Pass 3: Extract foundation models on clean slides only
python scripts/extract_dask.py \
    --slide-table results/data/slide_table_qc.csv \
    --max-workers 3
```

The QC pass is fast (small models, CPU-viable) and filters out slides that
would waste hours of GPU time on feature extraction.

---

## Execution Order

```
1. Convert non-pyramidal slides    (scripts/convert_pyramidal.sh)
2. Extract QC models               (extract_dask.py --models grandqc-artifact grandqc-tissue)
3. Filter by QC                    (argo qc --threshold 0.5 --output slide_table_qc.csv)
4. Extract foundation models       (extract_dask.py --slide-table slide_table_qc.csv)
5. Aggregate                       (sbatch scripts/aggregate.sh)
6. Train                           (sbatch scripts/train.sh)
```

Steps 2–4 use the Dask path. Steps 5–6 are short CPU jobs that stay as
SLURM arrays (no benefit from Dask for 30-min tasks).
