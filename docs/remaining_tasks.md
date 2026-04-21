# Full-Cohort Run — Execution Runbook

Step-by-step to take the ingested slide table through to trained
classifiers. The pieces (pyramidal conversion, dask extraction, QC
filtering) are all in place; this file is about the **order** and the
**commands**.

---

## 1. Convert non-pyramidal slides

LazySlide's `find_tissues` OOMs on slides with `n_levels == 1` because it
loads the full-resolution image. Convert them once, up front.

```bash
sbatch scripts/pyramidal.sh results/data/slide_table.csv
```

- Wraps `argo pyramidal` on a SLURM CPU node with 64G RAM (big conversions
  read the whole slide into memory).
- Writes `results/data/slide_table_pyramidal.csv` with `FILENAME` rewritten
  to the converted TIFFs; originals untouched.
- Idempotent: re-running skips slides that already have a `.pyramidal.tiff`.

---

## 2–3. QC — currently disabled (upstream LazySlide bug)

**Status:** skipped for the first full-cohort run (2026-04-21).

**Why:** `zs.tl.feature_extraction(wsi, model="grandqc-artifact")` is broken
in this LazySlide version. The dispatcher in
`lazyslide/tools/_features.py` calls
`MODEL_REGISTRY["grandqc-artifact"](model_path=model_path, token=token)`
but `GrandQCArtifact.__init__(variant='7x')` doesn't accept those kwargs
— every slide fails with
`GrandQCArtifact.__init__() got an unexpected keyword argument 'model_path'`.

**Proper fix (future work):** switch `filter_slides_by_qc` to consume
shapes produced by `zs.seg.artifact(wsi, tile_key=..., model="grandqc",
variant="7x")` — that API works but yields polygon shapes, not the
per-tile AnnData table our current reducer expects. Leave for after the
first baseline training run.

**Workaround for now:** extract foundation models directly on
`slide_table_pyramidal.csv` (next step).

---

## 4. Extract foundation models (Dask)

```bash
python scripts/extract_dask.py \
    --slide-table results/data/slide_table_pyramidal.csv \
    --max-workers 3
```

- Defaults to the 11 production models (see `DEFAULT_MODELS` in
  `extract_dask.py`).
- One worker per slide; failed slides are logged to
  `scripts/logs/dask/failed_slides.txt` but don't stop the run.
- Incremental: a slide whose zarr already has all requested models is
  skipped.

---

## 5. Aggregate

```bash
sbatch scripts/aggregate.sh
```

SLURM array, one task per model; produces
`results/embeddings/{model}_mean/` with `embeddings.npy`,
`metadata.csv`, and `embeddings.h5ad`.

---

## 6. Train

```bash
sbatch scripts/train.sh
```

SLURM array, one task per embedding dir; produces
`results/models/{embedding_dir}/classifier_comparison.csv` with mean/std
AUROC across LogReg, RandomForest, and SVM. Cross-validation uses
`StratifiedGroupKFold(groups=patient_id)` to prevent slide-level leakage.

---

## Driver location for the Dask script

The `extract_dask.py` driver submits SLURM workers via `dask-jobqueue`
and blocks on futures. The driver itself is light-weight but runs for
hours — submit it inside `tmux` on the head node **or** wrap it in a
tiny CPU sbatch job. Do not run it bare in a foreground shell that you
might lose.
