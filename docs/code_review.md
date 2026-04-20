# ARGO-DeepMSI Code Review

## Repository: `mat10d/ARGO-DeepMSI` (branch: `lazyslide-refactor`)

### Project Summary

This is a well-structured pipeline for **MSI (Microsatellite Instability) prediction from whole slide images** of Nigerian colorectal cancer patients, built on top of [LazySlide](https://github.com/rendeirolab/LazySlide). The project ingests clinical data from REDCap + Halo Link, extracts features from slides using pathology foundation models, aggregates to slide-level embeddings, and trains lightweight classifiers (Logistic Regression, Random Forest, SVM, MLP).

**Sites involved:** UITH, OAUTHC (retrospective MSK & OAU cohorts), all imaged in Nigeria.

---

## Architecture Overview

```
argo_deepmsi/
├── cli.py                 # Typer-based CLI (argo ingest/extract/aggregate/train/run)
├── data_ingestion.py      # REDCap + Halo Link → clinical_table + slide_table
├── feature_extraction.py  # LazySlide feature extraction + aggregation
├── training.py            # sklearn classifiers + PyTorch MLP/AttentionMIL
├── visualization.py       # Slide viz, UMAP, heatmaps
└── io_utils.py            # Path management, logging
```

---

## What's Working Well ✅

1. **Clean architecture** — Single CLI entry point (`argo`), well-separated modules, consistent docstrings.

2. **LazySlide integration done right** — Uses the canonical `zs.pp.find_tissues → zs.pp.tile_tissues → zs.tl.feature_extraction` pipeline. Smart use of `wsidata.open_wsi()` for both fresh slides and existing zarr.

3. **Incremental extraction** — `extract_features_single_slide()` checks existing zarr for already-extracted models and only runs new ones. This is critical for 800+ slides × 12+ models.

4. **Comprehensive model support** — 20+ patch-level models (uni2, virchow2, conch_v1.5, gigapath, h-optimus-1, chief, etc.) plus neural slide encoders (PRISM, TITAN, CHIEF, Madeleine). Good mix of gated and non-gated.

5. **Robust patient matching** — Clinical table creation properly handles the retrospective/prospective split, using `crc_redcap_number` to merge retrospective patients who may have multiple slides processed at different locations.

6. **Aggregation flexibility** — Both simple pooling (mean/max/median/sum via numpy) and neural encoders (via `zs.tl.feature_aggregation`) supported.

7. **SLURM scripts are well-designed** — 3-group parallelization for extraction is sensible for ~800 slides. Array jobs for aggregation and training are clean.

8. **Diagnostic reporting** — `plot_ingestion_diagnostics()` and `generate_ingestion_report()` produce publication-quality figures and a structured markdown report.

---

## Issues & Suggestions 🔧

### High Priority

#### 1. **Data leakage in training: slide-level vs patient-level splitting**
The `compare_classifiers()` function uses `StratifiedKFold` on the embedding matrix, but patients can have **multiple slides**. If two slides from the same patient land in different folds, you get data leakage. This is a critical issue for a classifier comparison study.

**Fix:** Use `GroupKFold` or `StratifiedGroupKFold` (sklearn ≥1.3) with patient ID as the group. The merged DataFrame already has `patient_id` — just pass it through.

```python
from sklearn.model_selection import StratifiedGroupKFold
cv = StratifiedGroupKFold(n_splits=n_splits)
# cross_val_score(..., cv=cv, groups=patient_ids)
```

#### 2. **Index alignment bug in `load_training_data()`**
```python
matched_indices = merged.index.tolist()
X = embeddings[matched_indices]
```
After the `merge()`, `merged.index` carries forward from the **metadata** DataFrame's original index, which is only correct if metadata's index is 0..N-1 and no rows were dropped. If the merge drops unmatched rows, the index has gaps, and `embeddings[matched_indices]` silently returns wrong rows. Use `.reset_index()` before merging or explicitly use `np.arange(len(metadata))` row positions.

**Fix:**
```python
metadata = metadata.reset_index(drop=True)  # ensure 0..N-1 index
merged = metadata.merge(clinical, ...)
X = embeddings[merged.index.values]  # now safe
```

#### 3. **Visualization module re-extracts features from scratch**
`visualize_feature_heatmap()` and `visualize_tile_clusters()` call `zs.tl.feature_extraction()` on the slide each time, even though features already exist in the zarr. This is extremely expensive for large slides. They should check for existing zarr files first (like `extract_features_single_slide` does).

#### 4. **Missing `resnet50` in PATCH_MODELS**
The README lists `resnet50` as a non-gated model, but it's not in the `PATCH_MODELS` dictionary in `feature_extraction.py`. Either add it or remove from the README.

### Medium Priority

#### 5. **`visualize` CLI command doesn't use `isMSIH` labels properly**
The `visualize` command reads `clinical_table` but never actually passes the labels to `plot_embedding_umap()`:
```python
labels = None
if clinical_table:
    pd.read_csv(clinical_table)
    # Match labels to embeddings
    # (simplified - assumes slide_id matches PATIENT)
```
This is a placeholder — the UMAP will always render unlabeled. The merge logic from `load_training_data` should be reused here.

#### 6. **Train script array index mismatch**
`scripts/train.sh` has `--array=0-14%5` but the `EMBEDDINGS` array only has ~12 entries (9 uncommented). Array indices beyond the list length will fail silently or crash because `${EMBEDDINGS[$SLURM_ARRAY_TASK_ID]}` will be empty. Should be `--array=0-8%5` to match the 9 uncommented models, or made dynamic.

Similarly `scripts/aggregate.sh` has `--array=0-14%5` but only 9 uncommented models.

#### 7. **`plt.savefig()` used in `plot_ingestion_diagnostics()`**
In `data_ingestion.py` lines 637 and 682, you use `fig.savefig(...)` which is correct! But then `plt.suptitle()` and `plt.tight_layout()` are called on the module level rather than the figure — use `fig.suptitle()` and `fig.tight_layout()` to be safe with multiple active figures.

#### 8. **No error handling for empty slide groups**
If a SLURM group gets 0 slides (possible with rounding), `extract.sh` will create an empty temp CSV and the CLI will process 0 slides without clear feedback. Add a guard:
```bash
if [ $NUM_SLIDES -le 0 ]; then
    echo "No slides for this group, exiting."
    exit 0
fi
```

### Low Priority / Nice-to-Have

#### 9. **`aggregate.sh` doesn't need GPU**
The script requests `--partition=short` (no GPU), which is correct for simple pooling. But if someone edits `METHOD` to `prism`, it would need a GPU. Consider adding a conditional GPU request or a note.

#### 10. **`process_slide_with_aggregation()` has dead code**
The check `if agg_key in wsi.sdata:` — the attribute is `.sdata` but LazySlide stores aggregation results in `.tables[feature_key].uns['agg_slide']` (as the neural encoder code correctly uses). This function may not work as intended for neural encoders.

#### 11. **Type inconsistency in `aggregate_features()`**
`simple_methods` list includes `"std"` and `"var"` but `aggregate_simple_pooling()` doesn't handle these — they'd fall through to the `else` clause and default to mean pooling silently.

#### 12. **Hardcoded paths in SLURM scripts**
`/lab/barcheese01/mdiberna/...` — consider using `$HOME` or a config variable for portability.

#### 13. **Missing `__main__.py`**
The SLURM script calls `python -m argo_deepmsi.cli`, which works because typer is invoked, but adding `argo_deepmsi/__main__.py` with `from .cli import app; app()` would be cleaner.

---

## Summary

This is a **solid, well-organized pipeline** that makes good use of LazySlide's design principles. The major issues are:

1. 🔴 **Patient-level cross-validation** (data leakage) — must fix before publishing results
2. 🔴 **Index alignment in training data loading** — potential silent data mismatch
3. 🟡 **Visualization re-extracting features** — performance issue for interactive use
4. 🟡 **SLURM array indices out of sync** with actual model lists

The code quality is high, the documentation (README, CLAUDE.md) is thorough, and the modular design makes it easy to extend. With the CV leakage fix, this would be ready for a rigorous benchmark comparison.
