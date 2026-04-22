# Execution Runbook

Phase 1 (baseline on 3 GPUs) is **DONE**. This file now tracks what's next: Phase 1e iteration, and Phase 2 expansion on a larger cluster.

---

## Phase 1 — ✅ Complete (2026-04-22)

Baseline on 3 foundation models, end-to-end, on the 808-slide Nigerian CRC cohort.

| Step | Outcome |
|---|---|
| 1a. Pyramidal conversion | 13 slides converted; default MPP (0.25 μm/px) stamped into MPP-less `generic-tiff` sources |
| 1b. Extraction (dask, 3 × A6000 @ 256G, 3 models) | 11.3h, 803/808 complete (5 data-issue failures — see CLAUDE.md error modalities), **zero OOMs** |
| 1c. Aggregation (mean pooling, auto-discovery) | 3 embedding sets: `conch_v1.5_mean` (768D), `uni2_mean` (1536D), `virchow2_mean` (2560D) |
| 1d. Training (LR / SVM / RF, StratifiedGroupKFold) | Best AUROC per model logged in `results/models/SUMMARY.txt` |

**Cohort:** 803 slides, 217 patients, 19% MSI-H (154 / 803 slides).

**Baseline AUROCs** (Logistic Regression is best for all three):
- `conch_v1.5_mean`: **0.603 ± 0.170**
- `virchow2_mean`: 0.579 ± 0.142
- `uni2_mean`: 0.553 ± 0.203

Random Forest's ~0.80 accuracy is the trivial majority-class predictor (19% MSI-H → 81% MSS baseline accuracy).

---

## Phase 1e — Iterate on the Phase 1 zarrs

All iteration happens on the per-tile features already in `<slide>.zarr/tables/{model}_tiles/`. No re-extraction needed until Phase 2.

### Priorities (highest expected lift first)

1. **Neural aggregators** — mean pooling throws away most of the discriminative signal. Published MSI-from-H&E models (~0.85+ AUROC) use MIL attention or WSI encoders. Low-risk wins:
   - PRISM on `virchow2` (`argo aggregate virchow2 --method prism`)
   - TITAN on `conch_v1.5` (`argo aggregate conch_v1.5 --method titan`)
   - CHIEF slide encoder on `chief` (requires Phase 2 extraction)

2. **Class-balanced classifiers** — 19% MSI-H means unbalanced loss is hurting us:
   - `LogisticRegression(class_weight='balanced')` in `argo_deepmsi/training.py`
   - `SVC(class_weight='balanced', probability=True)`
   - XGBoost with `scale_pos_weight = 649/154 ≈ 4.2`

3. **Attention-MIL** — the right architecture for this problem:
   - ABMIL (Ilse 2018) — simple attention aggregator
   - CLAM (Lu 2021) — clustering-constrained attention
   - TransMIL (Shao 2021) — transformer over tiles
   - All trainable on the zarr-stored per-tile features; no more foundation-model forward passes needed.

4. **Site-holdout CV** — the cohort spans LASUTH / OAUTHC / UITH / LUTH / retrospective_msk / retrospective_oau. Current `StratifiedGroupKFold(groups=patient_id)` mixes sites. A leave-one-site-out eval would reveal scanner-batch-effect robustness — useful for the paper.

5. **Scanpy exploration on `embeddings.h5ad`** — UMAP/PHATE colored by `isMSIH`, `site`, `patient_id`. Quick sanity check that the embedding space has signal at all.

6. **QC refactor** — LazySlide's `zs.tl.feature_extraction(model="grandqc-artifact")` dispatch is broken upstream. Rewrite `filter_slides_by_qc` against `zs.seg.artifact()` polygon output so we can finally prune bad tissue areas before training.

### Autoresearch scope

The right Phase 1e autoresearch loop:

- **Search space:** aggregation method × classifier × class-weight strategy × feature-model subset.
- **Fitness:** OOF AUROC under `StratifiedGroupKFold(groups=patient_id)` on the 803-slide cohort. Secondary: AUPRC (more robust under class imbalance), per-site held-out AUROC.
- **Seed config:** the three Phase 1 mean-pool baselines.
- **Expected winners:** PRISM/TITAN + class-balanced LR or XGBoost. Attention-MIL if the lift justifies wiring it in.

Don't kick autoresearch until (a) PRISM/TITAN baselines are in, so we know the ceiling isn't capped by flat mean-pooling, and (b) a class-balanced LR baseline establishes the real floor.

---

## Phase 2 — Full sweep on a bigger cluster

8 remaining foundation models on top of what Phase 1 produced. The existing zarrs are reused — extraction skips the 3 already-done models per slide and runs only the 8 new ones.

```bash
python scripts/extract_dask.py \
    --slide-table results/data/slide_table_pyramidal.csv \
    --models h-optimus-1 gigapath hibou-b musk chief ctranspath phikonv2 plip \
    --max-workers 15 \
    --partition <msk-gpu-partition>
```

Auto-discovery aggregate + train will pick up the new models automatically — no code edits.

### Candidate additions beyond the 11

- `h-optimus-0`, `hibou-l` (larger variants)
- `gpfm`, `path_orchestra`, `histoplus`, `rosie` (newer models)
- `nulite`, `pathprofiler` (specialized)
- Any future LazySlide-registered model

### Cluster portability

`scripts/extract_dask.py` is SLURM-specific but trivially swappable to cloud via `dask-cloudprovider` or `coiled`. Data movement for a cluster hop:
- ~200 GB of SVS + pyramidal TIFFs
- ~26 GB HF cache (gated foundation-model weights)
- Set `HF_TOKEN` on target, point `HF_HOME` at the synced cache

---

## Architecture guarantees

The pipeline is plug-and-play because:
1. **Zarr is the contract** — every model writes `{slide}.zarr/tables/{model}_tiles`
2. **Extraction is incremental** — existing models are skipped
3. **Aggregation auto-discovers** — reads whatever models exist in zarrs
4. **Training auto-discovers** — iterates over `results/embeddings/*/`
5. **`extract_dask.py --models`** accepts any list — no code changes to add models
