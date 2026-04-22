# Execution Runbook

Phase 1 baseline is DONE (0.60 AUROC with mean pooling). This file tracks
the path from 0.60 to 0.85+.

---

## Current State (2026-04-22)

**Cohort:** 803/808 slides, 217 patients, 19% MSI-H.

**Extracted features (in zarrs):**
- Phase 1: uni2, virchow2, conch_v1.5 (complete)
- Running now: ctranspath (incremental, adds to existing zarrs)

**Aggregation running:**
- PRISM on virchow2 (neural slide encoder)
- TITAN on conch_v1.5 (neural slide encoder)

**Baseline AUROCs** (mean pooling + Logistic Regression):
- conch_v1.5_mean: 0.603 ± 0.170
- virchow2_mean: 0.579 ± 0.142
- uni2_mean: 0.553 ± 0.203

---

## Immediate Actions (no re-extraction needed)

### 1. Wagner et al. zero-shot MSI classifier

The Cancer Cell (2023) transformer-based MSI predictor trained on 13,000+
patients from 16 CRC cohorts is publicly available. It uses CTransPath as
the feature extractor — which you're extracting now.

**What to do:** Download the published model weights and run inference
directly on the Nigerian slides. No training, no CV — just forward pass.
This gives the zero-shot generalization baseline: how well does the best
Western MSI classifier transfer to Africa?

```
Paper: Wagner et al., Cancer Cell 2023
       "Transformer-based biomarker prediction from colorectal cancer histology"
Code:  github.com/KatherLab (STAMP pipeline)
Model: CTransPath encoder + transformer aggregator, trained on DACHS/NLCS/QUASAR/TCGA
```

This is a single number but potentially the most important one in the paper.
If it works (AUROC > 0.80), the story is "foundation models generalize to
Africa." If it doesn't (AUROC < 0.70), the story is "domain gap exists,
here's how we bridge it."

### 2. Aggregate + train on PRISM/TITAN (running now)

Once aggregation completes:
```bash
sbatch scripts/aggregate.sh    # auto-discovers PRISM/TITAN embeddings
sbatch scripts/train.sh        # trains on whatever's in results/embeddings/
```

Expected: significant lift over mean pooling. PRISM/TITAN retain spatial
and attention information that mean pooling discards.

### 3. Aggregate + train on ctranspath (when extraction finishes)

ctranspath features enable both:
- Mean pooling baseline (comparable to the other models)
- Input for the Wagner zero-shot classifier

### 4. Multi-model ensemble

You have 3+ foundation model embeddings per slide. Ensemble approaches:

**Late fusion (simplest):**
```python
# Average predicted probabilities from per-model classifiers
p_final = (p_uni2 + p_virchow2 + p_conch) / 3
```

**Feature concatenation:**
```python
# Concatenate slide embeddings, single classifier
X = np.hstack([X_uni2, X_virchow2, X_conch])  # (803, 1024+2560+768)
```

**Stacking (meta-learner):**
```python
# First layer: per-model OOF predictions
# Second layer: logistic regression on stacked OOF scores
```

Literature shows multi-model fusion outperforms any single model, especially
in low-N settings where individual models are noisy.

### 5. Few-shot methods (no overfitting risk)

With ~40 MSI-H patients, classical few-shot approaches are natural:

**k-NN on slide embeddings:**
```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, metric='cosine')
```

**Prototypical networks:**
```python
# Compute class centroids, classify by cosine distance
proto_msih = X[y == 1].mean(axis=0)
proto_mss = X[y == 0].mean(axis=0)
# Score = cosine_sim(x, proto_msih) - cosine_sim(x, proto_mss)
```

These are hyperparameter-light and can't overfit — important at N=217.

### 6. Class-balanced classifiers

Verify `class_weight='balanced'` is actually active in training.py.
Also add:
- XGBoost with `scale_pos_weight = n_mss / n_msih ≈ 4.2`
- Balanced accuracy and AUPRC as evaluation metrics (AUROC alone is
  misleading at 19% prevalence)

---

## Analysis (no extraction needed)

### 7. Site-holdout CV (domain shift measurement)

The cohort spans LASUTH / OAUTHC / UITH / LUTH + retrospective MSK/OAU.
Leave-one-site-out cross-validation reveals:

- **Scanner batch effects** — do models perform worse on some sites?
- **Staining variation** — retrospective MSK slides were stained at MSK
  vs. Nigeria-stained slides. Same patients, different staining. This is
  a natural experiment for stain domain shift.
- **Paper figure** — heatmap of AUROC by (train site, test site) pairs

If MSK-stained retrospective slides outperform Nigeria-stained slides from
the same patients, that's evidence for stain normalization. If not, skip it.

### 8. Scanpy embedding exploration

```python
import scanpy as sc
adata = sc.read_h5ad("results/embeddings/conch_v1.5_mean/embeddings.h5ad")
adata.obs = adata.obs.merge(clinical[["PATIENT", "isMSIH", "SITE"]], ...)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.pl.umap(adata, color=["isMSIH", "SITE"])
```

Quick sanity check: is there any separation by MSI status? Is there site
clustering (batch effect)? This informs whether domain adaptation is needed.

---

## Attention-MIL (requires implementation, operates on zarr tile features)

### 9. ABMIL on tile-level features

The right architecture for WSI classification. Feasible at N=217 because:
- Feature extractor is frozen (uni2/virchow2/conch_v1.5 tile features from zarrs)
- Only training a lightweight attention head (~50K parameters)
- Each slide has ~15K tiles = ample instances for MIL
- Heavy regularization: dropout=0.5, weight decay=0.01, early stopping

```python
# Reads directly from zarr — no re-extraction
zarr_path = Path(slide_path).with_suffix(".zarr")
tiles = zarr.open(zarr_path)["tables"]["uni2_tiles"]["X"][:]
# tiles shape: (n_tiles, 1024)
# ABMIL: attention(tiles) -> weighted sum -> classifier -> MSI prediction
```

Options in order of complexity:
- **ABMIL** (Ilse 2018) — single attention layer, ~20 lines of PyTorch
- **CLAM** (Lu 2021) — clustering-constrained, instance-level supervision
- **TransMIL** (Shao 2021) — transformer over tiles

Start with ABMIL. It's the simplest and most widely used for biomarker
prediction. Published results on MSI with ABMIL: ~0.90+ AUROC on Western
cohorts.

---

## Stain Normalization (requires re-extraction — do last, if needed)

### 10. StainX (Rendeiro lab, same team as LazySlide)

`pip install stainx` — GPU-accelerated Macenko/Reinhard, batch processing,
8-11× speedup over standard PyTorch implementations.

```python
from stainx import Macenko
normalizer = Macenko(device="cuda")
normalizer.fit(reference_image)  # pick a "canonical" Nigeria slide
normalized = normalizer.transform(source_tiles)
```

**Critical:** Stain normalization operates on pixels before tiling and
feature extraction. It requires re-running the full extraction pipeline
on normalized tiles. This is expensive (~11h per 3 models).

**Decision rule:** Run site-holdout CV (item 7) first. If Nigeria-stained
slides significantly underperform MSK-stained slides from the same patients,
stain normalization is worth the cost. If the gap is small, modern foundation
models (trained with stain augmentation) are already robust enough.

**Not part of the current LazySlide pipeline.** LazySlide has no built-in
stain normalization — it's a separate preprocessing step. StainX is the
Rendeiro lab's companion tool for this.

---

## QC (blocked upstream — workaround available)

### 11. Quality control status

`zs.tl.feature_extraction(model="grandqc-artifact")` is broken in the
current LazySlide version (dispatcher signature mismatch). The correct API
is `zs.seg.artifact()` but it produces polygon shapes, not per-tile AnnData.

**Workaround without fixing the API:** Use tile-level features as a proxy.
Tiles with unusual embeddings (outliers in the feature space) likely contain
artifacts. A simple approach:
```python
# Per-slide: compute tile embedding distances to slide centroid
centroid = tiles.mean(axis=0)
distances = np.linalg.norm(tiles - centroid, axis=1)
# Filter tiles beyond 3σ before aggregation
```

This is a soft QC that doesn't require the broken grandqc models.

---

## Autoresearch Grid

Once PRISM/TITAN + ctranspath baselines are in:

```
Foundation models:   [uni2, virchow2, conch_v1.5, ctranspath, ensemble_all]
Aggregation:         [mean, max, PRISM, TITAN, ABMIL]
Classifier:          [LR_balanced, SVM_balanced, XGBoost, kNN_cosine, prototypical]
Feature engineering: [raw, PCA_100, concat_multi_model, stacked_meta]
CV strategy:         [GroupKFold(patient, k=5), LeaveOneSiteOut]
Metrics:             [AUROC, AUPRC, balanced_accuracy]
```

~200-400 configurations, each taking seconds on pre-computed embeddings.
Run this as a systematic sweep, not manual one-at-a-time.

---

## Phase 2 (MSK cluster, when available)

Full foundation model sweep (8+ additional models) on existing zarrs.
Incremental extraction — only new models run.

Also:
- ABMIL with larger model zoo
- Multi-model ABMIL (tile features from all models, late fusion attention)
- Spatial analysis on representative slides
- Vision-language zero-shot characterization

---

## Priority Order

| # | Action | Lift estimate | Effort | Blocked by |
|---|--------|---------------|--------|------------|
| 1 | Wagner zero-shot | Unknown (key result) | Low | ctranspath extraction |
| 2 | PRISM/TITAN train | High (0.60→0.75+) | None (running) | Aggregation completion |
| 3 | Multi-model ensemble | Medium (0.65→0.72) | Low | Phase 1 complete |
| 4 | k-NN / prototypical | Medium | Very low | Nothing |
| 5 | Class-balanced + XGBoost | Low-medium | Very low | Nothing |
| 6 | Site-holdout CV | Analysis only | Low | Nothing |
| 7 | Scanpy UMAP | Analysis only | Very low | Nothing |
| 8 | Autoresearch grid | Systematic | Medium | Items 2-5 |
| 9 | ABMIL | High (0.75→0.85+) | Medium | Implementation |
| 10 | Stain normalization | Unknown | High (re-extract) | Site-holdout result |
| 11 | QC workaround | Low-medium | Low | Nothing |
