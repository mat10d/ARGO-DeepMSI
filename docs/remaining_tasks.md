# Execution Runbook

## Where We Are

**Phase 1 baseline:** DONE. 803/808 slides, 217 patients, 19% MSI-H.
**Phase 1e analysis sweep:** DONE. A1–A3 domain shift, B1 Tier 1 grid.

### Key Findings

**A1 — Paired staining:** virchow2_mean is remarkably stain-robust
(r=0.94, κ=0.94). Foundation models do NOT need stain normalization
on this cohort. Neural aggregators (PRISM, TITAN) are LESS stain-robust
than mean pooling — they pick up stain-specific rather than biology-
specific patterns (PRISM: r=0.31/κ=0.28, TITAN: r=0.45/κ=0.36).

**A2 — Site-holdout:** Massive site variability (AUROC 0.17–0.75).
Site confounding dominates the embeddings. OAUTHC prospective is the
worst-performing site across all models.

**A3 — Wagner zero-shot:** Patient-level AUROC 0.659, slide-level 0.572.
That's a 0.29 AUROC drop from the published Western benchmark (0.95).
Wagner per-site: 0.75–0.92 on most sites, but 0.44 on OAUTHC prospective.

**B1 — Tier 1 grid:** Flat landscape. Everything between 0.43–0.60.
No classifier or feature engineering choice makes a meaningful difference.
The bottleneck is NOT the classifier — it's the embeddings/aggregation.

**A3 pipeline A/B:** On the 263 patients that overlap with an earlier
HistoBistro-native run (same Wagner weights, HistoBistro's own
CTransPath pipeline), our LazySlide CTransPath features score slightly
*higher* with the same classifier: AUROC **0.718 (new)** vs **0.684 (old)**,
Pearson r=0.78 between the two probability distributions, 100% label
concordance. So feature-extraction / normalization is *not* the source
of the Nigerian generalization gap. See `docs/a3_wagner_zeroshot.md` §
"Pipeline A/B vs HistoBistro-native run".

### What These Findings Tell Us

1. Stain normalization is NOT needed (virchow2 r=0.94 across staining)
2. Neural aggregators (PRISM/TITAN) hurt rather than help
3. Site confounding is the dominant signal masking MSI biology
4. The Western→African generalization gap is real (0.95→0.66)
5. OAUTHC prospective is the specific failure mode (0.44 vs 0.75+ elsewhere)
6. Classifier tuning at the slide-embedding level has hit its ceiling
7. **Re-extraction with different preprocessing is not a candidate fix** —
   our LazySlide features match/exceed HistoBistro's own preprocessing
   when fed to the same pretrained classifier on matched patients

---

## Step 1 — Failure Diagnosis on the Wagner Zero-Shot (IMMEDIATE)

The Wagner classifier is the strongest predictor we have (0.659 patient-
level), and it requires zero training. Before building anything new,
understand WHERE and WHY it fails.

### 1a. Biopsy vs resection stratification

Estimate specimen type from tissue area in the zarr (small tissue =
likely biopsy, large = likely resection).

```python
# Compute tissue area per slide from zarr shapes
for slide in slide_table.itertuples():
    zarr_path = Path(slide.FILENAME).with_suffix(".zarr")
    wsi = open_wsi(str(slide.FILENAME), store=str(zarr_path.parent))
    tissue_area = wsi.shapes["tissues"].geometry.area.sum()
    tile_count = len(wsi.shapes["tiles"])
    # Store: slide_id, tissue_area, tile_count, n_tissues
```

**Analysis:**
- Split slides into small/medium/large tissue area terciles
- Wagner AUROC per tercile — does the model work on large specimens
  but fail on small ones?
- Per-site tissue area distributions — is OAUTHC prospective mostly
  biopsies while other sites are resections?
- Scatter: tissue_area vs Wagner P(MSI-H) colored by true label

**If biopsy-driven:** The model works in Nigeria on resections, and overall
AUROC is dragged down by specimen type. This is a specimen selection issue,
not a generalization failure. Paper framing changes entirely.

### 1b. MSK-IMPACT score analysis

For patients with MSIsensor scores from MSK-IMPACT sequencing:

```python
# Cross-reference Wagner predictions with continuous MSI scores
merged = predictions.merge(impact_data[["PATIENT", "msisensor_score"]])
# Scatter: MSIsensor_score vs Wagner P(MSI-H)
# Color by correct/incorrect binary prediction
# Highlight the "borderline" zone (MSIsensor 5-15)
```

**Analysis:**
- Pearson correlation between Wagner P(MSI-H) and MSIsensor score
- Are "wrong" predictions concentrated at borderline MSI scores?
- If so → the binary label is lossy, not the model. The model may be
  capturing a real continuous signal that the threshold misclassifies.
- Regression target: predict MSIsensor score directly instead of binary

### 1c. Error profiling by metadata covariates

For every slide, compute:
- Wagner prediction (P(MSI-H), correct/incorrect)
- Site, stain_location, cut_location
- Tissue area, tile count, number of tissue regions
- Foundation model embedding distance to cohort centroid (outlier score)

**Analysis:**
- Logistic regression: correct/incorrect ~ site + tissue_area + stain_location
  + tile_count. Which covariate explains the most variance in errors?
- Confusion matrix stratified by site
- High-confidence correct vs high-confidence wrong slide comparison
  (for pathologist review)

### 1d. OAUTHC prospective deep dive

This is 60% of the cohort and the primary failure mode (AUROC 0.44).
The same site's retrospective slides score 0.80 with the same classifier.
Re-extracting features is *not* a candidate — see Key Findings bullet 7
and `docs/a3_wagner_zeroshot.md` pipeline A/B section.

**Hypotheses to test:**
- Label quality: compare `cmo_msi_status` (prospective) vs `msi_status_mmr`
  (retrospective). Any patients with both? Are they concordant?
- Specimen type: is prospective mostly biopsy while retro is resection?
- UMAP colored by (prospective vs retrospective) × site — are the
  embeddings separable? If yes → batch effect, treatable with Harmony.
  If no → something else.

---

## Step 2 — Targeted Improvements Based on Diagnosis (NEXT)

What you do here depends on what Step 1 reveals. Multiple paths:

### 2a. If biopsy/specimen-type is the driver:

- **Resection-only training:** Train and evaluate on resections only.
  Report AUROC on resections + separate AUROC on biopsies.
- **Tissue-area-weighted pooling:** Weight tiles by tissue density
  in aggregation (more tissue → higher weight). Simple, no new code.
- **Tile count minimum filter:** Drop slides below a tile count
  threshold (e.g., <500 tiles). Report how many slides are excluded.

### 2b. If site confounding / batch effect is the driver:

- **Harmony batch correction** on slide embeddings:
  ```python
  import scanpy as sc
  adata = sc.read_h5ad("results/embeddings/virchow2_mean/embeddings.h5ad")
  sc.external.pp.harmony_integrate(adata, key="SITE")
  # Re-run classifiers on corrected embeddings
  ```
  One function call. If site-corrected AUROC improves on leave-one-site-out,
  site confounding was masking MSI signal. This is the cheapest intervention.

- **ComBat** as alternative (statsmodels or scanpy):
  ```python
  sc.pp.combat(adata, key="SITE")
  ```

- **Site as covariate:** Add one-hot site encoding to the feature vector
  before classification. Lets the classifier learn to ignore site.

### 2c. If MSI labels are noisy / borderline:

- **Continuous MSI regression:** Predict MSIsensor score instead of
  binary MSI-H/MSS. More information per sample. Threshold predictions
  at different cutoffs to find optimal operating point.
- **Ordinal classification:** MSS / MSI-L / MSI-H as ordered outcome.
- **Label audit:** For OAUTHC prospective specifically, cross-check
  MSI labels against any available IHC/PCR confirmation.

### 2d. Multi-model fusion (cheap, always helps)

Regardless of what Step 1 shows, ensembling improves over any single model.

```python
# Late fusion — average predicted probabilities
p_fused = np.mean([p_conch, p_virchow2, p_uni2, p_ctranspath], axis=0)

# Feature concatenation — single classifier on stacked embeddings
X_concat = np.hstack([X_conch, X_virchow2, X_uni2, X_ctranspath])
# (803, 768+2560+1536+768) = (803, 5632)
# PCA to 100 dims first to regularize

# Stacking — per-model OOF predictions → meta-learner
meta_features = np.column_stack([oof_conch, oof_virchow2, oof_uni2, oof_ctranspath])
meta_clf = LogisticRegression(class_weight="balanced").fit(meta_features, y)
```

### 2e. Few-shot methods (zero overfitting risk)

```python
# k-NN on slide embeddings (cosine distance)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")

# Prototypical networks
proto_msih = X[y == 1].mean(axis=0)
proto_mss = X[y == 0].mean(axis=0)
score = cosine_sim(x, proto_msih) - cosine_sim(x, proto_mss)
```

Hyperparameter-light, can't overfit. Good for N=217.

---

## Step 3 — Autoresearch Tier 2+ (AFTER Step 2 decisions)

### 3a. Tier 2 — batch-corrected embeddings

If Harmony/ComBat helps (Step 2b), re-run the full Tier 1 grid on
corrected embeddings. Same {4 models} × {classifiers} × {feat-eng},
but on site-regressed features.

### 3b. Tier 3 — multi-model fusion grid

Systematic sweep of fusion strategies on best single-model configs:
- Late fusion (average probs): all 2/3/4-model combinations
- Feature concat + PCA: [model_A || model_B] → PCA_100 → classifier
- Stacking: OOF predictions → LR/XGBoost meta-learner
- ~50 configs, seconds each

### 3c. Tier 4 — continuous MSI regression (if IMPACT scores available)

Replace binary labels with MSIsensor scores. Run the Tier 1 grid but
with regression metrics (Pearson r, Spearman ρ, RMSE). Then threshold
predictions to recover classification metrics at optimal cutoff.

This is a legitimate methodological contribution — most MSI-from-H&E
work uses binary labels. Showing that regression outperforms classification
at low N is publishable.

### 3d. Autoresearch output

```
results/autoresearch/{run_id}/
├── config.yaml
├── results.csv           # model, agg, clf, feat, AUROC, AUPRC, bal_acc
├── best_config.yaml
├── oof_predictions.csv
└── figures/
    ├── auroc_heatmap.png
    ├── site_holdout.png
    └── fusion_comparison.png
```

---

## Step 4 — ABMIL on Tile Features (REQUIRES IMPLEMENTATION)

Deferred until Steps 1-3 are complete, because:
- If specimen type is the issue, ABMIL won't fix it (garbage in)
- If site confounding dominates, fix that first in the embeddings
- If labels are noisy, ABMIL amplifies the noise

But once upstream issues are addressed, ABMIL is the architecture
that gets published MSI papers to 0.85+.

### Implementation spec

```python
class ABMIL(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.5):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
        )
        self.classifier = nn.Linear(input_dim, 1)

    def forward(self, tiles):  # (n_tiles, input_dim) from zarr
        a = torch.softmax(self.attention(tiles), dim=0)
        z = (a * tiles).sum(dim=0, keepdim=True)
        return self.classifier(z).squeeze()
```

**Training recipe for low-N:**
- 5-fold StratifiedGroupKFold (patient-level)
- Epochs: 50, early stopping (patience=10)
- Adam, lr=1e-4, weight_decay=1e-2
- BCE with pos_weight=4.2
- Dropout 0.5, batch size 1 slide
- Data: tiles directly from zarr per slide

**Pre-training on TCGA:**
- TCGA-COAD/READ: ~630 patients with MSI labels + public H&E WSIs
- Extract same foundation model features on TCGA slides
- Pre-train ABMIL on TCGA, fine-tune on Nigerian cohort
- This is the domain adaptation play: large Western → small African

**Multi-model ABMIL fusion:**
```python
p_final = sigmoid(mean([abmil_uni2(tiles), abmil_virchow2(tiles), ...]))
```

### Public pre-training datasets

| Dataset | Patients | MSI-H % | Access |
|---|---|---|---|
| TCGA-COAD | ~460 | ~15% | GDC (public) |
| TCGA-READ | ~170 | ~5% | GDC (public) |
| CPTAC-COAD | ~110 | ~28% | TCIA (public) |
| TCGA-STAD | ~324 | ~20% | GDC (cross-tissue) |
| TCGA-UCEC | ~430 | ~36% | GDC (cross-tissue) |

UNI2 team released 25K+ pre-extracted WSI embeddings from TCGA/CPTAC.
If those include COAD/READ with MSI labels, skip extraction entirely.

---

## Step 5 — Harmony / Batch Correction (PARALLEL WITH STEP 2)

The A2 site-holdout heatmap shows site dominates embeddings. Harmony
is a one-liner that can be run immediately on existing h5ad files.

```python
import scanpy as sc

# For each foundation model:
adata = sc.read_h5ad("results/embeddings/virchow2_mean/embeddings.h5ad")
adata.obs = adata.obs.merge(clinical[["PATIENT", "isMSIH", "SITE"]])

# Harmony integration — regress out SITE
sc.external.pp.harmony_integrate(adata, key="SITE")
# adata.obsm["X_pca_harmony"] now has site-corrected embeddings

# Re-run classifiers on corrected features
X_corrected = adata.obsm["X_pca_harmony"]
```

**Evaluation:** Compare leave-one-site-out AUROC before vs after Harmony.
If it improves → site confounding was masking signal. Re-run full Tier 1
grid on corrected embeddings.

**Alternative: ComBat**
```python
sc.pp.combat(adata, key="SITE")
```

Both are fast, deterministic, well-established. Run both, compare.

---

## QC (Tile-Level Outlier Filtering)

```python
# Per-slide: flag artifact tiles by embedding distance
centroid = tile_embeddings.mean(axis=0)
distances = np.linalg.norm(tile_embeddings - centroid, axis=1)
clean_mask = distances < (distances.mean() + 3 * distances.std())
# Pool only clean tiles; pass clean_mask to ABMIL attention masking
```

No API dependency, no re-extraction. Integrate into aggregation and ABMIL.

---

## Phase 2 — MSK Cluster (when available)

Full foundation model sweep (11+ models) on existing zarrs.
ABMIL with the full model zoo.
TCGA pre-training pipeline.
Spatial analysis, vision-language queries.

---

## Execution Order

```
IMMEDIATE (Step 1 — all on existing data, no new training):
  ├─ 1a: Biopsy/resection stratification from tissue area
  ├─ 1b: MSIsensor score correlation with Wagner predictions
  ├─ 1c: Error profiling by metadata covariates
  ├─ 1d: OAUTHC prospective deep dive (labels, UMAP, batch)
  └─ 5:  Harmony batch correction on existing h5ad files

NEXT (Step 2 — based on Step 1 findings):
  ├─ 2a-c: Targeted fix (specimen filter / Harmony / regression)
  ├─ 2d: Multi-model fusion
  ├─ 2e: Few-shot methods (k-NN, prototypical)
  └─ QC:  Tile-level outlier filtering

AUTORESEARCH (Step 3 — systematic):
  ├─ 3a: Tier 2 on batch-corrected embeddings
  ├─ 3b: Tier 3 multi-model fusion grid
  └─ 3c: Tier 4 continuous MSI regression

IMPLEMENTATION (Step 4 — after upstream issues resolved):
  ├─ ABMIL on tile features
  ├─ TCGA pre-training + Nigerian fine-tune
  └─ Multi-model ABMIL fusion

PHASE 2 (MSK cluster):
  └─ Full model sweep + advanced analyses
```

---

## Priority Matrix

| # | Action | Expected lift | Effort | Blocked by |
|---|--------|---------------|--------|------------|
| 1a | Biopsy/resection split | Diagnostic | Very low | Nothing |
| 1b | MSIsensor correlation | Diagnostic | Very low | IMPACT data |
| 1c | Error profiling | Diagnostic | Low | Nothing |
| 1d | OAUTHC audit | Diagnostic (critical) | Low | Nothing |
| 5 | Harmony batch correction | 0.55→0.65+ (if site-driven) | Very low | Nothing |
| 2d | Multi-model fusion | +0.02-0.05 | Low | Nothing |
| 2e | Few-shot (k-NN, proto) | +0.01-0.03 | Very low | Nothing |
| 3a | Tier 2 on corrected emb | Unknown | Low | Step 5 |
| 3b | Tier 3 fusion grid | +0.02-0.05 | Low | Step 2d |
| 3c | Regression on MSIsensor | Novel method | Medium | IMPACT data |
| 4 | ABMIL | 0.65→0.80+ | Medium-high | Steps 1-3 |
| 4+ | TCGA pre-train + fine-tune | 0.70→0.85+ | High | TCGA data |
