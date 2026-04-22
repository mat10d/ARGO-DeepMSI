# Execution Runbook

Phase 1 baseline DONE (0.60 AUROC, mean pooling). This is the complete
plan to reach 0.85+ and generate the paper's key results.

---

## Current State

**Cohort:** 803/808 slides, 217 patients, 19% MSI-H.

**Slide table columns:** PATIENT, FILENAME, SITE, cut_location,
stain_location, image_location (always Nigeria).

**Sites:** UITH, LASUTH, OAUTHC, LUTH, retrospective_msk, retrospective_oau.
Retrospective patients (142-series) have slides stained at MSKCC *and* in
Nigeria — same patient, same scanner (Nigeria), different staining protocol.

**Extracted features (in zarrs):**
- Complete: uni2, virchow2, conch_v1.5
- Running: ctranspath (incremental)
- Running: PRISM on virchow2, TITAN on conch_v1.5 (neural aggregation)

**Baseline AUROCs** (mean pooling + LR):
- conch_v1.5: 0.603 ± 0.170
- virchow2: 0.579 ± 0.142
- uni2: 0.553 ± 0.203

---

## Part A — Domain Shift Analysis (the paper's scientific core)

This is what makes the paper novel: a rigorous three-level domain shift
analysis on the first African MSI-from-H&E cohort, using a natural
within-patient staining experiment.

### A1. Within-patient, across-staining (the gold test)

**Subset:** Retrospective patients who have slides with
`stain_location=MSKCC` AND slides with `stain_location` in
{OAUTHC, UITH, other Nigeria sites}. Same patient, same scanner
(all imaged in Nigeria), different staining protocol.

**Analysis:**
```python
# For each retrospective patient with both MSK and Nigeria staining:
#   1. Get slide-level MSI prediction from MSK-stained slide(s)
#   2. Get slide-level MSI prediction from Nigeria-stained slide(s)
#   3. Compute concordance

# Metrics:
#   - Per-patient prediction concordance (Cohen's kappa)
#   - Mean absolute prediction score difference (MSK vs Nigeria)
#   - Paired signed-rank test on prediction scores
#   - Scatter plot: P(MSI-H | MSK stain) vs P(MSI-H | Nigeria stain)
```

**Interpretation:**
- High concordance → foundation models are stain-robust, Nigeria
  deployment is viable without normalization
- Systematic MSK > Nigeria → stain domain shift exists, normalization
  needed for deployment
- Discordant both directions → noise, not systematic shift

**Paper figure:** Paired scatter with identity line, colored by true
MSI status. This is the hero figure for the domain shift story.

### A2. Across-site, within-Nigeria (scanner/protocol variation)

**Design:** Leave-one-site-out CV across the Nigerian sites.
Train on all slides from N-1 sites, test on held-out site.

```python
# For each site in [UITH, LASUTH, OAUTHC, LUTH, ...]:
#   Train on all other sites
#   Evaluate on held-out site
#   Record AUROC, AUPRC, balanced accuracy

# Also: StratifiedGroupKFold within each train set (patient-level)
# for honest hyperparameter selection
```

**Metrics:**
- Per-site held-out AUROC
- Cross-site AUROC heatmap (train site rows × test site columns)
- Average cross-site drop vs. within-site CV

**Paper figure:** Heatmap showing generalization across Nigerian sites.

### A3. Western → Africa generalization (the Wagner test)

**Design:** Run the Wagner et al. (Cancer Cell 2023) pre-trained MSI
classifier directly on the Nigerian cohort. No fine-tuning.

This classifier was trained on 13,000+ Western CRC patients from 16
cohorts (DACHS, NLCS, QUASAR, TCGA, etc.) using CTransPath features.
The paper noted "a generalization gap when intrinsic biological factors,
such as ethnicity, change."

```python
# 1. Extract CTransPath features (running now)
# 2. Download Wagner et al. model weights (publicly available)
# 3. Forward pass on all 803 Nigerian slides
# 4. Compare to their published AUROC (~0.95 on Western cohorts)
```

**Metric:** AUROC on Nigerian cohort vs. published Western AUROC.

**Paper figure:** ROC curve overlaid with the Wagner et al. published
curve. The gap between them IS the generalization penalty.

### A4. Domain adaptation analysis (optional, high novelty)

If A1 shows staining matters:
- Compare raw vs. StainX-normalized features (requires re-extraction)
- Test whether fine-tuning the Wagner classifier on even 50 Nigerian
  slides closes the gap (few-shot domain adaptation)

If A2 shows site matters:
- Batch-effect correction (ComBat/Harmony on slide embeddings)
- Site-aware CV as the standard evaluation going forward

---

## Part B — Autoresearch: Systematic Classifier Optimization

Once PRISM/TITAN + ctranspath are ready, run a systematic grid search
on pre-computed embeddings. Every configuration takes seconds — no
GPU needed.

### B1. Search space

```yaml
foundation_models:
  - uni2
  - virchow2
  - conch_v1.5
  - ctranspath
  # later: h-optimus-1, gigapath, hibou-b, musk, chief, phikonv2, plip

aggregation:
  simple_pooling:
    - mean
    - max
    - median
  neural_encoders:      # slide-level embeddings from pre-trained encoders
    - virchow2_prism
    - conch_v1.5_titan
  attention_mil:        # trained on tile features from zarrs
    - abmil
    - clam_sb           # single-branch CLAM
    - transmil           # if N supports it

classifiers:
  linear:
    - LogisticRegression(class_weight='balanced', C=[0.01, 0.1, 1, 10])
    - SVC(class_weight='balanced', kernel='rbf', probability=True)
  tree:
    - XGBoost(scale_pos_weight=4.2, max_depth=[3,5,7], n_estimators=[100,300])
    - RandomForest(class_weight='balanced', n_estimators=500)
  few_shot:
    - KNeighborsClassifier(n_neighbors=[3,5,7,11], metric='cosine')
    - PrototypicalClassifier(metric='cosine')  # custom, ~10 lines

feature_engineering:
  - raw                          # single model embedding
  - pca_100                      # PCA to 100 dims (regularization)
  - concat_top3                  # uni2 + virchow2 + conch_v1.5 concatenated
  - stacked_meta                 # OOF predictions from per-model classifiers → meta-learner

cv_strategy:
  - StratifiedGroupKFold(groups=patient_id, n_splits=5)
  - LeaveOneSiteOut              # from Part A2

metrics:
  primary: AUROC
  secondary: [AUPRC, balanced_accuracy, sensitivity_at_95_specificity]
```

### B2. Execution plan

**Tier 1 — no implementation needed (~100 configs, minutes):**
All combinations of {4 models} × {3 simple poolings} × {4 classifiers}
× {2 feature engineering} × {GroupKFold}. Run as a single Python script
on CPU.

**Tier 2 — PRISM/TITAN results added (~50 more configs):**
Same classifiers on the neural-encoder embeddings. Expect the biggest
AUROC jump here.

**Tier 3 — multi-model fusion (~30 configs):**
- Late fusion: average P(MSI-H) across models
- Feature concatenation: [uni2 || virchow2 || conch_v1.5] → classifier
- Stacking: per-model OOF predictions → LR meta-learner

**Tier 4 — Attention-MIL (~20 configs, requires implementation):**
ABMIL/CLAM on tile-level features from zarrs. One model at a time,
then multi-model late fusion of ABMIL outputs.

### B3. ABMIL implementation spec

```python
class ABMIL(nn.Module):
    """Attention-Based Multiple Instance Learning (Ilse 2018).
    
    Reads tile features directly from zarr — no re-extraction.
    Lightweight: ~50K trainable params for 1024D input.
    """
    def __init__(self, input_dim, hidden_dim=256, dropout=0.5):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 1),
        )

    def forward(self, tiles):
        # tiles: (n_tiles, input_dim) from zarr
        a = self.attention(tiles)                    # (n_tiles, 1)
        a = torch.softmax(a, dim=0)                  # attention weights
        z = (a * tiles).sum(dim=0, keepdim=True)      # weighted sum
        return self.classifier(z).squeeze()           # logit
```

**Training recipe for low-N (217 patients):**
- 5-fold StratifiedGroupKFold (patient-level)
- Epochs: 50, early stopping on validation AUROC (patience=10)
- Optimizer: Adam, lr=1e-4, weight_decay=1e-2
- Loss: BCE with pos_weight=4.2 (class imbalance)
- Dropout: 0.5 (heavy — prevents overfitting at low N)
- Batch size: 1 slide (standard for MIL)
- Data loading: read tiles directly from zarr per slide

**Multi-model ABMIL fusion:**
```python
# Train separate ABMIL per foundation model
# At inference: average logits or attention-weighted embeddings
p_final = sigmoid(mean([abmil_uni2(tiles), abmil_virchow2(tiles), ...]))
```

### B4. Output format

Every autoresearch run saves:
```
results/autoresearch/{run_id}/
├── config.yaml           # full search space + hyperparams
├── results.csv           # one row per config: model, agg, clf, AUROC, AUPRC, ...
├── best_config.yaml      # top config by primary metric
├── oof_predictions.csv   # OOF predictions for the best config
└── figures/
    ├── auroc_heatmap.png       # model × aggregation × classifier
    ├── site_holdout.png        # per-site AUROC
    └── paired_staining.png     # MSK vs Nigeria stain concordance
```

---

## Part C — QC and Preprocessing

### C1. Tile-level QC (no grandqc dependency)

Workaround for the broken `zs.tl.feature_extraction(model="grandqc-artifact")`:

```python
# Per-slide: flag outlier tiles by embedding distance
centroid = tile_embeddings.mean(axis=0)
distances = np.linalg.norm(tile_embeddings - centroid, axis=1)
threshold = distances.mean() + 3 * distances.std()
clean_mask = distances < threshold
# Use clean_mask to filter tiles before aggregation or ABMIL
```

Integrate into aggregation: only pool clean tiles. Integrate into ABMIL:
mask out outlier tiles before attention. Costs nothing, no API dependency.

### C2. Stain normalization (StainX — if domain shift analysis warrants)

Only pursue if Part A1 shows systematic staining effect.

```python
from stainx import Macenko
normalizer = Macenko(device="cuda")
normalizer.fit(reference_image)  # canonical Nigeria slide
# Apply to all tiles before feature extraction → re-extract
```

Requires re-extraction (~11h per 3 models). Defer until A1 results
are in hand.

---

## Part D — Phase 2 (MSK cluster)

Full model sweep on existing zarrs (incremental). Neural aggregation
with PRISM/TITAN on all models. ABMIL with the full model zoo.
Spatial analysis. Vision-language queries.

---

## Execution Order

```
NOW (embeddings already exist or running):
  ├─ A1: Paired staining analysis (retrospective patients)
  ├─ A2: Site-holdout CV
  ├─ B1: Autoresearch Tier 1 (simple pooling × classifiers)
  ├─ 6-8: Scanpy UMAP, class-balanced classifiers, k-NN
  └─ C1: Tile-level QC filtering

WHEN PRISM/TITAN + CTRANSPATH FINISH:
  ├─ A3: Wagner zero-shot evaluation
  ├─ B2: Autoresearch Tier 2 (neural encoder embeddings)
  └─ B3: Autoresearch Tier 3 (multi-model fusion)

NEXT SPRINT (requires implementation):
  ├─ B4: ABMIL implementation + training
  └─ Autoresearch Tier 4 (attention-MIL configs)

IF DOMAIN SHIFT WARRANTS:
  └─ C2: StainX normalization + re-extraction

PHASE 2 (MSK CLUSTER):
  └─ D: Full model sweep + advanced analysis
```

---

## Priority Matrix

| # | Action | Expected lift | Effort | Blocked by |
|---|--------|---------------|--------|------------|
| A1 | Paired staining analysis | Key paper result | Low | Nothing |
| A2 | Site-holdout CV | Key paper result | Low | Nothing |
| A3 | Wagner zero-shot | Key paper result | Low | ctranspath |
| B1 | Autoresearch Tier 1 | 0.60→0.65 | Low | Nothing |
| B2 | PRISM/TITAN classifiers | 0.60→0.75+ | None | Aggregation |
| B3 | Multi-model fusion | 0.65→0.72 | Low | B1 |
| B4 | ABMIL | 0.75→0.85+ | Medium | Implementation |
| C1 | Tile QC filtering | +0.01-0.03 | Very low | Nothing |
| C2 | Stain normalization | Unknown | High | A1 result |
