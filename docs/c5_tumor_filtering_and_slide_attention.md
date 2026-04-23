# C5 — Tumor Slide Filtering + Learned Slide Selection

The bag-size calibration finding (C4) showed that OAUTHC's multi-slide
bags contain non-informative slides that inflate false positives. This
doc specifies how to identify and filter those slides, and then how to
build the lightweight fine-tuning layer that is the paper's
methodological contribution.

---

## Phase 0: Preliminary Analysis — What Are the Low-P Slides?

Before building anything, characterize the slides that Wagner scores
low. This is pure analysis on existing data, no training.

### 0a. Threshold exploration

```python
# For each threshold in [0.05, 0.10, 0.15, 0.20, 0.25]:
#   - How many slides fall below?
#   - Which sites/patients do they come from?
#   - What's the tile-count distribution (low-P vs high-P)?
#   - What's their MSI-H prevalence vs the high-P slides?
#   - For multi-slide patients: how many slides per patient are low-P?

# Table: threshold × {n_slides_below, pct_of_total, by_site_breakdown,
#         median_tile_count, msi_h_prevalence}
```

### 0b. UMAP overlay

```python
# On existing UMAP coordinates (already computed in C4):
#   - Color slides by Wagner P with a diverging colormap
#   - Highlight slides below the 0.10 threshold as × markers
#   - Are low-P slides the UMAP outlier islands, or scattered?
```

If low-P slides cleanly map to the outlier clusters → filtering is
trivial (they're non-tumor). If they're scattered throughout the main
cluster → they're MSS-looking tumor, and filtering by Wagner P would
discard real signal.

### 0c. Patient-level impact test

```python
# For patients with mixed low-P and high-P slides:
#   - Recompute patient-level Wagner score using ONLY high-P slides
#   - Compare patient AUROC before vs after filtering
#   - Stratify by site and slide-count bin
#   - Specifically: do the 7+ slide OAUTHC patients recover?
```

This is the critical test. If dropping low-P slides recovers the 7+ bin
from 0.25 to >0.60, then filtering is the fix and the question is just
which filtering method is most principled for the paper.

### 0d. Visual inspection

```python
# Sample 20 low-P slides and 20 high-P slides
# For each: generate a thumbnail with tissue overlay from the zarr
# Visually confirm: are low-P slides non-tumor (fat, normal, lymph node)?
# This is the ground truth sanity check
```

Save the thumbnails — they're paper supplementary material showing the
problem.

**Decision point:** If 0b shows clean UMAP separation AND 0c shows AUROC
recovery, proceed to Phase 1. If low-P slides are scattered and
filtering doesn't help, skip to Phase 2 directly (the problem is
something else).

---

## Phase 1: Tumor Slide Filtering

Three approaches, ordered by rigor. Try all three, compare.

### 1a. Wagner P threshold (simplest, least principled)

```python
# Hard filter: drop slides with Wagner P < threshold
# Threshold selected from Phase 0a based on AUROC recovery
# Pro: zero additional computation
# Con: conflates "non-tumor" with "MSS-looking tumor"
```

### 1b. Embedding-based outlier detection (unsupervised)

```python
# Per foundation model:
#   1. Compute slide-level embedding centroid (from retrospective
#      slides as reference — these are curated tumor)
#   2. Cosine distance from each slide to the reference centroid
#   3. Slides beyond 2σ or 3σ are flagged as non-tumor
#
# Alternative: HDBSCAN on the UMAP coordinates
#   - The outlier clusters get label=-1 (noise)
#   - Main cluster = tumor

from sklearn.cluster import HDBSCAN
clusterer = HDBSCAN(min_cluster_size=20)
labels = clusterer.fit_predict(umap_coords)
non_tumor_mask = labels == -1
```

Pro: uses the embedding structure directly, no Wagner dependency.
Con: may be sensitive to UMAP hyperparameters.

### 1c. Tissue classifier head on CTransPath (most principled)

The NCT-CRC-HE-100K dataset provides a 9-class tissue type classifier
trained on CTransPath features. If a compatible linear head exists, we
can classify each TILE (not slide) as:

```
ADI (adipose), BACK (background), DEB (debris), LYM (lymphocytes),
MUC (mucus), NORM (normal), STR (stroma), TUM (tumor)
```

Then compute per-slide tumor fraction:
```python
# Per slide:
#   tile_labels = tissue_head(tile_embeddings)  # from zarr
#   tumor_fraction = (tile_labels == "TUM").mean()
#   slide is "tumor" if tumor_fraction > 0.3 (or calibrate)
```

**What to check:**
- Does a pre-trained NCT-CRC tissue classifier head for CTransPath
  exist? (Check KatherLab GitHub, HuggingFace, STAMP repo)
- If not for CTransPath, does one exist for UNI2 or CONCH?
- Alternatively: train a simple logistic regression on NCT-CRC-HE-100K
  tile embeddings (100K labeled tiles, 9 classes) — the tile embeddings
  are the foundation model features, the head is a single linear layer.
  This takes minutes to train.

**If no pre-trained head exists:**
- Download NCT-CRC-HE-100K (public, 100K labeled 224×224 patches)
- Extract CTransPath/UNI2/CONCH features from those patches
- Train logistic regression → 9-class tile classifier
- Apply to your zarr tile features → per-tile tissue labels
- Aggregate to slide-level tumor fraction → filter

This is lightweight (the extraction is one GPU hour on 100K patches)
and gives you a principled tissue classifier for the paper.

**For the paper:** 1c is the approach you'd report. 1a and 1b are
validation/ablation — show that simpler methods also work but are less
principled.

---

## Phase 2: Learned Slide Attention (the methodological contribution)

After filtering non-tumor slides (Phase 1), the remaining slides still
vary in informativeness. Learn which slides carry the MSI signal.

### Architecture

```python
class SlideAttentionMSI(nn.Module):
    """Learned attention over slides within a patient.

    Input: per-slide features (n_slides × D)
    D can be: Wagner P (1) + foundation model embedding (768)
    Output: single patient-level MSI probability
    """
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1),
        )
        self.head = nn.Linear(input_dim, 1)

    def forward(self, slide_features):
        # slide_features: (n_slides, D) — one patient's bag
        w = torch.softmax(self.gate(slide_features), dim=0)  # attention
        pooled = (w * slide_features).sum(dim=0)              # weighted mean
        return self.head(pooled).squeeze()                     # logit
```

### Input features per slide

For each slide in a patient's bag, concatenate:
```python
features = torch.cat([
    wagner_p,                    # (1,) — Wagner MSI probability
    foundation_embedding,        # (768,) — e.g., conch_v1.5 mean
    log_n_tiles,                 # (1,) — log(tile count), size proxy
    tumor_fraction,              # (1,) — from Phase 1c classifier
], dim=-1)
# Total: 771 dims → hidden_dim 64 → 1 attention weight
```

The tumor_fraction from Phase 1 is itself a feature — the model can
learn to downweight slides with low tumor content rather than hard-
filtering them.

### Training

- 5-fold StratifiedGroupKFold (patient-level)
- Epochs: 100, early stopping patience 15 (validation AUROC)
- Optimizer: Adam, lr=1e-3, weight_decay=1e-2
- Loss: BCE with pos_weight=4.2 (class imbalance)
- Batch: 1 patient (standard for MIL — variable bag size)
- Regularization: dropout 0.5, small hidden_dim (64)

~5,000 parameters total. Trainable on 217 patients without overfitting
concerns, especially with the heavy regularization.

### Ablation table for the paper

| Model | Description | Params | AUROC |
|---|---|---|---|
| Wagner mean | Baseline zero-shot mean pooling | 0 | 0.659 |
| Wagner max/√n | Bag-size calibration | 0 | 0.717 |
| Wagner + tumor filter | Filter non-tumor slides, then mean | 0 | ? |
| SlideAttn (Wagner P only) | Learned attention on 1D scores | ~200 | ? |
| SlideAttn (Wagner + emb) | + foundation model embedding | ~5K | ? |
| SlideAttn (+ tumor frac) | + tissue classifier output | ~5K | ? |

Each row adds one component. The lift from each row is the paper's
evidence that the component matters.

---

## Phase 3: Multi-Model Slide Attention Fusion

Train separate SlideAttentionMSI per foundation model:

```python
# Per model: different embedding, same Wagner P + metadata
score_conch = slide_attn_conch(patient_slides_conch)
score_virchow2 = slide_attn_virchow2(patient_slides_virchow2)
score_uni2 = slide_attn_uni2(patient_slides_uni2)
score_ctranspath = slide_attn_ctranspath(patient_slides_ctranspath)

# Late fusion: average logits
p_final = sigmoid(mean([score_conch, score_virchow2,
                         score_uni2, score_ctranspath]))
```

Each model attends to different slides — virchow2 may find signal in
different blocks than conch_v1.5. The ensemble captures complementary
morphological views.

### Attention visualization (paper figure)

For a representative OAUTHC patient with 10+ slides:
```python
# Show all slides as thumbnails, sized by attention weight
# Color by true MSI score
# Side-by-side: attention from conch vs virchow2 vs uni2
# Caption: "The models attend to different tissue blocks, suggesting
#  complementary morphological cues for MSI prediction"
```

---

## Phase 4: ABMIL on Tile Features (if needed)

If Phases 1-3 don't reach 0.80+, the remaining lever is tile-level
attention within the selected slides. The slide attention from Phase 2
selects WHICH slides matter; ABMIL determines WHICH TILES within
those slides carry the MSI signal.

This is the full two-level MIL architecture:
```
tiles → ABMIL → slide embedding → SlideAttn → patient prediction
```

But defer this until Phases 0-3 results are in. The simpler approach
may be sufficient, and the paper is cleaner with fewer moving parts.

---

## Execution Order

```
IMMEDIATE (Phase 0 — pure analysis, existing data):
  ├─ 0a: Threshold exploration (Wagner P < 0.05/0.10/0.15/0.20/0.25)
  ├─ 0b: UMAP overlay of low-P slides
  ├─ 0c: Patient-level AUROC with low-P slides dropped
  └─ 0d: Visual inspection of 20 low-P vs 20 high-P slide thumbnails

DECISION POINT: Does filtering help?

IF YES → Phase 1 (tumor filtering):
  ├─ 1a: Wagner P threshold baseline
  ├─ 1b: Embedding-based outlier detection (HDBSCAN)
  ├─ 1c: Tissue classifier head (NCT-CRC-HE-100K)
  └─ Re-run calibrated aggregation on filtered bags

THEN → Phase 2 (learned slide attention):
  ├─ Implementation (~50 lines PyTorch)
  ├─ Train with GroupKFold on 217 patients
  └─ Ablation table (each component's contribution)

THEN → Phase 3 (multi-model fusion):
  ├─ Per-model slide attention training
  ├─ Late fusion
  └─ Attention visualization for paper

IF STILL <0.80 → Phase 4 (tile-level ABMIL)
```

---

## Paper Narrative

"Western-trained MSI classifiers achieve clinical-grade performance
(AUROC 0.80-0.93) on curated single-slide African specimens, but
collapse (AUROC 0.44) on uncurated multi-slide bags from Nigerian
clinical workflow. We identify bag-size inflation — a previously
undocumented failure mode where non-tumor slides dilute the signal in
large multi-slide bags — and propose a lightweight adaptation pipeline:
(1) tissue-type filtering using foundation model embeddings,
(2) learned slide attention that selects informative slides within
each patient's bag, and (3) multi-model fusion across pathology
foundation models. This pipeline recovers clinical-grade performance
(AUROC X.XX) without requiring large labeled training sets, using only
217 Nigerian CRC patients for fine-tuning."
