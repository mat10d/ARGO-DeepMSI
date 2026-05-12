# C5 — Tumor Slide Filtering + Learned Slide Selection

The bag-size calibration finding (C4) showed that OAUTHC's multi-slide
bags contain non-informative slides that inflate false positives. This
doc specifies how to identify and filter those slides, and then how to
build the lightweight fine-tuning layer that is the paper's
methodological contribution.

---

## Phase 0 results (2026-04-23) — filtering does NOT recover the 7+ cohort

Ran `scripts/c5_phase0.py` on the persisted Wagner slide scores (803 slides,
217 patients). Outputs → `results/analysis/c5_phase0/`.

### 0a. Threshold sweep

| threshold | n_below | pct_below | MSI-H prev below | MSI-H prev above |
|---:|---:|---:|---:|---:|
| 0.05 |   3 |  0.4% | 0.333 | 0.191 |
| 0.10 |  19 |  2.4% | 0.211 | 0.191 |
| 0.15 |  64 |  8.0% | 0.172 | 0.194 |
| 0.20 | 101 | 12.6% | 0.149 | 0.198 |
| 0.25 | 151 | 18.8% | 0.152 | 0.201 |

MSI-H prevalence in low-P slides is not meaningfully depleted vs. the kept
pool — low-P slides are **not enriched for non-tumor across MSI-H**. At
t=0.10, dropping 19 slides would remove 4 MSI-H slides from 2 MSI-H patients.

Patients losing all slides at each threshold: 6 at 0.10 (1 MSI-H), 17 at
0.15 (1 MSI-H), 30 at 0.20 (2 MSI-H). Aggressive filtering costs data.

### 0c. Patient AUROC vs threshold (the critical test)

Overall (pooled, n=217):

| threshold | mean agg | max agg |
|---:|---:|---:|
| 0.00 (no filter) | 0.659 | 0.644 |
| 0.10 | 0.662 | 0.648 |
| 0.15 | 0.656 | 0.622 |
| 0.20 | 0.637 | 0.604 |

OAUTHC 7+ cohort (n=18, the failure mode we need to fix):

| threshold | mean agg | max agg |
|---:|---:|---:|
| 0.00 | **0.250** | 0.375 |
| 0.10 | 0.281 | 0.375 |
| 0.15 | 0.312 | 0.375 |
| 0.20 | 0.267 | 0.367 |
| 0.25 | 0.233 | 0.367 |

**The 7+ cohort does not recover.** AUROC peaks at 0.31 with aggressive
filtering — still far below chance-adjusted usefulness. The problem is
not that MSS patients accumulate false-positive slides; it is that the
one strong-MSI-H patient in the 7+ cohort (`P_0152`, cmo_score 30.3,
17 slides) has **every slide scoring < 0.5**. Wagner simply fails on
that patient's histology regardless of filtering.

### 0b. UMAP overlay (`umap_overlay_*.png`)

Low-P slides (Wagner P < 0.10, n=19) are scattered across every
embedding's UMAP — they do not map to isolated outlier islands.
Consistent across `conch_v1.5_{mean,titan}`, `virchow2_{mean,prism}`,
`uni2_mean`, `ctranspath_mean`.

### Decision (narrow — see open questions below)

Wagner-P-threshold filtering is ruled out on two grounds: (1) it fails
to recover the 7+ OAUTHC cohort, and (2) at t=0.10 it would strip MSI-H
signal from exactly the patients Wagner already struggles with. We do
**not** yet have grounds to skip Phase 1 entirely — 1a is dead, but 1b
(embedding-outlier filtering) and 1c (tile-level tissue classifier) are
orthogonal signals that haven't been tested.

### Open questions — what Phase 0 did NOT answer

1. **Phase 1b — UMAP outlier clusters.** C4 showed isolated outlier
   islands in every embedding. What's in them? HDBSCAN on UMAP coords
   (or cosine distance from retrospective-MSK centroid) would identify
   those slides without inheriting Wagner's miscalibration. Low-P-scatter
   only rules out Wagner P as the filter signal; it doesn't rule out
   filtering.

2. **Phase 1c — NCT-CRC-HE-100K tissue classifier.** The principled
   method the paper would report. A 9-class linear head on CTransPath
   (or UNI2) embeddings trained on 100K labeled tiles → per-slide
   tumor fraction. Orthogonal to both Wagner and UMAP. A slide that is
   5% tumor / 80% adipose gets filtered even if Wagner happened to
   score it high.

3. **MSS specificity, not just 7+ MSI-H sensitivity.** The 7+ cohort
   fails because P_0152 is unrescuable *and* MSS patients saturate via
   bag-size inflation. Filtering might not save P_0152 but could still
   reduce MSS false-positives in the 4-6 and 2-3 bins — a win the
   overall-AUROC table hides because 7+ dominates OAUTHC failure. Needs
   a per-bin AUPRC + calibration-error readout, not just AUROC.

4. **Within-patient stability for Phase 2.** Cleaner input slides make
   Phase 2 attention learnable. Even if filtering doesn't help the
   mean/max baseline, it could help downstream. That's not tested here.

5. **Characterize the outlier UMAP clusters themselves.** C4 mentions
   them; Phase 0 never described what they are (site, patient, tile
   count, stain, Wagner P distribution, MSI-H prevalence).

### Proposed next step

Run (a) HDBSCAN outlier characterization (~CPU hours) before committing
to (b) NCT-CRC-HE-100K training (~1 GPU-hour + download). (a) is cheap
enough to be worth knowing and informs whether (b) is worth the setup
cost. Only then is the Phase 1-vs-Phase-2 call well-founded.

---

## Phase 1b results (2026-04-23) — embedding-outlier filtering also fails

Ran `scripts/c5_phase1b.py`: HDBSCAN on each UMAP + distance from the
retrospective_msk centroid (MSK tiles are curated tumor). Tested three
filters per embedding: `hdbscan_noise` (cluster=-1), `msk_dist_p95`
(top 5% by distance), `msk_dist_p99` (top 1%). Outputs →
`results/analysis/c5_phase1b/`.

### Overall AUROC vs filter (mean aggregator, n=217)

| filter | n_pat | AUROC | AUPRC | Brier |
|---|---:|---:|---:|---:|
| **none** | 217 | **0.659** | 0.386 | 0.202 |
| conch_v1.5_mean : hdbscan_noise | 215 | 0.658 | 0.385 | 0.202 |
| conch_v1.5_titan : hdbscan_noise | 199 | 0.634 | 0.352 | 0.205 |
| ctranspath_mean : hdbscan_noise | 216 | 0.660 | 0.387 | 0.202 |
| uni2_mean : hdbscan_noise | 205 | 0.646 | 0.365 | 0.204 |
| virchow2_mean : hdbscan_noise | 214 | 0.661 | 0.392 | 0.201 |
| msk_dist_p95 (best of set) | 210 | 0.660 | 0.411 | 0.204 |

**No filter moves overall AUROC beyond noise.** AUPRC nudges up ~0.02 for
distance-based filters (they drop some low-scoring slides indiscriminately,
lifting prevalence in the kept pool) but Brier and mean predictions barely
move. Several HDBSCAN filters make things worse.

### OAUTHC 7+ cohort (n=18, 2 MSI-H) — does not recover

Every filter leaves the 7+ AUROC between **0.23 and 0.30**. The core
problem is P_0152 (17 slides, all Wagner-P < 0.5) — no filter that spares
Wagner's predictions on other patients can fix this.

### MSS 4-6 bin — most filters hurt it

| filter | n_pat | AUROC | mean_P(MSS) |
|---|---:|---:|---:|
| none | 17 | **0.733** | 0.386 |
| conch_v1.5_mean : hdbscan_noise | 16 | 0.600 | 0.389 |
| ctranspath_mean : msk_dist_p95 | 15 | 0.643 | 0.386 |
| virchow2_mean : hdbscan_noise | 15 | 0.769 | 0.368 |

virchow2_mean HDBSCAN noise is the only filter that *improves* a specific
bin (4-6 from 0.733 to 0.769), and it only drops 26 slides. This is the
strongest signal against "filtering is the answer": most filters move
metrics by less than random resampling would.

### What the outlier clusters actually are

Inspection of `cluster_summary_*.csv` shows that the HDBSCAN outlier
islands are **not non-tumor** — they are site/stain-specific morphological
clusters:

- `uni2_mean` cluster 3 (25 slides, all OAUTHC, mean Wagner=0.73, MSI-H
  prevalence=**0.000**) — high-Wagner-P *MSS* false positives, not
  non-tumor.
- `uni2_mean` cluster 7 (37 slides, all OAUTHC, MSI-H prev=**0.486**,
  median n_tiles=9360) — a high-MSI-H cluster. Filtering these would
  *drop* signal.
- `virchow2_prism` produces only 2 clusters total (slide-encoder
  over-compresses) — unusable for cluster-based filtering.

### Updated decision

Both 1a (Wagner-P threshold) and 1b (embedding-outlier) are dead ends on
this cohort. The outlier islands are site-of-origin artifacts, not
non-tumor tissue. The remaining principled option is 1c — a **direct**
tissue classifier trained on labeled NCT-CRC-HE-100K tiles. That test
is running now.

---

## Phase 1c results (2026-04-23) — classifier works on NCT, fails to transfer

Full pipeline:
- `scripts/c5_phase1c_train.py` + `.sh` — download NCT-CRC-HE-100K from
  `DykeF/NCTCRCHE100K`, extract CTransPath features on 100K train + 7180
  holdout tiles, train 9-class multinomial logistic regression head.
  Variants: `nonorm` (default, raw H&E — matches CTransPath pretraining
  and our cohort extraction) and `norm` (Macenko-normalized).
- `scripts/c5_phase1c_apply.py` + `.sh` — load head, open each slide's
  `ctranspath_tiles` zarr, predict per-tile class, compute per-slide
  tumor fraction + mean P(TUM), test filtering thresholds.

### Classifier holdout (CRC-VAL-HE-7K, 7180 tiles)

| variant | overall acc | TUM precision | TUM recall | TUM F1 |
|---|---:|---:|---:|---:|
| norm (Macenko) | **0.959** | 0.992 | 0.965 | 0.978 |
| **nonorm (raw H&E)** | 0.771 | **0.957** | 0.910 | **0.933** |

Both variants are publication-grade on the NCT holdout. We use **nonorm**
for the apply because CTransPath was pretrained on raw H&E and our cohort
slides are not stain-normalized.

### Apply on 803 cohort slides — predictions collapse

Tumor fraction (argmax-TUM) summary across the whole cohort:

| stat | value |
|---|---:|
| mean | 0.127 |
| median | 0.047 |
| p75 | 0.190 |
| max | 0.848 |

**12.7% mean tumor fraction is implausible** for a cohort of tumor
resection slides. Even retrospective_msk (curated tumor specimens from
A3) shows only 8% tumor fraction:

| site | n | mean tumor_fraction | mean P(TUM) |
|---|---:|---:|---:|
| retrospective_msk | 97 | 0.080 | 0.081 |
| LASUTH | 15 | 0.090 | 0.090 |
| LUTH | 31 | 0.091 | 0.087 |
| OAUTHC | 476 | 0.120 | 0.121 |
| retrospective_oau | 172 | 0.173 | 0.169 |
| UITH | 12 | 0.279 | 0.266 |

Class composition is spread ~roughly evenly across the nine classes
(MUC 17%, DEB 14%, MUS 13%, TUM 13%, LYM 12%, BACK 11%, NORM 10%,
ADI 5%, STR 4%) — no dominant class, which is the signature of a
classifier voting among ambiguous/mixed tiles.

### Wagner P vs tumor_fraction

Correlation is **negative**:

| | p_msih | tumor_fraction | mean_p_tum |
|---|---:|---:|---:|
| p_msih | 1.000 | -0.129 | -0.118 |
| tumor_fraction | -0.129 | 1.000 | 0.998 |

Slides with higher predicted tumor content score slightly *lower* on
Wagner MSI-H, the opposite of what we'd expect if the classifier
reflected real tumor content.

### Filtering by tumor_fraction — actively harmful

Overall patient AUROC under tumor_fraction thresholds:

| filter | n_pat | AUROC | AUPRC |
|---|---:|---:|---:|
| none | 217 | **0.659** | 0.386 |
| tumor_frac < 0.1 | 123 | 0.551 | 0.217 |
| tumor_frac < 0.2 | 90 | 0.456 | 0.178 |
| tumor_frac < 0.3 | 60 | 0.332 | 0.146 |
| tumor_frac < 0.4 | 40 | 0.281 | 0.155 |
| tumor_frac < 0.5 | 28 | 0.258 | 0.166 |

OAUTHC 7+ cohort: `tumor_frac<0.1` collapses to n=5 patients (n_msih=1),
AUROC 0.50 — not a recovery, just noise from tiny sample.

### Why the classifier doesn't transfer

NCT-CRC-HE-100K tiles are **hand-curated to contain a single dominant
tissue type per 224×224 tile**. Our cohort uses unselected
`zs.pp.tile_tissues(tile_px=256, mpp=0.5)` which produces random tiles
inside detected tissue regions. Those tiles routinely contain mixed
tissues (tumor + stroma + muscle, tumor + mucus, etc.), and the
classifier — trained on the "one tile, one class" assumption — collapses
mixed tiles to whichever subtype is numerically dominant.

The net is: the classifier discriminates tumor vs. non-tumor tiles as
defined by the *NCT training distribution*, but that decision boundary
is not useful on our *mixed-content resection tiles* from unselected
tissue regions.

### Conclusion — all three filtering methods fail

1. **1a (Wagner-P threshold)**: no AUROC movement; would strip MSI-H
   signal.
2. **1b (UMAP-outlier)**: outlier clusters are site-of-origin artifacts,
   not non-tumor; no AUROC movement.
3. **1c (NCT-CRC tissue classifier)**: classifier is strong on NCT
   holdout but fails to transfer; filtering actively *hurts* AUROC.

Tumor-slide filtering as an MSI prediction intervention is the wrong
lever on this cohort. The Nigerian failure mode is not "non-tumor
slides dilute signal" — it is "Wagner itself misses MSI-H morphology
on P_0152, and bag-size inflation accumulates MSS false positives on
the other 16 7+ patients." Only a learned aggregator can address both.

**Proceeding to Phase 2 — learned slide attention.**

### 0d. Thumbnails (`results/analysis/c5_phase0/thumbnails/`)

Rendered 18 low-P + 20 high-P slide thumbnails with tissue contours
(one low-P slide's stem mis-matched the pyramidal table — not worth
chasing for one sample).

Reinforces the decision: **4 of the 19 low-P slides are MSI-H** —
3 from `P_0152` (the hard-miss MSI-H patient from C4) and 1 from
`P_0076`. Filtering at Wagner P < 0.10 would actively strip MSI-H
signal from the two patients Wagner already struggles with. Any
threshold-based filter has this property — the failure mode is not
"non-tumor dilution," it is "Wagner assigns low P to real MSI-H
tissue on this cohort."

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


---

## Phase 2 Implementation (2026-04-23) — SlideAttentionMSI

### Script: `scripts/c5_phase2_slide_attention.py`

Implements the learned slide attention model specified above. Key design
decisions:

**Architecture:** Two-head design — separate attention gate and classification
head, both with hidden_dim=64. The gate learns WHICH slides to attend to;
the head learns WHAT the attended representation predicts. This separation
prevents the attention from collapsing to uniform weights.

**Feature sets (ablation):**

| Config | Features | Dim | Purpose |
|---|---|---|---|
| `wagner_only` | Wagner P(MSI-H) | 1 | Can attention alone rescue? |
| `emb_only` | Foundation model embedding | 768-2560 | Pure embedding signal |
| `wagner+meta` | Wagner P + log(n_tiles) | 2 | Minimal informative features |
| `emb+meta` | Embedding + log(n_tiles) | 769-2561 | Embedding + size proxy |
| `wagner+emb` | Wagner P + embedding | 769-2561 | Full signal, no metadata |
| `all` | Wagner P + embedding + log(n_tiles) | 770-2562 | Kitchen sink |

Crossed with 4 foundation models: conch_v1.5_mean, virchow2_mean,
uni2_mean, ctranspath_mean → 24 configurations total (minus redundant
embedding-free repeats = 18 configs).

**Training:**
- 5-fold StratifiedKFold on 217 patients (patient-level, no leakage)
- BCE loss with pos_weight=4.26 (19% MSI-H prevalence)
- Adam(lr=1e-3, weight_decay=1e-2)
- Early stopping: patience=20 on validation AUROC
- Per-patient batch size (standard MIL — variable bag size)
- ~5K parameters per model

**Baselines included:**
- Wagner mean pooling (AUROC 0.659) — zero parameters
- Wagner max/√n calibrated (AUROC 0.717) — zero parameters

**Outputs:**
- `ablation_results.csv` — full results table with per-site and per-bin AUROC
- `ablation_heatmap.png` — visual comparison
- `attention_weights.csv` — per-slide attention weights from best model
- `oof_predictions.csv` — out-of-fold patient predictions

### Key questions this answers:

1. **Can learned attention beat calibrated max?** (0.717 is the bar)
2. **Does the foundation embedding add signal beyond Wagner?**
   (wagner_only vs wagner+emb)
3. **Which foundation model's embedding is most useful for attention?**
4. **Does the 7+ slide OAUTHC cohort recover?** (per_bin AUROC)
5. **Is the model learning real biology or site artifacts?**
   (per_site AUROC breakdown)

### What "success" looks like:

- Overall AUROC > 0.72 (beating calibrated max)
- OAUTHC 7+ bin AUROC > 0.50 (above chance — currently 0.25-0.375)
- Attention weights correlate with Wagner P on non-OAUTHC patients
  (model agrees with Wagner where Wagner works)
- Attention weights DIVERGE from Wagner P on OAUTHC
  (model learns something Wagner can't see)

### If it doesn't work:

The 41 MSI-H patients may be insufficient to learn slide selection.
Next steps would be:
1. TCGA pre-training → Nigerian fine-tuning (Step 4 in runbook)
2. Tile-level ABMIL instead of slide-level attention
3. Accept the limitation and report the negative result as part of
   the "first West African MSI validation" contribution

---

## Phase 2 results (2026-04-23) — learned attention does NOT beat calibrated max

Full sweep ran cleanly (18 configs × 5 folds, ~45 min on one A6000).
Results in `results/analysis/c5_phase2/ablation_results.csv`, heatmap in
`ablation_heatmap.png`.

### Headline

| Model | AUROC | Params |
|---|---|---|
| **wagner_max/√n (C4b baseline)** | **0.717** | 0 |
| wagner_mean_pool (C4b baseline) | 0.659 | 0 |
| **Best learned:** wagner+meta (2D) | 0.665 | ~5K |
| Best embedding-based: virchow2 emb_only | 0.654 | ~5K |
| Worst: uni2 emb+meta | 0.560 | ~5K |

**Every learned configuration underperforms `wagner_max/√n`.** The
calibrated pooling aggregator from C4b (zero learned parameters) remains
the strongest patient-level MSI predictor on this cohort.

### Per-bin AUROC (best learned = wagner+meta)

| Slides/patient | n | AUROC |
|---|---|---|
| 1 | 73 | 0.675 |
| 2-3 | 109 | 0.735 |
| 4-6 | 17 | 0.733 |
| 7+ | 18 | **0.156** |

The 7+ bin (OAUTHC-heavy) collapses across *every* learned config
(range 0.16–0.66). Learned attention did not rescue the
more-slides-worse-AUROC pathology that motivated Phase 2.

### Interpretation

1. **Embedding signal is neutral-to-harmful.** Adding any foundation
   embedding to Wagner+meta reduces AUROC (0.665 → 0.60–0.63 across
   models). With 217 patients / 41 MSI-H, the attention gate can't learn
   to weight 768–2560-D embeddings better than a fixed prior.
2. **Attention over Wagner alone learns nothing new.** `wagner_only`
   (attention over 1D Wagner scores) scores 0.641 — worse than
   unweighted mean pooling (0.659). The learned gate is overfitting.
3. **`wagner_max/√n` is the right aggregator.** The bag-size-calibrated
   max captures what matters (one strongly-MSI-H slide is enough) and
   deflates inflated maxes from large bags. 5K-parameter attention
   can't beat it in the small-label regime.

### Decision

C5 is closed as a negative result. Neither tumor filtering (Phase 1b/1c)
nor learned slide attention (Phase 2) improves over the calibrated
`wagner_max/√n` aggregator from C4b. The OAUTHC 7+ degradation is
intrinsic to that cohort's slide mix (likely a sampling/staining artifact
per C4b analysis) and cannot be recovered from embeddings alone at this
sample size.

**Next directions** (not in C5 scope):
- TCGA pretraining → Nigerian fine-tuning (needs external cohort work)
- Tile-level ABMIL (full attention over tiles, not slides — ~1000× more
  parameters, needs proper compute budget)
- Accept and report: `wagner_max/√n` at AUROC 0.72 is the headline
  number for the "first West African MSI external validation" framing.

### Training health (fold-by-fold)

Across all 18 learned configs × 5 folds (90 models), validation AUROC
ranged **0.38–0.79** with early stops at epochs 21–69. Within each config
the 5 folds are highly dispersed — e.g. `wagner+meta`:
`[0.488, 0.576, 0.791, 0.644, 0.542]` (range 0.30, std ≈ 0.12). This
high fold variance is consistent with **insufficient positive-class
sample size** (9–10 MSI-H patients per val fold) rather than a training
bug: loss decreased, early stopping fired normally, no NaN/crash folds.
The result is a real floor, not a plumbing problem.

### Success-criteria checklist (pre-declared above)

| Criterion | Bar | Actual | Met? |
|---|---|---|---|
| Overall AUROC beats calibrated max | > 0.72 | 0.665 | ✗ |
| OAUTHC 7+ bin above chance | > 0.50 | 0.16 | ✗ |
| Attention agrees with Wagner off-OAUTHC | ρ > 0 | retro_msk +0.28, retro_oau +0.30 | ✓ |
| Attention diverges from Wagner on OAUTHC | ρ ≈ 0 or < | OAUTHC ρ = −0.05 (p=0.31) | ✓ |

So the attention mechanism *behaviorally* does what we hypothesized —
it learns that Wagner is informative on the retrospective cohorts and
uninformative on OAUTHC, and reweights accordingly. But on OAUTHC it has
no alternative signal to substitute (the embeddings carry no
MSI-discriminative information in the small-sample regime), so
"correctly ignoring Wagner" just means "predicting near-randomly."
This is strong evidence that the OAUTHC degradation is **not a slide
selection problem** — it's a signal-absence problem at the tile/slide
embedding level.

Attention also mildly anti-correlates with bag size
(Spearman −0.19 with `n_tiles`), so the gate is not just selecting the
biggest slide — a minor sanity check that it learned something
non-trivial.

### Reproduction

```bash
# From repo root, on a GPU node (single A6000 is fine)
sbatch scripts/c5_phase2.sh
# Runs scripts/c5_phase2_slide_attention.py end-to-end:
#   - 18 configs × 5 folds StratifiedKFold on 217 patients
#   - Writes results/analysis/c5_phase2/{ablation_results.csv,
#     ablation_heatmap.png, attention_weights.csv, oof_predictions.csv}
# Wall time: ~45 min on nvidia-A6000-20 partition
```

The two baseline rows (`wagner_mean_pool`, `wagner_max/√n`) are appended
to `ablation_results.csv` after training by `add_baselines()` in the
same script; they read from `results/analysis/wagner_zeroshot/` and
`results/analysis/calibrated_aggregation/` which must already exist
(both produced in earlier phases, checked into the repo).

Note: the 2026-04-23 run hit a `KeyError: 'y'` inside `add_baselines`
(duplicate column post-merge — fixed in commit adding this doc).
The crash was *after* the ablation was written to disk, so no learned
result was lost; the baseline rows were appended in-place from the
existing CSV.

### Artifacts

- `scripts/c5_phase2_slide_attention.py` — model + ablation + baselines
- `scripts/c5_phase2.sh` — SLURM wrapper (single A6000)
- `results/analysis/c5_phase2/ablation_results.csv` — 20 rows (2 baselines + 18 learned)
- `results/analysis/c5_phase2/ablation_heatmap.png` — embedding × feature-set heatmap
- `results/analysis/c5_phase2/attention_weights.csv` — per-slide attention (best config, all 803 slides × 5 folds)
- `results/analysis/c5_phase2/oof_predictions.csv` — per-patient OOF predictions
