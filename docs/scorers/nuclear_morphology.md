# nuclear_morphology

**Module:** `argo_deepmsi.scorers.nuclear_morphology`
**Resolution:** slide
**Trained on our cohort:** yes (LR head, 5-fold patient-grouped)

## Mechanism

Replaces 768-D foundation-model embeddings with ~10 interpretable
morphological features derived from nuclear segmentation +
classification, then fits a logistic-regression MSI head 5-fold on
our cohort.

Per slide:

1. Random subsample 100 tiles (256×256, 0.5 mpp).
2. Run **NuLite** (`zs.seg.cell_types(model='nulite', amp=False)`),
   which classifies every nucleus into one of 5 PanNuke classes:
   Neoplastic / Inflammatory / Connective / Dead / Epithelial.
3. Compute per-slide features:
   - 5 class fractions (`frac_neoplastic`, …)
   - 6 derived ratios: `til_density`, `stroma_ratio`,
     `immune_density`, `necrosis_fraction`, `mitotic_density`,
     `cells_per_tile`

A class-balanced `LogisticRegression(StandardScaler-piped)` is fit
under `StratifiedGroupKFold(patient_id, k=5)`. Patient aggregation =
max / √n.

## Inputs

- Per-slide `shapes/tiles` (re-tiled if needed)
- NuLite weights (auto-downloaded by lazyslide)

## Outputs

- `results/scorers/nuclear_morphology/slide_scores.csv` (canonical schema
  + raw cell counts + features)
- `results/scorers/nuclear_morphology/features.csv` (783 slides × 11 features)

## Results

| metric            | full cohort | QC-clean (refit) |
|-------------------|-------------|------------------|
| slide AUROC       | 0.438       | tbd              |
| patient AUROC     | 0.529       | **0.513**        |

QC-clean rank: **7 (last)**, but the gap to chance closed when the LR
was refit on QC-clean folds (0.473 post-hoc filter → 0.513 refit, +0.04).
783 / 808 slides retained at extraction (~97%); 25 dropped to a
lazyslide `concat` bug on mostly-blank slides.

## Interpretation

Cell-class composition features (TIL density, tumor / stroma / dead /
epithelial fractions, mitotic / necrosis fractions, cells / tile)
carry **no discriminative signal for MSI in this cohort**. Consistent
with the C5 finding that a tumor classifier failed, and with the
broader pattern that MSI signal here is sparse and not captured by
global composition.

NuLite was trained on PanNuke (Western tissue); absolute class
assignments on Nigerian tissue likely shift. But a per-class
calibration shift would still leave the relative MSI vs MSS contrast
intact if it existed — the signal is just not in this feature set on
this cohort.

## Caveats

- 100 random tiles per slide → cells/tile noise inherits 1/√k variance.
  Larger samples would tighten CIs but won't flip the headline.
- LR is linear; a tree-based head (XGBoost / RF) could squeeze 1–3 pts.
- The 25 lost slides are a known lazyslide bug; not a confound.

## When to revisit

- If a tumor-aware tile filter (`tumor_tile_classifier`) lands, re-run
  on tumor-tiles-only — composition statistics should be much more
  discriminative once stromal / fat / background tiles are stripped.
- Try HistoPLUS (richer immune-cell decomposition) once we get HF
  access to `Owkin-Bioptimus/histoplus`.
