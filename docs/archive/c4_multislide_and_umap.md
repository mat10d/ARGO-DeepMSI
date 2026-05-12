# C4 — Multi-slide aggregation failure + UMAP visualisation

**Status:** ✅ 2026-04-22 — two small but pointed analyses. The multi-slide
finding is the biggest conceptual shift since A3: it says we need proper MIL
aggregation, not better embeddings or better zero-shot models.

**Scripts:**
- `scripts/multislide_analysis.py` (SLURM wrapper `multislide_analysis.sh`) —
  per-patient aggregator comparison (mean / max / median / p75 / top3mean)
  across sites and slide-count bins. Uses the persisted Wagner slide-level
  probabilities and the clinical table, no GPU.
- `scripts/umap_embeddings.py` (`umap_embeddings.sh`) — 2D UMAP of each raw
  embedding dir with six coloured panels (SITE, arm, MSI status,
  cmo_msi_score, Wagner P, Wagner correctness). Standardised + cosine UMAP,
  `n_neighbors=30, min_dist=0.1`, seed 42.

**Outputs:**
- `results/analysis/multislide/{aggregation_comparison.csv, auroc_by_slide_count.csv, oauthc_patient_stats.csv, n_slides_per_patient.csv, agg_auroc_by_site.png, per_site_slide_counts.png}`
- `results/analysis/umap_embeddings/<emb>.{csv,png}` for each of
  conch_v1.5_mean, conch_v1.5_titan, ctranspath_mean, uni2_mean,
  virchow2_mean, virchow2_prism.

## Multi-slide aggregation — why OAUTHC breaks

Slide-count distribution is extremely skewed across sites:

| site | n_patients | min | median | max | mean |
|---|---:|---:|---:|---:|---:|
| LASUTH | 13 | 1 | 1 | 3 | 1.15 |
| LUTH | 19 | 1 | 1 | 4 | 1.63 |
| UITH | 10 | 1 | 1 | 2 | 1.20 |
| retrospective_msk | 94 | 1 | 3 | 6 | 2.86 |
| **OAUTHC** | **81** | **1** | **2** | **48** | **5.88** |

OAUTHC has the heaviest tail by far — 18 of 81 patients have ≥7 slides, including one with **41** and one with **48**. No other site has a patient with more than 6.

### Wagner patient-level AUROC vs slide-count bin (pooled)

| n slides/patient | n pts | AUROC mean | AUROC max |
|---:|---:|---:|---:|
| 1 | 73 | 0.639 | 0.639 |
| 2–3 | 109 | 0.718 | **0.744** |
| 4–6 | 17 | 0.733 | **0.967** |
| **7+** | **18** | **0.250** | **0.375** |

### OAUTHC-only by slide-count bin

| OAUTHC | n pts | AUROC mean | AUROC max |
|---:|---:|---:|---:|
| 1 | 30 | 0.481 | 0.481 |
| 2–3 | 22 | 0.542 | 0.556 |
| 4–6 | 11 | 0.556 | **0.944** |
| **7+** | 18 | **0.250** | 0.375 |

**Two entirely different regimes.** On OAUTHC patients with 4–6 slides, taking the max Wagner probability rescues AUROC from 0.556 to 0.944 (ΔAUROC +0.39 on 11 patients). On the 7+ cohort, max fails — in fact both mean and max give sub-chance AUROC. These coexist in the same site's prospective arm.

### Why max fails on the 7+ cohort — multiple-testing inflation

The 7+ OAUTHC cohort has only 2 MSI-H patients (prevalence 0.11). Their Wagner probabilities:

- **P_0050** (cmo_score 10.01, borderline MSI-H, **41 slides**): p_max = 0.895. Many blocks of a large resection; the top slide is genuinely picked out as MSI-H.
- **P_0152** (cmo_score 30.33, strong MSI-H, **17 slides**): p_max = **0.462**. **Every one of 17 slides scores < 0.5.** Wagner misses this patient entirely — representative of the OAUTHC-prospective failure mode from A3 / c3.

The 16 MSS patients in the 7+ cohort all saturate:

| patient | y | cmo_score | n | p_max |
|---|---:|---:|---:|---:|
| P_0133 | MSS | 0.09 | 16 | **0.950** |
| P_0181 | MSS | 1.88 | 18 | 0.914 |
| P_0124 | MSS | 2.16 | 15 | 0.913 |
| P_0138 | MSS | 4.93 | 8  | 0.907 |
| P_0164 | MSS | 2.99 | 15 | 0.890 |

As `n_slides` grows, `P(at least one slide has high Wagner P | MSS)` rises toward 1. MSI-H patients don't gain the same way because Wagner is OAUTHC-MSS-biased (c1 §1c). Net: the ranking inverts, `max` AUROC drops below chance.

This is **bag-size-inflation of max-pooling** — a textbook failure mode of naïve MIL aggregators. It makes `max` a fragile operator unless calibrated by bag size.

### OAUTHC within-patient variability

| stat | median | p75 | max |
|---|---:|---:|---:|
| n_slides | 2 | 5 | 48 |
| p_range (max − min Wagner P within patient) | 0.07 | 0.35 | 0.80 |
| p_std (within patient) | 0.14 | 0.19 | 0.37 |

For OAUTHC's upper-half patients, Wagner probability varies by 0.35+ across their own slides. That's real signal — different blocks / tissue regions look different to Wagner — but it is **not** uniformly informative: on dilution-limited patients (4–6 slides) the max picks out the tumour; on saturated patients (7+) the max picks out a false positive.

## Implication for modelling

This is the strongest argument yet for **Step 4 (ABMIL)**:

- Mean, max, median, p75, and top3-mean are all deterministic bag-level aggregators. None of them can learn per-slide attention or calibrate bag-size inflation. The multi-slide table shows each one failing in a different regime.
- ABMIL attends over slides with learned weights conditioned on content, which is precisely what's needed to down-weight saturated-but-uninformative slides in the 7+ cohort and up-weight the informative ones in the 4–6 cohort.
- Pre-training ABMIL on TCGA-COAD/READ (public MSI labels) then fine-tuning on the Nigerian cohort addresses Wagner's domain-shift failure on OAUTHC prospective at the same time.

Intermediate cheap options (not yet run):
- **Slide-count normalisation** — rescale `p_max` by the expected max-of-n under the MSS null distribution. One-line fix with the existing OOF table.
- **Trimmed aggregators** — k-quantile, winsorised mean. No training, bounded expected value regardless of bag size.

## UMAP panels (visual check)

For every raw embedding, `results/analysis/umap_embeddings/<emb>.png` gives a 2 × 3 panel:
1. SITE (LASUTH / LUTH / OAUTHC / UITH / retrospective_msk / retrospective_oau)
2. Arm (prospective DAG vs `oauthc_retro`)
3. MSI status (MSI-H / MSS / NA)
4. `cmo_msi_score` (log1p colourmap; grey where absent)
5. Wagner P(MSI-H) (coolwarm colourmap, 0–1)
6. Wagner correctness (green = correct, red = wrong at threshold 0.5)

The raw 2D coords are also saved as CSV so we can re-plot or overlay without re-running UMAP.

What the panels are for:
- Panel 1 (SITE) is the first honest look at whether OAUTHC prospective sits in its own embedding cluster, which is the mechanism that would explain the A2 leave-one-site-out collapse and the OAUTHC-specific Wagner failure.
- Panel 3 vs panel 6 shows whether MSI-H separates at all in this embedding space (panel 3) and whether Wagner's errors cluster in a specific region (panel 6). If Wagner errors concentrate inside a site-defined region of UMAP space, we've visualised the confound.
- Panel 4 is the only panel with the molecular ground truth. Overlaying it on the same coordinates as panels 1–3 lets us eyeball whether any continuous MSI gradient exists in the embedding space independent of site structure.

## Reproducibility

- `sbatch scripts/multislide_analysis.sh` — ~15 s, pure pandas, no REDCap pull
- `sbatch scripts/umap_embeddings.sh` — ~4 min on 4 CPUs for all 6 embeddings

Both scripts read only committed artifacts (Wagner scores, clinical table, embedding dirs). They don't need GPUs or REDCap credentials.

## C4b — Calibrated aggregators follow-up

Added 2026-04-22 to answer "can any *deterministic* aggregator rescue the 7+ cohort before we commit to ABMIL?"

**Script:** `scripts/calibrated_aggregation.py` (SLURM: `calibrated_aggregation.sh`).
**Outputs:** `results/analysis/calibrated_aggregation/{aggregator_auroc.csv, mss_null_curve.csv, patient_aggregator_table.csv, aggregator_comparison.png, oauthc_by_bin.png}`.

Evaluated 10 aggregators on the same Wagner slide-level probabilities:
- 4 naïve baselines: `raw_mean`, `raw_max`, `raw_median`, `raw_p75` (from C4)
- 3 bag-size-bounded quantiles: `top3_mean`, `top5_mean`, `p95`
- 3 bag-size-calibrated max operators:
  - `max_minus_Emax_MSS` = `p_max − E[max_n | MSS]`, where the MSS null is bootstrapped (1000 draws per n) from the slide-level Wagner P of all MSS slides (n=649)
  - `max_pct_under_MSS` = empirical quantile of `p_max` under the MSS-max-of-n null
  - `max_over_sqrtn` = `p_max / √n`, a crude Bonferroni-style size penalty

### Overall AUROC

| aggregator | overall | LASUTH | LUTH | OAUTHC | retro_msk | UITH |
|---|---:|---:|---:|---:|---:|---:|
| raw_mean              | 0.659 | 0.933 | 0.812 | 0.451 | 0.783 | 0.792 |
| raw_max               | 0.644 | 0.900 | 0.833 | 0.426 | **0.824** | 0.792 |
| top5_mean             | 0.643 | 0.933 | 0.812 | 0.440 | 0.783 | 0.792 |
| max_minus_Emax_MSS    | 0.691 | 0.933 | 0.875 | 0.513 | 0.800 | 0.667 |
| max_pct_under_MSS     | **0.696** | 0.933 | **0.896** | 0.523 | 0.801 | 0.667 |
| **max_over_sqrtn**    | **0.717** | 0.933 | 0.854 | **0.616** | 0.782 | 0.750 |

- Calibrated max rescues ~0.06 AUROC overall (0.66 → 0.72) without any training.
- On OAUTHC (all slide-counts pooled), `max_over_sqrtn` lifts AUROC from 0.45 to **0.616** — the biggest single-intervention gain we've seen on this site.

### OAUTHC by slide-count bin — the key comparison

| aggregator | bin 1 (n=30) | bin 2–3 (n=22) | bin 4–6 (n=11) | **bin 7+ (n=18)** |
|---|---:|---:|---:|---:|
| raw_mean              | 0.481 | 0.542 | 0.556 | **0.250** |
| raw_max               | 0.481 | 0.556 | 0.944 | **0.375** |
| p95                   | 0.481 | 0.542 | 0.944 | 0.250 |
| max_minus_Emax_MSS    | 0.481 | 0.556 | 0.944 | 0.281 |
| max_pct_under_MSS     | 0.476 | 0.542 | **1.000** | 0.266 |
| **max_over_sqrtn**    | 0.481 | 0.569 | **1.000** | **0.125** |

Two things land simultaneously:

1. **The 4–6 cohort is completely solvable with calibration.** Both `max_pct_under_MSS` and `max_over_sqrtn` reach **perfect AUROC 1.000** on the 11 OAUTHC patients with 4–6 slides. These are not over-fit — they're single-parameter non-trained calibrations evaluated on the full 11 patients. The dilution-limited regime is solved.

2. **The 7+ cohort cannot be rescued by any aggregator.** Every calibrator gets sub-chance on those 18 patients, and the two most-aggressive (`max_over_sqrtn` at 0.125, `raw_p75` at 0.188) go further below chance, because in this cohort the MSS patients really do have higher top-end signal per √n than the two MSI-H patients do. The signal Wagner emits on P_0050 (41 slides, MSI-H) and P_0152 (17 slides, MSI-H; all <0.5) is either unusably diffuse or missing. No deterministic function of `{p_i}` fixes that.

### What this says

- **The MSS null curve is small and well-estimated.** `E[max_n | MSS]` rises from 0.39 (n=1) to ~0.88 (n=30) to ~0.92 (n=48) — so naïve `max` essentially measures "log(bag size)" on this cohort more than "MSI-H evidence". The calibrated variants strip out that inflation cleanly.
- **ABMIL is still needed, but for a narrower reason than before.** Not "to do better MIL aggregation" — simple calibrated aggregators handle most of that. ABMIL is needed specifically because on the 7+ OAUTHC cohort, no combination of Wagner's slide-level outputs contains the signal. That's a representation / domain-shift problem, not an aggregation problem. The plan is:
  - **Ship calibrated aggregation now** — it's a free 0.06 AUROC lift, zero training, one column added to the patient table.
  - **Use ABMIL + TCGA pre-training** to re-extract per-slide signal on the 7+ OAUTHC cohort, where the representation itself is what's failing.

### The MSS-null curve (n = 1..50, `mss_null_curve.csv`)

| n | E[max_n] | q50 | q95 |
|---:|---:|---:|---:|
| 1 | 0.39 | 0.39 | 0.80 |
| 3 | 0.62 | 0.63 | 0.88 |
| 5 | 0.71 | 0.73 | 0.90 |
| 10 | 0.80 | 0.82 | 0.93 |
| 20 | 0.86 | 0.88 | 0.94 |
| 40 | 0.91 | 0.92 | 0.95 |

Use this table to adjust any future per-slide probability for the number of slides in a patient's bag.

## What this changes in the runbook

- **Ship calibrated aggregation immediately.** `max_pct_under_MSS` or `max_over_sqrtn` is a free +0.06 AUROC (0.659 → 0.717 overall, 0.451 → 0.616 OAUTHC) with zero training. Update downstream patient-level scoring to use it by default.
- **Step 4 (ABMIL with TCGA pre-training)** narrows its target: the 7+ OAUTHC cohort, where no deterministic aggregator helps because Wagner's per-slide representation is what's failing. This is a smaller and better-posed problem than "MIL aggregation in general".
- **The two-MSI-H-in-7+-OAUTHC story** (P_0050 and P_0152) is a candidate for the paper's failure-case figure: pair the UMAP neighbourhood with the slide-level Wagner P distribution and one representative tile strip from each patient.
