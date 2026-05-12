# C3 — Quantitative MSI score from REDCap + Wagner-vs-score correlation

**Status:** ✅ 2026-04-22 — Step 1b of the runbook (blocked on "no IMPACT data")
is now unblocked via `cmo_msi_score` in REDCap. Step 3c (continuous
regression) ran and is a null result for linear readouts on our 123-patient
subset.

**Scripts:**
- `argo_deepmsi/data_ingestion.py` — `create_clinical_table` now carries
  `cmo_msi_score`, `cmo_msi_status`, `msi_method`, `msi_status_mmr`,
  `mlh1/msh2/msh6/pms2`, `sample_type`, and the three provenance fields
  (`tissue_processing_site`, `slide_staining_site`, `slide_imaging_site`)
  through to `clinical_table.csv`. Existing 6 core columns untouched.
  Re-run with `argo ingest`.
- `scripts/msiscore_regression.py` (SLURM: `msiscore_regression.sh`) —
  Ridge-regress log1p(cmo_msi_score) on each slide embedding, under
  GroupKFold(5) patient CV and leave-one-site-out. Reports Spearman ρ,
  Pearson r, RMSE on log scale, and AUROC at the molecular MSI-H
  threshold (score ≥ 10).

**Outputs:**
- `results/data/clinical_table.csv` now has 18 columns (12 new,
  same 217 patients, same `isMSIH` column bit-identical)
- `results/analysis/msiscore_regression/{regression_results.csv, oof_predictions.csv, patient_table.csv}`
- `results/analysis/wagner_zeroshot/wagner_vs_cmo_msi_score.png` — side-
  by-side patient/slide scatters, Wagner P vs cmo_msi_score

## What's in REDCap

`cmo_msi_score` is an MSIsensor-style percent-unstable-sites metric,
populated for the **prospective CMO-molecular arm only** (batch 3). Every
prospective record also carries a categorical `cmo_msi_status`, and the two
align cleanly with no overlap between classes:

| `cmo_msi_status` | n records | median score | range |
|---|---:|---:|---|
| Stable | 151 | 1.27 | 0.09 – 2.99 |
| Indeterminate | 77 | 4.70 | 3.02 – 9.88 |
| Instable (MSI-H) | 67 | 27.04 | 10.01 – 59.77 |

Retrospective records (batches 1, 2) have no `cmo_msi_score` by design —
they were labelled via local **MMR IHC** (fields `mlh1`, `msh2`, `msh6`,
`pms2` all populated, `msi_method=1`, `msi_status_mmr ∈ {1,2}`).

In our 217-patient cohort: **123 patients have a numeric `cmo_msi_score`**,
**94 patients** have the four IHC calls, and nothing overlaps. 191 REDCap
records are in the assay pipeline with no result yet — they're all patients
without matching slides so they don't enter our modelling.

## Wagner P(MSI-H) vs `cmo_msi_score`

Computed directly from `results/analysis/wagner_zeroshot/patient_scores.csv`
(`p_msih` = sigmoid of the transformer's CLS logit, averaged across the
patient's slides — see `scripts/wagner_zeroshot.py` lines 195-227).

| Scope | n | Pearson | Spearman | AUROC @ score≥10 |
|---|---:|---:|---:|---:|
| Patient, pooled | 123 | 0.166 | 0.113 | 0.588 |
| Slide, pooled   | 534 | −0.022 | 0.027 | 0.477 |

Per-DAG patient-level Pearson:

| DAG | n | Pearson | Spearman |
|---|---:|---:|---:|
| lasuth | 13 | **+0.590** | +0.374 |
| luth | 19 | **+0.461** | +0.263 |
| uith | 10 | **+0.483** | +0.600 |
| **oauthc** | **81** | **−0.045** | **+0.009** |

OAUTHC prospective carries 66% of the sample with ~zero correlation and
drags the pooled number down from ~0.5 to 0.17. On the other three sites
Wagner's probabilities track MSIsensor at r ≈ 0.45–0.60 — ordinary domain-
shift behaviour. OAUTHC is the single anomaly.

Wagner probability stratified by true status (patient level):

| status | n | mean Wagner P | mean score |
|---|---:|---:|---:|
| Stable (<3) | 56 | 0.394 | 1.44 |
| **Indeterminate (3–10)** | 40 | **0.361** | 4.81 |
| Instable (≥10) | 27 | 0.456 | 28.39 |

Wagner separates Instable from Stable by only ΔP≈0.06 and **cannot
distinguish Indeterminate from Stable at all** (0.361 vs 0.394). The MSS
bias from 1c is specifically against the upper-Indeterminate / Instable
zones on OAUTHC prospective.

## Step 3c — regressing `cmo_msi_score` on our embeddings

Ridge(α=1) on standardised features → log1p(score), 533 slides / 123
patients after aligning to CMO-score-available.

| embedding | regime | Spearman ρ | Pearson r | AUROC@10 |
|---|---|---:|---:|---:|
| conch_v1.5_mean | patient_cv | −0.039 | −0.010 | 0.539 |
| conch_v1.5_mean | loso       | −0.091 | −0.052 | 0.355 |
| ctranspath_mean | patient_cv | −0.225 | −0.085 | 0.365 |
| ctranspath_mean | loso       | −0.110 | −0.086 | 0.356 |
| uni2_mean       | patient_cv | −0.135 | −0.021 | 0.419 |
| uni2_mean       | loso       | −0.011 | +0.057 | 0.372 |
| virchow2_mean   | patient_cv | −0.122 | −0.048 | 0.412 |
| virchow2_mean   | loso       | −0.038 | +0.013 | 0.355 |
| conch_v1.5_titan| patient_cv | −0.095 | +0.036 | 0.457 |
| conch_v1.5_titan| loso       | −0.023 | −0.044 | 0.464 |
| virchow2_prism  | patient_cv | −0.043 | −0.028 | 0.519 |
| virchow2_prism  | loso       | −0.107 | −0.048 | 0.360 |

Null result across the board — no embedding / regime combination clears
|ρ|>0.2 or AUROC@10>0.55. A linear regressor trained on 123 patients from
scratch cannot extract a signal that Wagner (a 32M-parameter transformer
pre-trained on ~13k Western CRC slides) only extracts at r=0.17 pooled.

## Reading

- **The MSIsensor metric is real and well-calibrated** — zero overlap
  between status classes, clean bins at 3% and 10%.
- **Wagner picks up the continuous gradient on 3 of 4 prospective sites.**
  Where it works (LASUTH/LUTH/UITH), r ≈ 0.5. That's what a Western-trained
  zero-shot classifier should look like on African data.
- **OAUTHC prospective breaks this**, and the evidence converges across
  AUROC, correlation with the continuous metric, and error-covariate
  analysis. Whatever is different about OAUTHC-prospective slides
  (scanner, fixation, block age, case selection) is upstream of anything
  we can fix at the embedding or classifier level.
- **Linear regression from scratch is hopeless at this cohort size.** Step
  3c doesn't justify more hyperparameter tuning — it points straight at
  Step 4 (MIL with pre-training on TCGA).

## What's now reproducible

- `argo ingest` → `results/data/clinical_table.csv` with the 12 extra
  fields. Same 217 patients, `isMSIH` bit-identical to prior runs.
- `sbatch scripts/msiscore_regression.sh` → rebuilds the regression table
  end-to-end (live REDCap pull, aligns across the 6 embedding dirs, Ridge
  under both regimes). Needs `.env` with `REDCAP_API_{URL,TOKEN}`.
- `sbatch scripts/failure_diagnosis.sh` → regenerates 1a/1c/1d tables and
  plots (no REDCap needed — runs off CSVs + Wagner scores).
- `sbatch scripts/harmony_integrate.sh` → PCA(50) + Harmony over SITE on
  every embedding dir, writes `results/embeddings/<emb>_harmony/` and runs
  LOSO comparison.
- `sbatch scripts/fusion.sh` → stacking / late-average / concat / few-shot
  comparison on the base 4 + aggregator variants.
