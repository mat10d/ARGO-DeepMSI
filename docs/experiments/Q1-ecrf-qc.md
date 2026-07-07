# Q1 — eCRF QC layer → cohort_clean.csv v0

## Method
Built the canonical clean-cohort table from the frozen inputs only (no new data):

1. **Join.** `slide_table_pyramidal.csv` (808 slides) ⋈ `clinical_table.csv` on `PATIENT`,
   label `y = (isMSIH == "MSI-H")`. Inner join covers all 808 slides (every slide has a
   clinical record).
2. **Tile counts.** `n_tiles` pulled from the widest cached `results/embeddings/*/metadata.csv`
   (tile counts are model-independent). 5/808 slides have no features (unreadable SVS — the
   documented `fastslide` failures).
3. **eCRF layer.** `passes_ecrf_qc = slide_id ∉ problem_slides.csv[passes_qc=False]`
   (514 pathologist-flagged names; 295 match a cohort slide).
4. **Clean set.** `in_clean_set = passes_ecrf_qc AND n_tiles > 0` — an unreadable slide can
   never enter training/scoring, so it is excluded even though it passed eCRF.

Entrypoint: `python -m argo_deepmsi.eval.cohort`
(writes `results/data/cohort_clean.csv` + `cohort_manifest.json`).
Columns: `slide_id, patient_id, site, y, n_tiles, passes_ecrf_qc, in_clean_set`.

## Headline result
Not a scorer — no MSIntuit operating point to report. This task *defines the cohort* every
later scorer is scored on. The clean set reproduces the established champion cohort exactly:
**509 slides / 198 patients, MSI-H prevalence 0.218.**

## Per-site drop table (eCRF layer)

| site | n_total | dropped_eCRF | missing_features | in_clean_set |
|---|---|---|---|---|
| LASUTH | 15 | 4 | 0 | 11 |
| LUTH | 31 | 10 | 0 | 21 |
| OAUTHC | 481 | 281 | 5 | 196 |
| UITH | 12 | 0 | 0 | 12 |
| retrospective_msk | 97 | 0 | 0 | 97 |
| retrospective_oau | 172 | 0 | 0 | 172 |

The eCRF exclusion is almost entirely an **OAUTHC** phenomenon (281/295 slide drops) — the
prospective Nigerian-imaged site. The retrospective MSK/OAU sets are untouched. This is the
first quantitative confirmation of the "OAUTHC problem" the R-phase tasks target.

## Verdict
Clean cohort v0 frozen at the eCRF layer. `in_clean_set` here is provisional — Q2 (artifact
QC) and Q3 (tumor filter) will AND further constraints in, and Q4 finalizes `in_clean_set` +
the no-regression floor. Gate green (`test_fit_only_sees_our_cohort_patients`).
