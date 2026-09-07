# E1 — Wagner error anatomy, Stage A (covariate attribution)

**Date:** 2026-08-04
**Cohort:** 217 patients / 803 slides / 47 MSI-H (primary estimand).
**Champion:** frozen Wagner/CTransPath, max/√n patient aggregation.
**Operating point:** threshold 0.1375 at sensitivity 0.95.
**Reproduce:** `python -m argo_deepmsi.eval.error_anatomy` → `results/analysis/error_anatomy/`.

## Headline

At the sens-0.95 operating point Wagner makes **146 false positives vs 2 false negatives**
(TP 45, TN 24). The screening failure is almost entirely a **specificity** problem — Wagner
**over-calls MSI-H on Nigerian MSS tissue** — not a sensitivity problem. Specificity ≈ 0.14,
consistent with the frozen leaderboard.

## Method note — enrichment gating (why the first cut was wrong)

The naive attribution (assign any error whose covariate crosses an absolute threshold) tagged
**high-artifact as 48% of errors**. That is an artifact of ubiquity, not causation: artifact
burden is present in 87% of the whole cohort. The validity check is enrichment —
P(covariate | error) / P(covariate | correct):

| Covariate | P(cond \| error) | P(cond \| correct) | enrichment | causal? |
|---|---:|---:|---:|---|
| high-artifact (≥0.5) | 0.68 | 0.87 | **0.78×** | no — *depleted* |
| low-tumor-content (≤0.01) | 0.12 | 0.04 | **2.80×** | yes |
| label-suspect | 0.22 | 0.13 | **1.66×** | yes |

Among negatives, false-positive rate actually *falls* as artifact rises (1.00 in the lowest
artifact quartile → 0.74 in the highest). Attribution is therefore **enrichment-gated**: a
covariate bucket may claim an error only if it is over-represented among errors by ≥ 1.2×.
`high-artifact` is excluded. This matches the D2 verdict that hard artifact-QC exclusion does not
earn its place, and it prevents recommending an artifact clean-up that cannot help.

## Error Pareto (enrichment-gated)

| Cause | overall | OAUTHC | circumvention lever |
|---|---:|---:|---|
| borderline-score | 57 | 19 | operating-point / selective abstention |
| **unexplained** (confident, no enriched covariate) | 43 | 16 | Stage C: spatial attention introspection |
| label-suspect | 32 | 21 | molecular/pathology re-adjudication worklist |
| low-tumor-content | 16 | 3 | re-tile / stronger tumor detection (QC-refix) |

Worklists written: `worklist_label_adjudication.csv` (32 patients),
`worklist_qc_refix.csv` (low-tumor + retained artifact rows for review).

## Reading

- **~33% of errors (label-suspect + low-tumor) are addressable by data/QC work**, and both are
  genuinely enriched. These are the concrete, low-risk levers; the worklists are their inputs.
- **~39% are borderline** — MSS cases sitting just above a very low (0.137) threshold. This is an
  operating-point / abstention question, not a modelling one.
- **~29% are confident, unexplained false positives** — Wagner is confidently wrong on MSS
  tissue that QC and labels do not explain. This is the true hard core and the target of Stage C:
  is Wagner keying on genuine MSI-like morphology, or on a site/stain shortcut? OAUTHC carries 16
  of these 43.

## Caveats

- Covariate fractions are patient-level aggregates (mean tumor/artifact fraction, summed tiles).
- `mmr_cmo_concordance` is 0 everywhere: MMR status exists only for the retrospective cohort,
  where it *is* the label source, so there is no independent CMO/MMR cross-check to mine.
- `CONFIDENT_MARGIN=0.15` sets the borderline/unexplained split; recorded in `A_manifest.json`.
  The borderline bucket is sensitive to this constant and should be read as "near-threshold," not
  a hard category.
