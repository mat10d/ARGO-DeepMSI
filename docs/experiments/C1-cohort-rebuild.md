# C1 — Full-cohort and validation reset

## What changed

The primary estimand is now every patient with at least one feature-complete slide:
**803 slides / 217 patients / 47 MSI-H**. Pathologist, artifact, and tumor-QC outputs remain
descriptive columns; the historical 428-slide / 181-patient subset is retained as
`in_qc_sensitivity_set`. `in_clean_set` is a compatibility alias for `in_primary_set`.

Patient cohort and slide processing location are now separate. The 83 patients with both
`retrospective_msk` and `retrospective_oau` slides form one patient cohort, `retrospective`.
Prospective `Indeterminate` remains MSS in the primary label policy and is excluded in a
pre-specified label-certainty sensitivity analysis.

## REDCap freshness

A read-only live REDCap audit on 2026-08-03 found the same 581 clinical identities as the
2026-04-22 local full snapshot. Seventeen non-benchmark patients have newly populated CMO
status/score fields, but none has a slide in the 808-slide corpus. All 217 benchmark patients
were found live, with zero changes across binary MSI labels, CMO status/score, MMR/IHC calls,
or acquisition-site fields. No local WSI or Halo export is newer than the benchmark snapshot.
Thus the current image benchmark is label-current; the broader `clinical_table_full.csv` is
stale for those 17 patients and should be refreshed before they acquire slides or enter a
future cohort.

## Rebaseline

| estimand | patients | positives | Wagner AUROC (95% bootstrap CI) | spec@sens95 |
|---|---:|---:|---:|---:|
| primary full | 217 | 47 | **0.717 (0.631–0.799)** | 0.141 |
| historical QC sensitivity | 181 | 41 | 0.713 (0.625–0.795) | 0.179 |
| definite labels only | 177 | 47 | 0.719 (0.629–0.805) | 0.123 |

On the 181 shared patients, re-adding their QC-excluded slides changes AUROC by +0.019,
but the paired 95% bootstrap interval crosses zero (−0.006 to +0.047). The earlier claim that
the full-cohort ceiling was ~0.732 was therefore a selected-patient result, not the 217-patient
estimand.

Primary Wagner per-cohort AUROC: LASUTH 0.933, LUTH 0.854, OAUTHC 0.616, UITH 0.750,
retrospective 0.782. Small-site estimates remain highly uncertain.

## Validation reset

`nested_linear_probe` selects among CONCH-mean, CTransPath-mean, UNI2-mean, and
Virchow2-mean only inside each outer training fold. Scaling and LR fitting are fold-local;
three repeated 5-fold outer splits had zero patient overlap.

The valid nested result is **AUROC 0.530 (0.429–0.636)**, versus ~0.656 for the prior grid
that selected its winning configuration on the same OOF predictions it reported. Encoder
selection was unstable (UNI2 6 folds, CONCH 5, CTransPath 3, Virchow2 1), supporting a
selection-variance rather than a hidden-best-encoder interpretation.

## Unified board policy

The current leaderboard contains every usable cached scorer row, but it does not pretend that
all rows estimate the same thing. `comparable_primary` requires at least 98% coverage of the 217
patients, and `confirmatory_valid` excludes cohort-trained methods whose configuration was
selected on the OOF predictions it reports. The old 181-patient rows remain visible as
historical experiments. The expensive legacy `simple_grid` rerace was stopped after more than
five CPU-hours because refining a select-on-OOF row cannot change the confirmatory conclusion;
the valid nested replacement is complete.

## Verdict

The zero-shot task-specific Wagner representation remains the only reliable signal. Hard QC
does not show a statistically resolved benefit, but neither does the evidence justify claiming
that removing QC raises the 217-patient ceiling. Frozen generic encoders plus a cohort-trained
linear head are at chance once configuration selection is nested.
