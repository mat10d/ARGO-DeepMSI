# E4 — Wagner error anatomy: synthesis and the recipe

**Date:** 2026-08-04
**Cohort:** 217 patients / 803 slides / 47 MSI-H. Champion: frozen Wagner/CTransPath, max/√n.
**Operating point:** threshold 0.1375 at sensitivity 0.95.
**Inputs:** E1 (Stage A), E2 (Stage B), E3 (Stage C).

## The failure, in one line

At the screening operating point Wagner makes **146 false positives vs 2 false negatives**. The
problem is **specificity** — Wagner over-calls MSI-H on Nigerian MSS tissue — not sensitivity. It
catches the disease; it cannot rule it out.

## Decomposition of the 148 patient-level errors

| Cause | count | % of errors | evidence | circumvention lever |
|---|---:|---:|---|---|
| borderline-score | 57 | 39% | near a very low (0.137) threshold | operating point / **selective abstention** |
| unexplained (confident FP) | 43 | 29% | E3: attention tracks tumor normally | **new signal / abstention** — not another head |
| label-suspect | 32 | 22% | enriched 1.66× (indeterminate→MSS) | **re-adjudication worklist** (written) |
| low-tumor-content | 16 | 11% | enriched 2.80× | **re-tile / QC-refix worklist** (written) |

`high-artifact` was **excluded**: it looked like 48% of errors under naive thresholding but is
*depleted* among errors (0.78×) — ubiquitous, not causal. Chasing an artifact clean-up would not
help (consistent with D2).

## What each lever can and cannot do

1. **Data/QC work (~33%, addressable now).** The label-suspect (32) and low-tumor (16) buckets are
   genuinely enriched and come with concrete worklists. Re-adjudicating the indeterminate labels
   and re-tiling/dropping the tumor-poor slides is the highest-confidence, lowest-risk move, and it
   requires no new modelling. `worklist_label_adjudication.csv`, `worklist_qc_refix.csv`.
2. **Operating point / abstention (~39%).** The borderline bucket is MSS sitting just above a very
   low threshold. This is a deployment decision (selective rule-out on serviceable cases, abstain
   near the boundary), not a modelling failure — and it dovetails with the T-phase conformal work.
3. **The hard core (~29%).** Confident false positives where QC and labels do not explain the
   error and Stage C shows Wagner attends to tumor normally. This is a genuine representational
   limit: Wagner mis-reads Nigerian MSS morphology as MSI-H. The honest levers are new signal
   (acquisition-robust or differently-pretrained encoder — the Waiv test in progress) or adjudicated
   labels, plus abstention. **Another aggregator/attention head on the same CTransPath features will
   not fix it.**

## Cross-cutting evidence

- **Acquisition is mostly not the driver (E2).** Across paired MSK/OAU scans of the same tissue,
  only 4.8% of calls flip (r=0.81), with a modest OAU downward bias. This tempers the prior that an
  acquisition-robust encoder alone closes OAUTHC, and is consistent with A1/A2 failing.
- **OAUTHC concentrates the addressable buckets.** Of its errors, label-suspect (21) and borderline
  (19) dominate — arguing OAUTHC's deficit is as much label/threshold as irreducible biology.

## The recipe (what to do instead of training at no end)

1. Act on the two worklists (label re-adjudication, tumor-poor slide QC) — highest confidence.
2. Treat the borderline mass as an operating-point/abstention problem, not a modelling one.
3. For the hard 29%, the only defensible modelling moves are new signal (Waiv encoders; adjudicated
   labels) and honest abstention — validated on data that did not set the choice. Do not add heads
   on frozen CTransPath.
4. Re-run this anatomy after the Waiv encoders land to see whether the hard core shrinks.

Reproduce: `python -m argo_deepmsi.eval.error_anatomy` (A+B), `sbatch scripts/error_anatomy_c.sh` (C).
