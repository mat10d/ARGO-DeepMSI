# Wagner Error Anatomy — design spec

**Date:** 2026-08-04
**Status:** approved (design); pending implementation plan
**Branch:** autorun-scaffold
**Author:** Matteo DiBernardo + Claude

## Motivation

The project has run a broad, breadth-first search over methods (S/R/T/A/D/W phases) and
produced a large pile of negative results. The consistent finding is that the zero-shot
supervised Wagner/CTransPath model (patient AUROC **0.717**, 95% CI 0.631–0.799) is the only
reliable signal, and that every learned recipe over frozen embeddings, batch correction, stain
normalization, and PEFT fails to beat it. Nested validation (V1) confirms frozen generic
encoders plus a cohort-trained linear head are at chance (0.530).

We have **not** done the complementary depth-first work: understanding *which specific slides and
patients Wagner gets wrong, and why*. Wagner works because it is actually trained on the MSI
label; the fruitful next move is to anatomize its failures so we can **circumvent each failure
with knowledge of the challenge**, rather than training more models at no end.

The organizing principle: **every attributed error cause must map to a concrete circumvention
lever.** A diagnosis with no action is out of scope.

## Goal

Produce a per-patient/per-slide **error ledger** for the frozen Wagner champion on the 217-patient
/ 803-slide primary cohort, attribute each error to a cause, and translate the cause distribution
into a ranked set of circumvention levers. Deliver this in three staged passes (A → B → C), each
of which stands alone and produces an actionable artifact.

## Non-goals (YAGNI)

- No new classifier, no retraining, no threshold tuning for performance.
- No changes to the frozen Wagner weights or the frozen extraction contract.
- Stage C touches only the error subset, never the full cohort.
- No new data acquisition (the worklists this produces are *inputs* to a later, separate
  data/label effort governed by the roadmap's Phase 2).

## Data grounding (verified 2026-08-04)

All inputs already exist on disk:

- `results/scorers/wagner_zeroshot/slide_scores.csv` — 803 rows:
  `slide_id, patient_id, site, y, p_msih, stain_location`.
- `results/data/cohort_clean.csv` — 21 cols incl. `patient_id, site, n_tiles, artifact_fraction,
  n_tumor_tiles, tumor_fraction, in_qc_sensitivity_set, processing_site, patient_cohort,
  cmo_msi_status, cmo_msi_score, msi_status_mmr, label_source, label_certainty`.
- `results/data/clinical_table.csv` — `PATIENT, isMSIH, cmo_msi_status, cmo_msi_score, msi_method,
  msi_status_mmr, tissue_processing_site, slide_staining_site, slide_imaging_site`.
- Label provenance is pre-flagged: `label_source` = {prospective_cmo 539, retrospective_mmr 269};
  `label_certainty` = {definite 642, indeterminate_as_mss 166}.
- Paired-scan structure: `processing_site` = {OAUTHC 481, retrospective_oau 172, retrospective_msk
  97, LUTH 31, LASUTH 15, UITH 12}; ~83 retrospective patients have BOTH msk and oau processing
  of the same tissue.
- QC masks: `results/data/tumor_tiles/<slide>.npy` (TUM tile indices), artifact fractions.
- OOD distance: `results/analysis/ood/` (T2 Mahalanobis / kNN, already computed).
- Wagner model: `argo_deepmsi/models/wagner.py` — slide transformer over CTransPath tiles with a
  CLS token, using `F.scaled_dot_product_attention` (fused; attention weights not currently
  returned).

## Core object: the error ledger

Patient-resolution table (one row per patient, matching the champion's max/√n aggregation) with a
linked per-slide detail table.

Because Wagner is zero-shot, "error" is defined two ways:

1. **Rank-residual** — the signed distance between the patient's Wagner score and its label under
   the score ranking (calibration-free).
2. **Operating-point error class** — at the deployed sens-0.95 threshold: TP / TN / **FN** / FP,
   with a **confident-wrong** flag (score far on the wrong side of the threshold — a confident
   miss or false alarm, distinguished from a boundary wobble). Confident-wrong is the informative
   class.

Patient row fields: `patient_id, y, wagner_patient_score (max/√n of p_msih), n_slides,
rank_residual, error_class, confident_wrong, attributed_cause, circumvention_lever`, plus
covariates: `label_source, label_certainty, cmo_msi_score, msi_method, mmr_cmo_concordance, site,
processing_site, stain_location, imaging_site, n_tiles, tumor_fraction, artifact_fraction,
ood_distance`.

Slide detail row: `slide_id, patient_id, p_msih, y, processing_site, stain_location, n_tiles,
tumor_fraction, artifact_fraction`.

## Stage A — covariate attribution (CPU, no GPU)

Build the ledger from existing artifacts and apply deterministic, transparent attribution rules.
Each bucket names a circumvention lever:

| Bucket | Rule (confident-wrong unless noted) | Circumvention lever |
|---|---|---|
| **label-suspect** | `label_certainty=indeterminate_as_mss` ∨ CMO/MMR disagreement ∨ `cmo_msi_score` within a band of the assay cutoff | molecular/pathology re-adjudication worklist; label-certainty sensitivity re-run |
| **low-tumor-content** | `tumor_fraction` below floor | re-tile / better tumor detection; QC-refix worklist |
| **high-artifact** | `artifact_fraction` above ceiling | artifact removal / re-scan request; QC-refix worklist |
| **borderline-score** | \|score − threshold\| small | inherent difficulty — informs abstention / operating-point choice |
| **site-structured** | residual within-site error elevation after conditioning on the above | routes to Stage B (acquisition vs biology) |
| **unexplained** | none of the above | routes to Stage C (spatial introspection) |

Thresholds (tumor floor, artifact ceiling, `cmo_msi_score` band, confident-wrong margin) are
pre-specified constants at the top of the module, chosen from the existing D2/D3/Q3 ablation
ranges, and recorded in the output manifest. Buckets are assigned in priority order; a row may
carry a primary + secondary cause.

**Outputs:** `results/analysis/error_anatomy/A_error_ledger.csv`,
`A_slide_detail.csv`, `A_pareto.{csv,png}` (overall + OAUTHC-first),
`A_per_site_decomposition.csv`, `worklist_label_adjudication.csv`, `worklist_qc_refix.csv`,
`docs/experiments/E1-wagner-error-anatomy-A.md`.

## Stage B — paired-scan natural experiment (CPU)

For the ~83 retrospective patients with both MSK and OAU processing, compare Wagner's per-slide
`p_msih` across the two acquisitions of the same tissue.

- **Metrics:** within-patient Δp_msih, call-flip rate at the operating threshold, and whether
  flips concentrate by `stain_location` / imaging site vs processing site.
- **Interpretation:** large Δ on identical biology ⇒ acquisition shortcut (supports OAUTHC =
  acquisition shift, a circumventable nuisance); small Δ but still wrong ⇒ biology/label (a
  harder floor).
- **Effect:** partitions A's `site-structured` and `unexplained` buckets into acquisition-driven
  vs not, and localizes the shift to a specific acquisition sub-step (processing/staining/imaging).
- **Circumvention lever:** if acquisition-driven, target the implicated sub-step
  (stain protocol / scanner) or an acquisition-invariant encoder (e.g. the Waiv encoders now
  extracting) rather than more heads.

**Outputs:** `B_paired_scan.csv`, `B_flip_analysis.png`, verdict appended to
`docs/experiments/E2-wagner-error-anatomy-B.md`.

## Stage C — Wagner spatial introspection (GPU, error subset only)

For the confident-wrong + still-unexplained patients from A/B, export Wagner's CLS→tile attention
and re-infer on just those ~tens of slides.

- **Mechanism:** an additive forward hook / context manager that replaces the fused
  `F.scaled_dot_product_attention` with an explicit `softmax(QKᵀ/√d)` returning the weights, used
  only in introspection mode. The frozen forward and its numerics are unchanged in normal mode
  (guarded by a frozen-equivalence test).
- **Metric:** fraction of Wagner's attention mass on tumor vs artifact/non-tumor tiles (using the
  existing `tumor_tiles/<slide>.npy` masks), for errors vs matched correct controls.
- **Interpretation:** attention off-tumor ⇒ fixable tiling/QC; attention on-tumor but still wrong
  ⇒ genuine representational/biology limit.
- **Circumvention lever:** off-tumor ⇒ tumor-restricted tiling at inference; on-tumor ⇒ this is
  the true hard residual, and the honest recommendation is selective abstention (T-phase) or new
  signal, not another head.

**Outputs:** `C_attention_summary.csv`, per-slide overlay PNGs under
`results/analysis/error_anatomy/overlays/`, `scripts/error_anatomy_c.sh` (SLURM GPU wrapper),
`docs/experiments/E3-wagner-error-anatomy-C.md`.

## Synthesis — the recipe

A single figure + table: *"Wagner's N errors decompose into X% label, Y% acquisition, Z% QC/tiling,
W% hard-biology,"* each bucket annotated with its circumvention lever and rough addressable-error
count. Written to `docs/experiments/E4-error-anatomy-synthesis.md` and folded into `docs/summary.md`
as the successor to "current path forward".

## Components

- `argo_deepmsi/eval/error_anatomy.py` — ledger builder, attribution rules, paired-scan analysis,
  synthesis. Pure functions over dataframes; no I/O in the compute core; thresholds as module
  constants.
- Attention-export hook in / alongside `argo_deepmsi/models/wagner.py` — additive, off by default.
- `scripts/error_anatomy_c.sh` — SLURM GPU wrapper (nvidia-A6000-20, `--gres=gpu:1`) for the
  Stage-C subset re-inference.
- CLI/entrypoint: `python -m argo_deepmsi.eval.error_anatomy` runs A + B; Stage C via the script.

## Testing

- Unit test per attribution rule: synthetic ledger rows → expected bucket (incl. priority order
  and primary/secondary assignment).
- Paired-scan join test: yields exactly the expected retrospective patients, zero cross-patient
  leakage, correct Δp sign convention.
- Attention-mass test: synthetic attention vector + tumor mask → known mass fraction.
- Frozen-equivalence test: Wagner forward with introspection mode OFF reproduces the current
  `p_msih` to tight tolerance (the hook must not perturb numerics).
- Determinism: A + B reproduce byte-stable ledgers from cached inputs (no RNG).

## Risks / open questions

- **Confident-wrong margin and thresholds are judgment calls.** Mitigation: pre-specify, record in
  manifest, and report the Pareto's sensitivity to ±1 reasonable step.
- **Paired scans are same-patient, not guaranteed same physical section.** We state this
  explicitly; Δp still bounds acquisition sensitivity even if it slightly conflates section
  sampling.
- **Attention ≠ causal attribution.** Stage C attention-mass is descriptive evidence, not a causal
  claim; report it as such.
- **Wagner SDPA hook** must handle the multi-head, multi-layer structure; the frozen-equivalence
  test is the guardrail.
