# W1 — CTransPath adaptation week and post-transition distillation

**Date:** 2026-08-03
**Decision window:** one final week of experiments before packaging and transition
**Development cohort:** 217 patients, 803 feature-complete slides, 47 MSI-H patients
**Reference:** frozen Wagner/CTransPath, patient AUROC 0.717 (95% CI 0.631–0.799)

This document is the tactical plan for the final experimental week. The longer-term scientific,
data, engineering, and validation program is defined in [`docs/future-roadmap.md`](../future-roadmap.md).

## Implementation status — 2026-08-03

The W1 runtime has been implemented outside the archived code tree:

- Canonical Wagner model: `argo_deepmsi.models.wagner`
- Trainable current CTransPath adapter: `argo_deepmsi.models.ctranspath`
- Cached-feature heads: `argo_deepmsi.models.w1_heads`
- Immutable data/fold contract: `argo_deepmsi.w1`
- Cached outer-fold runner: `argo_deepmsi.w1_training`
- Raw-tile PEFT path: `argo_deepmsi.w1_peft` and `argo_deepmsi.w1_peft_training`
- Results root: `results/experiments/w1_ctranspath`

The historical `old/HistoBistro` checkout is not imported or read at runtime. The published
checkpoint is staged under `artifacts/checkpoints/wagner`, verified by SHA-256, and loaded by the
project-local implementation. A full-slide probability reproduced its pre-refactor cached value
to within `5.96e-08`.

The patient-fold contract is frozen at 217 patients, 803 slides, and 47 MSI-H patients. Every
candidate uses the same five outer folds and cohort digest
`1b4c517dc9b2bfa31fa4e59808d04cea73fd3f6574dd072931d82f3b936dbdbd`.

### Completed W1 results

| ID | Adaptation | Patient AUROC | Paired delta vs full W1-0 | Spec@sens95 | OAUTHC AUROC |
|---|---|---:|---:|---:|---:|
| W1-0 | frozen Wagner, all tiles | 0.717 | reference | 0.141 | 0.616 |
| W1-0c | frozen Wagner, 1,024-tile cap | 0.693 | -0.024 | 0.112 | 0.573 |
| W1-1 | final normalization/head | 0.606 | -0.111 | 0.082 | 0.528 |
| W1-2 | final Wagner block/head | 0.603 | -0.114 | 0.118 | 0.496 |
| W1-3 | full Wagner aggregator | 0.653 | -0.064 | 0.176 | 0.601 |
| W1-4 | gated-attention MIL | 0.528 | -0.188 | 0.088 | 0.607 |
| W1-5 | query/cross-attention pooler | 0.536 | -0.181 | 0.129 | 0.605 |
| W1-6 | residual adapter + final Wagner block | 0.661 | -0.055 | 0.135 | 0.619 |
| W1-7 | CTransPath BitFit/norm + full Wagner | 0.634 | -0.083 | 0.147 | 0.509 |
| W1-9 | CTransPath final stage + full Wagner | 0.625 | -0.091 | 0.112 | 0.517 |

The cap itself accounts for about 0.024 AUROC, but supervised adaptation accounts for a larger
additional drop. W1-6 is the strongest learned cached-feature candidate and preserves OAUTHC,
but its paired delta from full frozen Wagner is `-0.055` (95% CI `-0.132` to `+0.015`; bootstrap
probability delta <= 0 is 0.936). No learned cached-feature candidate passes the full promotion
gate.

### True CTransPath adaptation status

Raw tile reconstruction now follows the current WSI `TileSpec`, current LazySlide resize, current
CTransPath transform, and current registered weights. Re-encoded versus cached embeddings have
mean cosine similarity `0.9999994` in the GPU smoke test.

- BitFit/normalization: 64,644 trainable encoder parameters; backward and optimizer step pass.
- Final-stage unfreeze: 14,185,392 trainable encoder parameters; backward and optimizer step pass.
- Both use eight differentiably re-encoded raw tiles inside a larger aligned cached-feature bag.
- The fixed raw cache contains 6,424 tiles across all 803 slides. Direct reconstruction and the
  cache were pixel-identical on alignment checks.
- W1-7 finished at patient AUROC `0.634` (95% CI `0.550`–`0.723`), paired delta from W1-0
  `-0.083` (95% CI `-0.162` to `-0.003`), specificity at sensitivity 0.95 `0.147`, and OAUTHC
  AUROC `0.509`.
- W1-9 finished at patient AUROC `0.625` (95% CI `0.541`–`0.715`), paired delta from W1-0
  `-0.091` (95% CI `-0.175` to `-0.005`), specificity at sensitivity 0.95 `0.112`, and OAUTHC
  AUROC `0.517`.

W1-8 LoRA is blocked in the current environment because LazySlide distributes CTransPath as a
`RecursiveScriptModule`, which does not allow submodule replacement or forward hooks. LoRA
requires rehydrating the verified state dict into an eager CTransPath architecture. That is a
transition task; the current week nevertheless tested genuine encoder unfreezing through BitFit
and the complete final stage.

### W1 lock decision

Lock W1-0, frozen full-bag Wagner/CTransPath, as the transition baseline. None of the eight
learned candidates passed the combined overall-AUROC, OAUTHC, and high-sensitivity gate. W1-6 is
the strongest learned cached-feature result and remains useful as a compact transition control,
but it is not a replacement for W1-0. W1-7 and W1-9 show that genuine encoder plasticity is
technically feasible in the current codebase, while their negative paired results close the case
for deeper CTransPath unfreezing on this development cohort. Do not run W1-10 this week.

The complete machine-readable comparison is in
`results/experiments/w1_ctranspath/comparison.csv`; the rendered table is in
`results/experiments/w1_ctranspath/comparison.md`.

## Decision

The final experimental week used a staged, compute-bounded search over improvements to the
CTransPath/Wagner system. It included genuine CTransPath adaptation, not only new classifiers
over frozen embeddings, and stopped after deeper unfreezing failed the advancement gate.

After transition to the larger-GPU machine, retain frozen full-bag CTransPath/Wagner as the
benchmark and the W1-6 adapter design as a student template rather than a promoted model. Then
evaluate newer pathology foundation models as task-specific teachers and distil their
complementary signal into a compact deployable model.

This division is useful because work on sampling, aggregation, losses, validation, and patient
pooling transfers directly to the newer encoders. CTransPath experiments are therefore not
throwaway work even if CTransPath is ultimately replaced.

## Important distinction: two different kinds of unfreezing

The current Wagner pipeline has two learned components:

```text
image tile
   -> CTransPath encoder
   -> cached 768-dimensional tile vector
   -> Wagner projection + two-block slide transformer
   -> slide probability
   -> max/sqrt(n_slides) patient score
```

There are two separate adaptation questions:

1. **Wagner-head adaptation:** unfreeze the projection, slide-transformer blocks, CLS token,
   and classification head. This can use the existing cached CTransPath tile vectors and is
   comparatively inexpensive.
2. **CTransPath encoder adaptation:** backpropagate through some of the CTransPath image
   encoder itself. This requires reading image tiles again; cached vectors are insufficient.
   It is substantially more expensive and has greater overfitting risk.

Both should be explored, but they should not be conflated in reporting.

## Experimental principles

- Split and evaluate at the patient level. No patient may appear in both train and evaluation
  partitions, including patients with multiple slides or paired acquisition variants.
- Use one versioned fold manifest for the entire week.
- Treat the existing 217-patient cohort as development data. It has already supported extensive
  model selection and is not a pristine final test set.
- Prespecify a small number of configurations. With only 47 MSI-H patients, a broad numerical
  hyperparameter grid would primarily optimize validation noise.
- Produce out-of-fold predictions for every candidate and report paired deltas from the frozen
  Wagner reference. Because the winning candidate is selected on these results, its estimate is
  developmental; confirmation must come from the sealed temporal cohort.
- Evaluate patient AUROC, AUPRC, specificity at 95% and 96% sensitivity, NPV, calibration,
  OAUTHC performance, and fold/seed stability. Overall AUROC alone is not sufficient.
- Use patient-balanced sampling and inverse-slide-count weighting so patients with many slides
  do not dominate gradient updates.
- Keep the final slide set separate from model selection. Freeze code, weights, thresholds, and
  preprocessing before scoring newly available patients.

## Search ladder

### Level 0 — Frozen reference

Reproduce the published Wagner checkpoint with the existing CTransPath features and current
full-cohort contract.

- Frozen CTransPath encoder.
- Frozen Wagner projection and transformer.
- Current max/sqrt(n_slides) patient aggregation.
- This remains the no-training reference and no-regression comparator.

### Level 1 — Learn over cached CTransPath features

This level is inexpensive enough to search first and establishes whether the limitation lies in
the externally trained Wagner aggregator rather than the tile representation.

Prespecified candidates:

1. **Wagner head-only:** freeze projection and transformer; refit the final normalization and
   classification layer.
2. **Wagner last-block:** unfreeze the second slide-transformer block and classification head;
   keep the first block and CTransPath frozen.
3. **Wagner full aggregator:** fine-tune the projection, both transformer blocks, CLS token, and
   head while keeping CTransPath frozen.
4. **Gated-attention MIL control:** a small attention pooler over the same CTransPath tiles.
5. **Query/Perceiver pooler:** a small fixed-query pooler whose cost is linear in tile count,
   avoiding full quadratic slide attention.

For all candidates:

- Warm-start Wagner-compatible components from the external checkpoint when possible.
- Use deterministic tile caps plus stochastic tile dropout during training.
- Compare uniform random sampling with tumor-enriched sampling using the existing CTransPath
  tissue head, without making a new label-dependent tile-selection rule.
- Limit each architecture to two regularization settings at most.
- Use early stopping on patient-level inner validation, not slide-level accuracy.

### Level 2 — Feature-space adapters

Before changing CTransPath weights, test whether a small residual adapter can reshape its cached
features for Nigerian tissue:

```text
z_adapted = z + alpha * MLP(LayerNorm(z))
```

This is not true encoder fine-tuning, but it tests a closely related hypothesis at far lower
cost. The adapter should have a narrow bottleneck and strong weight decay. It is trained jointly
with the best Level 1 aggregator.

One optional auxiliary objective is acquisition consistency: paired retrospective scans from the
same patient should have nearby slide representations or predictions. This must operate only
inside the training fold.

### Level 3 — Parameter-efficient CTransPath adaptation

This is the first level that updates the image encoder. Use the best Level 1 aggregation recipe
and reopen a bounded number of image tiles per patient.

Prespecified candidates:

1. **BitFit + normalization:** train bias and normalization parameters in CTransPath while all
   weight matrices remain frozen.
2. **Last-stage LoRA:** add low-rank adapters to attention projections in the final CTransPath
   stage; train LoRA, normalization parameters, and the slide head.
3. **Final-stage unfreeze:** fully unfreeze the last CTransPath stage and head, but keep earlier
   stages frozen.

Use two LoRA ranks at most, with the smaller rank as the default. Do not add a broad learning-rate
grid: choose conservative encoder and head learning rates in advance, with the encoder rate at
least an order of magnitude below the head rate.

Training mechanics:

- Sample a bounded bag, initially 128–256 tiles per slide, and vary the sampled tiles across
  epochs.
- Use mixed precision, gradient accumulation, and gradient clipping.
- Apply stain and geometric augmentation conservatively at the image level.
- Balance patients and classes at the sampler or loss, not by duplicating entire large slide
  bags.
- Save exact sampled tile coordinates for reproducibility audits.
- Run an overfit-one-batch test and a frozen-equivalence test before launching cross-validation.

### Level 4 — Deeper CTransPath unfreezing

Do not schedule full encoder fine-tuning by default. Advance to the last two encoder stages only
if Level 3 shows a consistent benefit over both frozen Wagner and the feature-space adapter.

A reasonable advancement gate is:

- positive paired AUROC delta in most folds and seeds;
- no material degradation at OAUTHC;
- no collapse in specificity at high sensitivity;
- stable training without a widening train/validation gap; and
- improvement large enough to justify the added extraction and deployment cost.

Full CTransPath unfreezing belongs after transition unless this gate is clearly met. At the
current sample size, it is more likely to memorize site, stain, or patient-specific morphology
than to learn a general MSI representation.

## Bounded experiment matrix

| ID | Encoder | Slide learner | Purpose | This week? |
|---|---|---|---|---|
| W1-0 | frozen | frozen Wagner | reference | yes |
| W1-1 | frozen | Wagner head-only | minimal supervised adaptation | yes |
| W1-2 | frozen | Wagner last block + head | task-head plasticity | yes |
| W1-3 | frozen | full Wagner aggregator | strongest warm-started cached-feature model | yes |
| W1-4 | frozen | gated-attention MIL | architecture control | yes |
| W1-5 | frozen | query/Perceiver pooling | scalable aggregation control | yes |
| W1-6 | frozen + residual adapter | best Level 1 head | low-cost representation adaptation | yes |
| W1-7 | BitFit/norm | full warm-started Wagner | cheapest true encoder adaptation | complete; negative |
| W1-8 | final-stage LoRA | best Level 1 head | preferred PEFT candidate | blocked by TorchScript packaging |
| W1-9 | final stage unfrozen | full warm-started Wagner | upper bound for shallow unfreezing | complete; negative |
| W1-10 | last two stages or full encoder | best head | deeper fine-tuning | stopped; gate not met |

This is a comprehensive staged search over meaningful mechanisms, not an exhaustive Cartesian
grid. W1-0 through W1-6 should determine which head and sampling recipe deserves the expensive
image-level experiments.

## Validation and selection

### During the week

Each prespecified candidate receives patient-grouped out-of-fold predictions on the same fold
manifest. Candidate-level estimates are valid descriptions of those candidates, but selecting
the largest observed AUROC makes the selected maximum optimistic. Report the complete candidate
table and label the selected winner as developmental.

For cheap Level 1 candidates, nested inner selection may choose between the two permitted
regularization settings. For expensive Level 3 candidates, fix settings from smoke tests and do
not use final-fold outcomes to tune them.

Minimum reporting:

- number of trainable and total parameters;
- GPU-hours and peak memory;
- tile sampling rule and number of tiles seen per patient;
- patient AUROC with patient-bootstrap interval;
- paired AUROC delta from W1-0 with interval;
- specificity at sensitivity 0.95 and 0.96;
- per-site metrics and OAUTHC delta;
- calibration and score distributions;
- fold-wise and seed-wise results;
- patient coverage and any failed slides.

### After selection

Freeze the following together as one model artifact:

- code commit;
- environment/container and dependency lock;
- CTransPath and Wagner source checkpoint hashes;
- learned adapter/head weights;
- tile extraction magnification, size, normalization, and sampling policy;
- fold and cohort manifests;
- patient aggregation and operating threshold;
- label mapping and REDCap freshness audit;
- exact inference command.

The current REDCap audit found 17 newly labelled nonbenchmark patients but no corresponding
slides in the present corpus. If their slides arrive later, they must not enter this week's model
selection. They can contribute to the sealed temporal evaluation if eligibility and labels are
fixed before inference.

## One-week execution schedule

### Days 1–2: establish the training substrate

- Freeze the patient-fold manifest and candidate configurations.
- Reproduce W1-0 from the packaged entrypoint.
- Implement the common tile-bag dataset, patient-balanced sampler, training loop, and fold-safe
  evaluation writer.
- Run leakage, frozen-equivalence, and one-batch-overfit tests.

### Days 2–3: cached-feature race

- Run W1-1 through W1-6 using cached CTransPath features.
- Select the aggregation and sampling recipe used by the encoder-adaptation lane.
- Reject candidates that improve pooled AUROC by sacrificing OAUTHC or high-sensitivity
  specificity.

### Days 3–5: true CTransPath adaptation

- Verify raw tile reconstruction matches the existing 256-pixel, 0.5-mpp extraction contract.
- Run W1-7 and W1-8.
- Run W1-9 only if memory and wall time permit.
- Launch W1-10 only if the prespecified advancement gate is met.

### Day 6: lock results

- Generate all out-of-fold scores, paired uncertainty, per-site analyses, and failure summaries.
- Select one frozen-feature model and at most one PEFT model for transition.
- Freeze weights and inference settings; do not reopen the current results for further tuning.

### Day 7: package and communicate

- Build a clean-machine smoke test and small non-identifying fixture dataset.
- Verify ingest, extraction, training, evaluation, and inference entrypoints.
- Create the experiment table and figures for the final slide set.
- Record negative results and compute costs as first-class outputs.

If infrastructure consumes more than one day, drop W1-9 and W1-10 before reducing validation
quality or packaging time.

## Post-transition: modern foundation models and teacher–student learning

The larger machine should first reproduce the frozen package. Only after reproduction should it
expand the representation search.

### Phase A — fair teacher comparison

Extract aligned tile features from UNI2, Virchow2, CONCH, and any accessible robustness-focused
models. Give each encoder the same winning aggregation, tile-sampling, loss, and patient-level
evaluation recipe developed in W1. Mean-pooled linear probes are useful controls but should not
be treated as a fair test of an encoder's tile-level MSI signal.

Tile identity should be expressed in slide coordinates and physical scale so teachers with
different native input sizes or magnifications can be aligned to the same tissue regions.

### Phase B — cross-fitted multi-teacher model

Train small task-specific heads for the strongest complementary encoders. Generate teacher
outputs out of fold at the patient level. Combine teachers only when they add complementary
signal, particularly at OAUTHC and the high-sensitivity operating point.

The ensemble teacher may expose:

- calibrated MSI logits;
- slide representations;
- tile or region importance maps; and
- predictive uncertainty or teacher disagreement.

### Phase C — distillation

Distil the teacher ensemble into a compact student. The first candidate is frozen CTransPath with
the W1-6-style residual adapter/aggregator, trained from cross-fitted teacher targets rather than
promoted on its current supervised result. A smaller modern encoder is the alternative if compute
permits at deployment. Encoder unfreezing should be reopened only after independent data growth or
clear teacher-supervised evidence, because W1-7 and W1-9 both regressed on the current labels.

Use a weighted objective containing:

```text
true-label loss
+ temperature-scaled teacher-logit loss
+ slide-representation alignment
+ spatial attention/ranking alignment
+ paired-acquisition consistency
```

Do not force raw CTransPath vectors to equal UNI2 or Virchow2 vectors directly. Their dimensions,
architectures, magnifications, and representation geometries differ. Distil task-relevant
predictions, projected slide representations, and spatial rankings instead.

Teacher predictions used for student training must be generated without exposure to the target
patient. Within every outer fold, teachers and student are trained only from that fold's training
patients. Distillation cannot be allowed to become an indirect label-leakage path.

### Phase D — semi-supervised domain adaptation

Once additional Nigerian WSIs are available, including slides without MSI labels, use the
multi-teacher consensus as a soft target and add self-supervised or consistency learning on local
tissue. Paired acquisitions from the same patient are especially valuable for learning stain and
scanner invariance while preserving biology.

### Phase E — sealed temporal evaluation

Score the locked model on newly accrued eligible patients before using those patients for any
training, teacher fitting, threshold selection, or model choice. After results are released, the
temporal cohort may be incorporated into a future training set only if another prospective or
external test cohort is reserved.

## Success criteria

The week succeeds even if no adapted model beats frozen Wagner, provided it leaves:

1. a leakage-safe and reproducible CTransPath training pipeline;
2. a documented unfreezing curve from frozen head to encoder PEFT;
3. honest paired uncertainty and site-specific failure analysis;
4. one locked model ready for temporal testing; and
5. an encoder-agnostic interface ready for modern teachers and distillation.

The scientific question is not simply whether more trainable parameters increase internal
AUROC. It is whether controlled adaptation extracts MSI morphology that transfers across Nigerian
sites without learning stronger site shortcuts.
