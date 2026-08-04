# ARGO-DeepMSI — Transition and future research roadmap

**Date:** 2026-08-03
**Horizon:** final experiment week, transition to a larger-GPU system, and subsequent research
**Tactical companion:** [`W1 — CTransPath adaptation week`](experiments/W1-ctranspath-adaptation-and-transition.md)

## North-star goal

Develop and validate a clinically useful, reproducible MSI screening model for colorectal
histopathology that transfers across Nigerian sites, scanners, staining workflows, and acquisition
periods. The model should operate at high sensitivity, provide useful specificity and NPV, expose
uncertainty, and fail in ways that can be audited rather than hidden by a pooled AUROC.

This is not merely a leaderboard objective. The program must determine which apparent MSI signal
is biological, which is site or acquisition shortcut, and what amount and type of data are needed
to support increasingly trainable models.

## Current position

- The primary development cohort contains 217 patients, 803 feature-complete slides, and 47
  MSI-H patients.
- Frozen Wagner/CTransPath with max/sqrt(n_slides) aggregation is the current reference:
  patient AUROC 0.717 (95% CI 0.631–0.799).
- Specificity at 95% sensitivity is 0.141, substantially below the intended screening utility.
- Performance is heterogeneous by site; OAUTHC is the principal weakness.
- A properly nested linear probe over four frozen foundation-model means is near chance
  (AUROC 0.530), so mean pooling is not an adequate comparison of representation quality.
- The hard-QC subset does not materially improve the reference result. The limitation is not
  explained by the historical slide-exclusion rule alone.
- The live REDCap audit found 17 newly labelled nonbenchmark patients, but none currently has a
  corresponding slide in the image corpus. Labels for all 217 benchmark patients are unchanged.
- The completed W1 ladder found no promotable supervised adaptation: the strongest learned
  cached-feature model reached AUROC 0.661, BitFit/norm reached 0.634, and final-stage CTransPath
  unfreezing reached 0.625. Frozen full-bag Wagner/CTransPath remains locked for transition.
- The current cohort has already supported extensive experimentation and must be treated as
  development data, not as a pristine final test set.

## Program hypotheses

The future work is organized around six linked hypotheses:

1. **Aggregation matters:** sparse MSI morphology may be lost by slide-level mean pooling.
2. **Task-specific adaptation matters:** a modest learned head or parameter-efficient encoder
   update may outperform frozen general representations without requiring de-novo training.
3. **Modern encoders contain additional signal:** UNI2, Virchow2, CONCH, and robustness-focused
   models may provide richer or complementary features than CTransPath when tested with fair
   tile-level aggregation.
4. **Multiple teachers are more useful than one replacement encoder:** different foundation
   models may capture complementary morphology, semantics, and invariances that can be distilled
   into a smaller student.
5. **Local unlabeled slides are valuable:** Nigerian H&E can support self-supervised and
   consistency-based domain adaptation even when MSI labels are unavailable.
6. **More trainable parameters require more independent patients:** full end-to-end learning
   becomes defensible only after data growth and genuinely external or temporal validation are
   secured.

## Data roles and governance

Every patient must have exactly one role at a given experimental freeze:

| Role | Purpose | May influence model selection? |
|---|---|---|
| Current 217-patient development cohort | training, inner validation, method development | yes |
| Newly accrued eligible patients | sealed temporal evaluation | no, until first locked evaluation is released |
| Additional unlabeled Nigerian slides | self-supervision and teacher-consensus learning | yes, without using unavailable labels |
| Future external/prospective cohort | final generalization and clinical-operating-point validation | no |

Required data controls:

- Replace sequential `P_####` identifiers with stable, privacy-preserving REDCap-derived keys in
  the transition package.
- Version patient, slide, label, assay, site, acquisition, and QC manifests together.
- Preserve raw assay state, label source, and label certainty rather than only a binary target.
- Ensure paired retrospective acquisitions remain assigned to one patient group.
- Checksum every WSI, feature table, checkpoint, fold file, and evaluation cohort.
- Audit REDCap and image-store freshness before each cohort freeze.
- Define temporal eligibility before inspecting model scores or outcomes.

## Phase 0 — Final experiment week

The staged CTransPath/Wagner adaptation program in the tactical W1 plan is complete:

- cached-feature heads and adapters did not beat the frozen reference;
- BitFit/normalization and bounded final-stage unfreezing were feasible but regressed;
- LoRA was blocked by the distributed TorchScript module and is deferred to eager rehydration;
- patient-grouped evaluation and paired uncertainty were preserved; and
- frozen full-bag Wagner/CTransPath was selected as the sole locked transition model.

This phase shows that additional supervised plasticity is not the near-term path on the current
labels. It also leaves the aggregation, sampling, loss, and training infrastructure needed for
fair future encoder and teacher comparisons.

## Phase 1 — Transition and exact reproduction

The first task on the new machine is reproduction, not another search.

Required transition artifact:

```text
versioned cohort + fold manifests
             +
container or locked environments
             +
checkpoint and feature hashes
             +
ingest -> extract -> train -> evaluate -> infer commands
             +
small non-identifying smoke fixture
             +
expected metrics and tolerances
```

Acceptance criteria:

- The frozen Wagner reference reproduces within a declared numerical tolerance.
- The selected W1 model reproduces its predictions and patient metrics.
- A fresh feature extraction matches stored features on the smoke fixture.
- Fold membership, label mapping, patient aggregation, and thresholds are identical.
- GPU count changes throughput but not cohort membership or the statistical estimand.
- Model sources, licenses, gated-access requirements, and dependency constraints are documented.

Training and extraction environments may be separated when foundation models require conflicting
library versions. Feature stores are the stable interface between those environments.

## Phase 2 — Data and label foundation

Before increasing model complexity, improve the information entering the problem.

Priorities:

1. Acquire and ingest slides for newly labelled REDCap patients without releasing them into
   model development before the first temporal evaluation.
2. Reconcile prospective `Indeterminate` cases and preserve assay-specific uncertainty.
3. Audit OAUTHC sample preparation, fixation, staining, scanning, and label pathway.
4. Obtain targeted pathologist review for tumor presence and diagnostically relevant morphology
   in high-impact discordant cases.
5. Record tissue block, section, stain location, scanner, magnification, and acquisition date in
   a machine-readable schema.
6. Expand the unlabeled Nigerian archive even when confirmed MSI outcomes are not yet available.

Potential label formulation should include more than a single binary endpoint:

- confirmed MSI-H versus confirmed MSS;
- assay type and quantitative assay score where available;
- MMR/IHC state;
- indeterminate or discordant state;
- label confidence; and
- missingness indicators that prevent silent relabeling.

A multi-task, assay-aware model may learn more effectively than a binary model while exposing
discordant or uncertain cases rather than forcing them into MSS.

## Phase 3 — Fair modern-encoder evaluation

Evaluate modern pathology foundation models using the same tile-level learning protocol. Priority
encoders include UNI2, Virchow2, CONCH, and accessible robustness-focused models such as Phaet or
Mascaret. Gated models enter only after access and environment requirements are resolved.

The comparison must hold constant:

- physical tissue coordinates and magnification policy;
- tissue/tumor sampling rules;
- number of tiles seen per patient;
- aggregation capacity;
- loss and patient balancing;
- folds, endpoints, and uncertainty estimation; and
- compute reporting.

For each encoder, evaluate:

1. frozen mean-pool control;
2. frozen tile-level winning aggregator from W1;
3. a small residual feature adapter; and
4. one parameter-efficient encoder adaptation if compute and licensing allow.

An encoder is not rejected solely because its mean-pooled linear probe is weak. Conversely, newer
or larger pretraining is not assumed to imply better MSI transfer.

## Phase 4 — Multi-teacher learning and distillation

### Teacher construction

Train small task-specific heads over the strongest diverse encoders. Teachers must generate
patient-safe out-of-fold outputs. Measure diversity using error overlap, rank correlation,
site-specific complementarity, and high-sensitivity behavior rather than selecting teachers only
by individual AUROC.

The teacher ensemble may combine:

- visual self-supervised models such as UNI2 and Virchow2;
- vision-language models such as CONCH;
- the best CTransPath/Wagner model;
- robustness-focused models if access is granted; and
- explicit morphology or tissue-composition features when they add orthogonal information.

### Student objectives

Candidate students are the best compact W1 model and a smaller modern encoder. Train with a
weighted combination of:

- supervised MSI loss;
- temperature-scaled teacher-logit distillation;
- slide-representation alignment after learned projection;
- teacher spatial-attention or tile-ranking alignment;
- teacher-disagreement-aware weighting; and
- paired-acquisition consistency.

Do not require raw feature equality between different encoders. Their native dimensions,
architectures, fields of view, and embedding geometries differ. Distil task-relevant behavior and
aligned representations instead.

### Unlabeled-data extension

Run teachers over additional Nigerian slides and train the student from consensus pseudo-targets
with confidence filtering. Use self-supervised local/global consistency and paired acquisitions
to learn stain and scanner invariance. Patient-level independence must still be maintained when a
patient later receives a confirmed label.

## Phase 5 — Domain-specific representation learning

If modern frozen encoders plus PEFT and distillation plateau, adapt the representation to the
target domain using Nigerian tissue without immediately attempting de-novo supervised training.

Preferred order:

1. self-supervised continued pretraining on Nigerian H&E;
2. multi-site stain and scanner consistency objectives;
3. paired-acquisition contrastive learning;
4. teacher-guided masked image modeling or local/global alignment;
5. LoRA or adapter tuning during downstream MSI learning; and
6. partial unfreezing of later encoder stages.

The representation should be explicitly audited for site predictability before and after domain
adaptation. Removing all site information is not necessarily desirable—real biological and case
mix differences can correlate with site—but improved MSI performance must not be explained only
by stronger site separation.

## Phase 6 — Full end-to-end or de-novo training

Full encoder fine-tuning is a later-stage option, not the immediate default. De-novo foundation
model training is not justified by the current labelled cohort and would require substantially
more images, compute, engineering, and independent validation.

Advance to deeper end-to-end training only when all of the following hold:

- the number and diversity of independent patients have materially increased;
- enough MSI-H patients exist across multiple sites to estimate site-specific behavior;
- a separate temporal or external test cohort is secured;
- frozen, adapter, PEFT, and distillation baselines are established;
- training can be repeated across seeds and sites within the compute budget; and
- data provenance and licensing permit the intended deployment and publication.

Even then, the preferred starting point is continued self-supervision or partial fine-tuning of a
modern pathology encoder, not random initialization.

## Phase 7 — Prospective clinical validation

The final objective is a locked screening system evaluated on newly accrued patients under a
prespecified protocol.

Primary clinical endpoints:

- sensitivity with confidence interval;
- specificity at the target sensitivity;
- NPV at observed and clinically relevant prevalence;
- assay failure and model coverage;
- calibration;
- performance by site, acquisition workflow, stain/scanner, sex, age, and other available
  clinically relevant groups; and
- discordant-case review against molecular and IHC evidence.

Operational evaluation should also measure inference time, GPU/CPU requirements, failure modes,
quality-control interventions, repeatability across scanned sections, and whether an abstention
path safely handles out-of-distribution cases.

## Evaluation contract throughout the program

Every reported learned model should satisfy:

- patient-grouped splits;
- no preprocessing, feature selection, calibration, or model selection on the evaluation fold;
- nested selection when an internal CV estimate is presented as confirmatory;
- patient-stratified bootstrap intervals and paired deltas against the locked reference;
- complete patient coverage or an explicit selective-prediction analysis;
- separate primary, QC-sensitivity, and label-certainty estimands;
- per-site reporting with sample sizes and uncertainty;
- fixed high-sensitivity operating-point evaluation; and
- explicit designation as developmental, temporal, external, or prospective evidence.

No single internal-CV winner should silently replace the frozen reference. Promotion requires a
locked evaluation on data that did not influence its selection.

## Decision gates

| Gate | Question | Advance when |
|---|---|---|
| G1: W1 adaptation | Does bounded CTransPath adaptation add stable signal? | paired improvement without OAUTHC/high-sensitivity regression |
| G2: transition | Is the pipeline reproducible on the new machine? | predictions, manifests, and metrics match tolerance |
| G3: modern encoders | Does a newer encoder beat CTransPath under a fair tile-level learner? | stable patient-level and site-specific benefit |
| G4: teachers | Are encoder errors sufficiently complementary? | ensemble improves more than calibration alone |
| G5: distillation | Can a compact student retain teacher benefit? | small clinically acceptable loss with lower compute and stable sites |
| G6: domain adaptation | Does local self-supervision improve transfer rather than site memorization? | temporal/external gain and reduced shortcut evidence |
| G7: end-to-end scale | Is there enough data and independent validation? | data and validation prerequisites are met before deeper training |

## Packaging and final slide-set deliverables

The end-of-week handoff should include:

- frozen cohort, label, fold, and REDCap-freshness manifests;
- current and QC/label-sensitivity leaderboards;
- complete W1 candidate table including negative results;
- locked baseline and selected-model predictions and weights;
- environment/container definitions and checkpoint hashes;
- a minimal clean-machine smoke dataset;
- one-command or clearly sequenced ingest, extract, train, evaluate, and infer entrypoints;
- model cards covering provenance, license, intended use, and limitations;
- a transition runbook for the larger-GPU system; and
- final presentation figures generated from versioned result files.

The final slide set should distinguish:

1. what was learned reliably from the current cohort;
2. what remains developmental;
3. why common approaches failed;
4. what the last CTransPath adaptation week tested;
5. how the transition package preserves reproducibility; and
6. why modern multi-teacher distillation and local domain adaptation are the next logical steps.

## Intended long-term trajectory

```text
frozen external Wagner/CTransPath reference
                    -> bounded CTransPath head and PEFT adaptation
                    -> reproducible transition to larger compute
                    -> fair tile-level comparison of modern encoders
                    -> cross-fitted multi-teacher ensemble
                    -> compact teacher-distilled student
                    -> Nigerian self-supervised/domain adaptation
                    -> deeper end-to-end learning as patient count grows
                    -> sealed temporal and prospective clinical validation
```

The central discipline is to scale model flexibility only when data volume and validation
independence scale with it.
