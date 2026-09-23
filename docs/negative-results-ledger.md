# Negative-results ledger

One row per experiment phase ever run on the Nigerian MSI cohort, positive or negative.
Numbers are copied from the original write-ups; nothing here is re-estimated. Removed
write-ups and code live on the frozen branch **`archive/pre-iris-2026-09`**; read them with
`git show archive/pre-iris-2026-09:<path>`.

**Code column.** `iris:` = still on this branch. `archive:` = only on
`archive/pre-iris-2026-09`. `none` = interactive analysis with no committed code.

**Cohort column.** Read every number against its cohort and validation design:

| tag | meaning |
|---|---|
| **217/803 primary** | current estimand: 217 patients / 803 feature-complete slides / 47 MSI-H (C1) |
| 217/803 pre-C1 | April 2026 runs on all slides, before the cohort contract existed |
| 198 eCRF | pathologist-QC exclusion only (509 slides / 198 patients) |
| **181 hard-QC** | historical eCRF + artifact + tumor-floor cohort (428 slides / 181 patients / 41 MSI-H); superseded, now only a sensitivity set |
| non-nested | configuration chosen on the same OOF predictions it reports (optimistic; `confirmatory_valid=false`) |
| nested | selection inside inner patient folds; outer predictions untouched (V1 contract) |

Strategy families follow [AGENTS.md](../AGENTS.md). `analysis` marks QC, diagnostics, and
cohort work that fits no model.

## What we learned

- **Only the pretrained task-specific model carries reliable signal.** Frozen
  Wagner/CTransPath with max/√n patient pooling is AUROC **0.717 (0.631–0.799)** on
  217/803, spec 0.141 at sens 0.95. No method trained on this cohort has beaten it.
- **Nesting erased the cohort-trained leaderboard.** The pre-specified nested four-encoder
  linear probe is **0.530 (0.429–0.636)**. The historical 0.60–0.67 rows (grids, attention
  heads, fusion) chose their configuration on the OOF predictions they reported.
- **Bag-size calibration was the only aggregation gain.** max/√n moved Wagner from 0.659 to
  0.717 at zero parameters. Every learned aggregator lost: slide attention, Deep-Sets, stacked
  fusion, learned LR pooling, and a from-scratch Wagner-style transformer.
- **QC and tumor filtering do not earn their place.** Restoring the hard-QC slides changes
  AUROC by +0.019 (CI −0.006 to +0.047). Tumor floors ≥0.05 hurt monotonically. Wagner-P,
  embedding-outlier, and NCT tissue-classifier filters all failed.
- **Site is encoded everywhere, and it is entangled with MSI.** Raw TITAN separates
  OAUTHC-prospective from retro-OAU at site-prediction AUROC 1.000. ComBat, targeted
  alignment, Macenko stain normalization, FLEX, and fmMAP either left OAUTHC unchanged or
  pushed it to chance. Soft Harmony helped only on the non-nested 181-patient cohort.
- **Stain and acquisition are not the main driver.** Mean-pooled encoders are stain-robust
  within patient (virchow2 r 0.94). Wagner flips only 4.8% of calls between paired MSK/OAU
  scans. Neural slide encoders (PRISM, TITAN) are the least stain-robust.
- **The failure is specificity, not sensitivity.** At sens 0.95, Wagner makes 146 false
  positives vs 2 false negatives. Of the errors, 39% are borderline, 29% are confident but
  unexplained (attention on tumor is normal), 22% have suspect labels, and 11% have low tumor
  content.
- **False positives are structured but not separable.** A probe separates FP from TN
  (0.70–0.80, cross-validated). Out of fold, however, the FP axis cannot separate true MSI-H
  from false alarms (≤0.600). The two-stage rejector cascade was retired.
- **Better encoders help only under low-capacity heads.** Waiv (Phaet + Mascaret) is the
  first frozen encoder above the nested baseline (0.619). A from-scratch transformer on Waiv
  collapses to 0.542. The binding constraint is aggregator pretraining, not tile features.
- **Supervised adaptation of Wagner/CTransPath lost to the frozen model.** The best of eight
  W1 candidates (W1-6, 0.661) is −0.055 vs frozen. Encoder unfreezing (BitFit, final stage)
  was significantly negative.
- **Abstention is safe only where signal exists.** A global rule-out skips ~49% of testing
  at ~9% FOR. Per-site FOR is unequal, OAUTHC cannot be calibrated (FOR 0.198), and OOD
  distance does not predict error.
- **Zero-shot vision-language is a replicated dead end.** vl_text_cosine ~0.56–0.60,
  Tip-Adapter 0.444, and VL tumor forensic 0.518 on true labels. TITAN has no generation head.

## Retractions and corrections

| item | what was claimed | correction | where |
|---|---|---|---|
| D4 correction | true-MSI-vs-false-alarm separability 0.615–0.744 (uni2 best) | in-sample leak on the FP side; out-of-fold it is 0.368–0.600 (chance) | commit `69bbb8a` |
| B1 cascade retire | UNI2 false-alarm rejector as a second stage | premise falsified by the D4 correction; grounding numbers retracted; never built | commit `73d05f6` |
| FM-MSI benchmark | CONCH spec 0.65 @ sens 0.90 and 0.45 @ sens 0.94 on TCGA/PAIP (S2, S3, F2, backlog) | closed-access paper never read; operating points and journal attribution stripped; cite only PII S0895611125001892 as a named comparator | commits `e02bf7e`, `70b257e` |
| D2 "0.732 ceiling" | dropping hard QC lifts the champion to 0.732 | selected-patient result; on 217/803 the paired delta is +0.019 (CI −0.006 to +0.047) | C1 |
| calibrated_pool 0.729 | +1.9 pt over Wagner | MSS null curve built from a dirty MSS pool; rebuilt on clean MSS it ties Wagner (0.710) | historical leaderboard |
| slide_attention_mil 0.682 | strongest cohort-trained head | post-hoc eval filter of dirty OOF; the true clean refit is 0.669 | historical leaderboard |
| select-on-OOF rows | simple_grid 0.670, Harmony/grid ~0.656 | non-confirmatory; nested replacement 0.530 | V1 |

## Cohort, QC, and validation

| ID | Hypothesis | Family | Cohort / validation | Headline | Verdict | Code |
|---|---|---|---|---|---|---|
| apr-C7 | Pathologist eCRF exclusion cleans the cohort | analysis | 803 → 198 eCRF | Wagner clean 0.710 vs dirty 0.717 | small loss; became the historical board cohort | archive: `scripts/archive/c7_qc_exclusion.py` |
| apr-C5 P0 | Low Wagner-P slides are non-tumor and can be dropped | analysis | 217/803 pre-C1 | overall 0.659 → 0.662 at t=0.10; OAUTHC 7+ bin ≤0.312 | negative; strips MSI-H signal (4 of 19 low-P slides MSI-H) | archive: `scripts/archive/c5_phase0.py` |
| apr-C5 P1b | Embedding-outlier (HDBSCAN / MSK-distance) slides are non-tumor | analysis | 217/803 pre-C1 | best 0.661 vs 0.659; 7+ bin 0.23–0.30 | negative; outlier islands are site clusters, not non-tumor | archive: `scripts/archive/c5_phase1b.py` |
| apr-C5 P1c | NCT-CRC 9-class tissue head filters non-tumor slides | analysis | 217/803 pre-C1 | NCT holdout TUM F1 0.933 (nonorm); filter `tumor_frac<0.1` → 0.551 | negative as a filter; head later reused by Q3 at floor 0.01 | iris: `scripts/qc/tumor_tile_classifier.py`, `scripts/qc/tumor_tile_classifier_apply.py` |
| Q1 | eCRF layer defines a clean cohort | analysis | 808 → 509 sl / 198 pt | 281/295 drops are OAUTHC | kept as descriptive column | iris: `argo_deepmsi/eval/cohort.py` |
| Q2 | GrandQC artifact fraction flags unusable slides | analysis | 509 sl | 392/507 flagged at 0.5; OAUTHC 99%, UITH 100%, MSK 20% | flag only; site-skewed, not a valid exclusion | iris: `scripts/qc/artifact_qc.py` |
| Q3 | Tumor floor purifies the cohort | analysis | 509 → 428 sl / 181 pt, non-nested floor choice | floor 0.01: 0.7127, spec@sens95 0.154 → 0.179 | adopted then superseded by D3/C1 | iris: `scripts/qc/tumor_tiles_apply.py`; archive: `scripts/tumor_filter_smokeoff.py` |
| Q4 | Freeze v1 manifest and re-race board | analysis | 181 hard-QC | champion 0.7127; OAUTHC 0.579 | superseded by C1 | iris: `argo_deepmsi/eval/cohort.py`, `argo_deepmsi/eval/qc_comparison.py` |
| D2 | Hard QC exclusion helps | analysis | same 181 pt, non-nested | champion hard 0.713 vs none 0.732; OAUTHC 0.579 vs 0.617; soft weighting no better | drop hard QC (magnitude later corrected by C1) | archive: `scripts/run_qc_ablation.py`, `argo_deepmsi/eval/reliability_weight.py` |
| D3 (floor) | Tumor floor should be raised | analysis | same 181 pt, non-nested | floors ≥0.05 drop champion to 0.61–0.67 | keep floor ≤0.01 or off | archive: `scripts/run_tumor_agg_ablation.py` |
| C1 | Full feature-complete cohort is the primary estimand | pretrained_end_to_end | **217/803 primary**, patient bootstrap | Wagner 0.717 (0.631–0.799), spec@sens95 0.141; 181-pt 0.713; definite-label 0.719 | adopted; hard-QC gain not resolved (+0.019, CI crosses 0) | iris: `argo_deepmsi/eval/cohort.py`, `scripts/audit_redcap_freshness.py` |
| V1 | Nested selection gives an honest cohort-trained estimate | frozen_foundation_trained_head | 217/803, nested 3×5 outer folds | 0.530 (0.429–0.636); OAUTHC 0.510; encoder choice unstable | adopted; prior grids non-confirmatory | iris: `argo_deepmsi/eval/validation.py`, `argo_deepmsi/scorers/nested_linear_probe.py` |

## Aggregation and heads

| ID | Hypothesis | Family | Cohort / validation | Headline | Verdict | Code |
|---|---|---|---|---|---|---|
| apr-B1 | Classifier / feature engineering is the bottleneck | frozen_foundation_trained_head | 217/803 pre-C1, 48 configs, non-nested | best conch_v1.5_mean + LR 0.603 ± 0.170 | negative; LR ties everything | archive: `scripts/archive/autoresearch_tier1.py` |
| apr-C2 | Late fusion / stacking / few-shot improve generalization | frozen_foundation_trained_head | 217/802 pre-C1, patient-CV and LOSO, best-of-bucket | patient-CV late-avg 0.586 (+0.010); LOSO stacking 0.580 vs 0.522; kNN/proto ≤ single | within-cohort flat; few-shot useless | history only (`scripts/fusion.py`, removed in `b1e9338`) |
| apr-C4 | Multi-slide bags break naive pooling | analysis | 217/803 pre-C1 | 7+ slide bin (n=18) 0.250 mean / 0.375 max | diagnosed bag-size inflation | archive: `scripts/archive/multislide_analysis.py` |
| apr-C4b | Bag-size-calibrated max rescues Wagner | pretrained_end_to_end | 217/803 pre-C1, zero params | max/√n 0.659 → 0.717; OAUTHC 0.451 → 0.616; 7+ bin 0.125 | **kept**, now the champion aggregator | iris: `argo_deepmsi/scorers/calibrated_pool.py` |
| apr-C5 P2 | Learned slide attention beats max/√n | frozen_foundation_trained_head | 217 pt, 5-fold, best-of-18, non-nested | best wagner+meta 0.665; 7+ bin 0.156 | negative | archive: `scripts/slide_attention_mil.py` |
| slide_attention_mil | Attention-MIL over patient slide bags | frozen_foundation_trained_head | 198 eCRF refit, non-nested | 0.669 clean / 0.665 full | negative vs 0.710 | archive: `argo_deepmsi/scorers/slide_attention_mil.py` |
| simple_grid | Small heads × 6 embeddings × raw/PCA | frozen_foundation_trained_head | 198 eCRF refit, 36 configs, non-nested | best conch_v1.5_titan LR PCA-100 0.670; 0.646 on 181 | non-confirmatory (V1) | archive: `argo_deepmsi/scorers/simple_grid.py` |
| score_fusion | Late-average conch + virchow2 LR heads | frozen_foundation_trained_head | 198 eCRF refit, non-nested | 0.578 clean / 0.601 full | negative | archive: `argo_deepmsi/scorers/score_fusion.py` |
| S1 | Linear probe on slide-FM embeddings | frozen_foundation_trained_head | 181 hard-QC, 5-fold, non-nested | TITAN 0.646, spec@sens95 0.229; K≤16 ~0.50–0.56 | below champion; not few-shot capable | iris: `argo_deepmsi/scorers/slidefm_linearprobe.py`; archive: `scripts/run_slidefm_linearprobe.py` |
| S2 | ProtoNet on cluster-aggregated CONCH tumor tiles | frozen_foundation_trained_head | 181 hard-QC | 0.521; few-shot flat ~0.51 | negative | archive: `argo_deepmsi/scorers/protonet_cluster.py`, `scripts/build_protonet_features.py` |
| S4 | Gated ABMIL + multi-fidelity fusion on tumor tiles | frozen_foundation_trained_head | 181 hard-QC, inner-val top-k | 0.607; OAUTHC 0.629 vs champion 0.579 | below champion; most site-uniform | iris: `argo_deepmsi/scorers/clam_tilemil.py`, `scripts/run_clam_tilemil.py`, `scripts/build_tilemil_bags.py` |
| S5 | Learned Deep-Sets aggregator beats max/√n | frozen_foundation_trained_head | 181 hard-QC | 0.516; single-slide 0.405, 5+ 0.136 | negative | archive: `argo_deepmsi/scorers/setencoder_agg.py` |
| F1 | Stacked fusion of top-3 scorers | mixed | 181 hard-QC, patient-grouped meta-LR | 0.703 vs 0.713; spec@sens95 0.107 vs 0.179; OAUTHC 0.641 | negative for rule-out | archive: `argo_deepmsi/scorers/fusion_top3.py` |
| D1 | Scorer errors are disjoint enough for a learned gate | analysis | 180 hard-QC, 15 scorers | both-err 0.13–0.18 vs either-err 0.59–0.66; oracle 1.000 is an upper bound only; OAUTHC routing 32/63 | complementarity is patient-level; gate never validated | none (outputs on archive: `results/analysis/error_structure/`) |
| D3 (agg) | One aggregator fits all scorers | mixed | same 181 pt, non-nested | champion max/√n best (0.732 off); Harmony mean 0.713 OAUTHC; learned_lr always ≤ best simple | aggregation is scorer-specific; no learned pooling | archive: `scripts/run_tumor_agg_ablation.py` |

## Encoders and vision-language

| ID | Hypothesis | Family | Cohort / validation | Headline | Verdict | Code |
|---|---|---|---|---|---|---|
| apr-A3 | Western Wagner transfers zero-shot | pretrained_end_to_end | 217/803 pre-C1, no fitting | patient 0.659 (mean), slide 0.572; OAUTHC slide 0.439; LazySlide vs HistoBistro features 0.7178 vs 0.6839 (263 rows) | reference baseline; extraction not the cause | iris: `scripts/wagner_zeroshot.py`, `argo_deepmsi/scorers/wagner_zeroshot.py`, `argo_deepmsi/models/wagner.py` |
| apr-C3 | Embeddings regress continuous `cmo_msi_score` | frozen_foundation_trained_head | 123 pt, patient-CV and LOSO | no config |ρ|>0.2 or AUROC@10>0.55; Wagner pooled r 0.166, OAUTHC −0.045 | null | archive: `scripts/archive/msiscore_regression.py` |
| nuclear_morphology | NuLite cell-class fractions carry MSI signal | frozen_foundation_trained_head | 198 eCRF refit | 0.513 clean / 0.529 full | negative | archive: `argo_deepmsi/scorers/nuclear_morphology.py` |
| vl_text_cosine | Zero-shot TITAN text prompts score tiles | pretrained_end_to_end | 217 full / 198 eCRF | tile_max patient 0.598; clean 0.562; slide_score 0.461 | weak; slide-level text axis wrong | iris: `argo_deepmsi/scorers/vl_text_cosine.py`, `scripts/domain_shift/vl_text_cosine.py` |
| transductive_smoothing | kNN smoothing of VL tile scores helps | pretrained_end_to_end | 198 eCRF | 0.555 clean; smoothing −1 to −3 pts vs raw | negative | archive: `argo_deepmsi/scorers/transductive_smoothing.py` |
| S3 | Tip-Adapter / CoOp-lite on CONCH-TITAN | frozen_foundation_trained_head | 181 hard-QC | Tip 0.444, CoOp-lite 0.502, zero-shot 0.438 | negative (below chance) | archive: `argo_deepmsi/scorers/tip_adapter.py` |
| A4 | TITAN can generate morphology descriptions | analysis | capability gate | no generation head in `MahmoodLab/TITAN` | blocked; resolved null | none |
| E5 | VL tumor axis adjudicates Wagner FPs | pretrained_end_to_end | 195 pt | tumor VL-MSI AUROC on true labels 0.518 | instrument invalid; VL lever retired | archive: `scripts/vl_tumor_forensic.py` |
| B2 (nested) | Waiv acquisition-robust encoders beat base-4 | frozen_foundation_trained_head | **217/803 primary, nested** | waiv 0.619 (0.524–0.712), OAUTHC 0.594; base-4 0.530; all-6 0.573 | first honest frozen-encoder gain; still < Wagner | iris: `argo_deepmsi/models/waiv.py`, `scripts/domain_shift/waiv_race.py` |
| B2 (ABMIL) | Waiv gain survives a supervised head | frozen_foundation_trained_head | `in_clean_set` 487 sl / 195 pt, tumor-only, M=3 | phaet 0.643, mascaret 0.623, concat 0.614; OAUTHC 0.53–0.58 | replicates encoder gain; no Wagner beat | archive: `scripts/waiv_mil.py` |
| B2 (from scratch) | Wagner-style transformer on Waiv tiles | frozen_foundation_trained_head | 217/803, W1 folds | 0.542 (0.449–0.641); OAUTHC 0.445 | negative; aggregator pretraining is the wall | archive: `scripts/waiv_wagner.py` |

## Domain shift and batch

| ID | Hypothesis | Family | Cohort / validation | Headline | Verdict | Code |
|---|---|---|---|---|---|---|
| P1 | Stain protocol shifts FM predictions within patient | frozen_foundation_trained_head | 83 paired retrospective pt, patient-grouped OOF | virchow2_mean κ 0.94 / r 0.94; PRISM r 0.31 (Wilcoxon p 0.035); TITAN r 0.45 | mean-pooled FMs stain-robust; slide encoders not | iris: `scripts/domain_shift/paired_staining.py` |
| P2 | FMs generalize leave-one-site-out | frozen_foundation_trained_head | 217/803 pre-C1, LOSO | best large-site AUROC 0.53–0.57; OAUTHC ≤0.53 | negative; unseen site at chance | iris: `scripts/domain_shift/site_holdout.py` |
| apr-C1 | Specimen size, site covariates, and Harmony explain failure | analysis | 217/803 pre-C1 | tercile AUROC 0.582/0.473/0.615; Harmony LOSO conch_titan 0.450 → 0.549, uni2 −0.061 | size not the driver; Harmony not a generic rescue | archive: `scripts/archive/failure_diagnosis.py`; iris: `scripts/domain_shift/harmony_integrate.py` |
| apr-C4 UMAP | Embeddings cluster by site, not MSI | analysis | 217/803 pre-C1 | visual panels only | site-structured embedding space | iris: `scripts/domain_shift/umap_embeddings.py` |
| R1 | S4 calls are brittle to crop/stain | frozen_foundation_trained_head | 181 hard-QC, feature-space TTA proxy | inter-condition κ 0.927; AUROC 0.584–0.597 | robust but weak; κ is an upper bound | iris: `scripts/domain_shift/run_stainaug_tta.py` |
| R2 | FLEX adversarial bottleneck gives site invariance | frozen_foundation_trained_head | 181 hard-QC | 0.623; OAUTHC 0.514; every per-site Δ negative (mean \|Δ\| 0.209) | negative | iris: `argo_deepmsi/scorers/flex_bottleneck.py` |
| R3 | Site-residualized supervised UMAP (fmMAP) | frozen_foundation_trained_head | 181 hard-QC | 0.550; OAUTHC 0.499; mean \|Δ\| 0.211 | negative | iris: `argo_deepmsi/scorers/fmmap_probe.py` |
| T2 | OOD distance flags champion errors | analysis | 181 hard-QC | kNN OOD 0.837, Mahalanobis 0.998; OOD→error 0.552 / 0.545 | shift pervasive; does not predict error | iris: `argo_deepmsi/eval/ood.py` |
| D0 | OAUTHC deficit is a processing batch effect | analysis | 181 hard-QC | raw TITAN site-pred 1.000, Harmony 0.823; OAUTHC 0.611 → 0.685 | batch present; later shown entangled with MSI | none |
| A0 | Harmony CONCH-TITAN probe recovers OAUTHC | frozen_foundation_trained_head | 181 hard-QC, non-nested | OAUTHC 0.683 (+0.075), overall 0.656 | best OAUTHC point, non-nested | iris: `argo_deepmsi/scorers/harmony_probe.py` |
| A1 | Stronger linear batch correction | frozen_foundation_trained_head | 181 hard-QC | targeted retro-OAU: site-pred 0.079, OAUTHC 0.615; ComBat 0.552 | negative; batch-sep decoupled from MSI | iris: `argo_deepmsi/scorers/batch_corrected_probe.py` |
| A2 | Macenko stain-norm + re-extraction of OAUTHC | frozen_foundation_trained_head | 181 hard-QC | OAUTHC 0.484, overall 0.591 | negative; destroys MSI signal | iris: `scripts/domain_shift/stain_norm_oauthc.py`, `argo_deepmsi/scorers/stainnorm_probe.py` |
| A3 | OAUTHC-specific few-shot ABMIL adaptation | frozen_foundation_trained_head | 181 hard-QC | K=0 0.445; K=all 0.612; full-cohort adapted 0.590 | adaptation needed but caps below Harmony | iris: `argo_deepmsi/scorers/tilemil_oauthc_adapt.py` |
| A6 | OAUTHC-recovery synthesis | mixed | 181 hard-QC, non-nested | Harmony 0.579 → 0.683 on OAUTHC (~47% of gap to retro-OAU 0.80) | partial recovery; not at MSIntuit spec | iris: `scripts/domain_shift/build_oauthc_synthesis.py` |
| D3 (Harmony) | Harmony + mean pooling lifts OAUTHC further | frozen_foundation_trained_head | same 181 pt, non-nested | OAUTHC 0.713 (floor 0.01, mean) | non-nested; B2 nested OAUTHC 0.594 is the honest comparator | archive: `scripts/run_tumor_agg_ablation.py` |
| E2 | Acquisition drives Wagner calls | pretrained_end_to_end | 83 paired retrospective pt | call flips 4.8% (4/83), r 0.811, OAU mean p 0.316 vs MSK 0.411 | acquisition mostly not the driver | iris: `argo_deepmsi/eval/error_anatomy.py` |

## Calibration and abstention

| ID | Hypothesis | Family | Cohort / validation | Headline | Verdict | Code |
|---|---|---|---|---|---|---|
| T1 | Conformal rule-out overlay on the champion | pretrained_end_to_end | 181 hard-QC, split-conformal ×50 | rules out 48.6% at empirical FOR 9.3%; per-site FOR 0.000–0.288 | works globally; unfair per site | iris: `argo_deepmsi/scorers/selective_abstention.py` |
| T3 | Group-conditional conformal makes rule-out fair | pretrained_end_to_end | 181 hard-QC | OAUTHC own-threshold FOR 0.198; AUROC gap 0.421; coverage-adjusted gap 0.399 | fairness gate FAIL | iris: `argo_deepmsi/eval/fairness.py` |
| A5 | Per-site operating point fixes OAUTHC | frozen_foundation_trained_head | 181 hard-QC, 5-fold CV on OAUTHC | global threshold sens 1.000 / NPV 1.000 / spec 0.275; per-site sens 0.923 | per-site recalibration only costs sensitivity | iris: `argo_deepmsi/eval/per_site_calibration.py` |
| F2 | Head-to-head vs MSIntuit | pretrained_end_to_end | 181 hard-QC | spec@sens96 0.079 vs MSIntuit 0.46–0.47; NPV 0.926 | not MSIntuit-competitive | archive: `scripts/build_paper_tables.py` |

## Adaptation (W1)

Frozen 5-fold contract on **217/803 primary**, cohort digest `1b4c517d…`. Every candidate is
developmental (selected on these folds). Paired Δ is vs W1-0.

| ID | Hypothesis | Family | Headline (AUROC / Δ / spec@sens95 / OAUTHC) | Verdict | Code |
|---|---|---|---|---|---|
| W1-0 | Frozen Wagner, all tiles | pretrained_end_to_end | 0.717 / ref / 0.141 / 0.616 | locked baseline | iris: `argo_deepmsi/models/wagner.py` |
| W1-0c | 1,024-tile cap | pretrained_end_to_end | 0.693 / −0.024 / 0.112 / 0.573 | cap costs ~0.024 | archive: `scripts/w1_run_capped_reference.py` |
| W1-1 | Refit final norm + head | adapted_pretrained_head | 0.606 / −0.111 / 0.082 / 0.528 | negative | archive: `argo_deepmsi/w1_training.py`, `argo_deepmsi/models/w1_heads.py` |
| W1-2 | Final Wagner block + head | adapted_pretrained_head | 0.603 / −0.114 / 0.118 / 0.496 | negative | archive: same |
| W1-3 | Full Wagner aggregator | adapted_pretrained_head | 0.653 / −0.064 / 0.176 / 0.601 | negative | archive: same |
| W1-4 | Gated-attention MIL | frozen_foundation_trained_head | 0.528 / −0.188 / 0.088 / 0.607 | negative | archive: same |
| W1-5 | Query / cross-attention pooler | frozen_foundation_trained_head | 0.536 / −0.181 / 0.129 / 0.605 | negative | archive: same |
| W1-6 | Residual adapter + final Wagner block | adapted_pretrained_head | 0.661 / −0.055 (CI −0.132 to +0.015) / 0.135 / 0.619 | best learned; not promoted | archive: same |
| W1-7 | CTransPath BitFit/norm + full Wagner | backbone_finetune | 0.634 / −0.083 (CI −0.162 to −0.003) / 0.147 / 0.509 | negative | archive: `argo_deepmsi/w1_peft.py`, `argo_deepmsi/w1_peft_training.py`; iris: `argo_deepmsi/models/ctranspath.py` |
| W1-8 | Final-stage LoRA | backbone_finetune | not run | blocked (TorchScript CTransPath) | none |
| W1-9 | CTransPath final stage + full Wagner | backbone_finetune | 0.625 / −0.091 (CI −0.175 to −0.005) / 0.112 / 0.517 | negative | archive: as W1-7 |
| W1-10 | Deeper unfreezing | backbone_finetune | not run | advancement gate not met | none |

## Error anatomy

| ID | Hypothesis | Family | Cohort / validation | Headline | Verdict | Code |
|---|---|---|---|---|---|---|
| apr-C1 1c | Metadata predicts Wagner errors | analysis | 217/803 pre-C1, slide level | largest \|coef\|: y −1.24, cut_location_OAUTHC −0.76 | errors concentrate on OAUTHC | archive: `scripts/archive/failure_diagnosis.py` |
| D4 | False positives are embedding-structured and separable | analysis | 180 hard-QC | FP-vs-TN 0.759–0.796 (PRISM 0.559); out-of-fold true-MSI-vs-FP 0.368–0.600; FP cmo score 2.43 vs TN 2.29 (p 0.13) | structured but not a specificity lever (see retractions) | none |
| B1 cascade | Second-stage false-alarm rejector | frozen_foundation_trained_head | — | premise falsified by D4 correction | retired, never built | none |
| E1 | Covariates explain Wagner errors | analysis | 217/803 primary, threshold 0.1375 at sens 0.95 | 146 FP / 2 FN; enrichment: low-tumor 2.80×, label-suspect 1.66×, high-artifact 0.78× (depleted) | ~33% addressable by data/QC work | iris: `argo_deepmsi/eval/error_anatomy.py` |
| E3 | Confident FPs come from spatial mis-attention | analysis | 76 error vs 59 control slides | tumor attention / availability 1.06× vs 1.09× | negative; genuine representational residual | iris: `scripts/qc/error_anatomy_c.sh`, `argo_deepmsi/models/wagner.py` |
| E4 | Synthesis of E1–E3 | analysis | 217/803 primary | borderline 39%, unexplained 29%, label-suspect 22%, low-tumor 11% | act on worklists and abstention; no new heads on frozen CTransPath | iris: `argo_deepmsi/eval/error_anatomy.py` |

Detailed write-ups still on `iris`: `docs/experiments/` (P1, P2, C1, V1, Q1–Q4, D0, A0–A6,
R1–R3, T1–T3, E1–E5, B2, W1). All other IDs above are on the archive branch under
`docs/experiments/`, `docs/archive/`, or `docs/scorers/`.
