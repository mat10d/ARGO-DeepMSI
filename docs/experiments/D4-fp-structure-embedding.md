# D4 — The failure is structured false-positives, not noise (embedding diagnostic)

## Setup
Champion (wagner_zeroshot) at the rule-out operating point: 55 false positives vs 6 false
negatives on 180 patients. Sensitivity is fine; SPECIFICITY is the wall (spec@sens95 ~0.16,
MSIntuit 0.46). Question (from PI): the FPs look like noise in slide-QC metadata — but are they
structured in the EMBEDDING, and does the clinical MSI score explain them?

## Finding 1 — FPs are NOT QC-visible but ARE embedding-structured, across 4 independent FMs
QC metadata cannot tell FP from correct calls (artifact_fraction 0.78 vs 0.77; tumor_fraction 0.20
vs 0.14). But in FM embedding space, a linear probe separates FP from true-negative (within MSS,
5-fold CV AUROC):
| FM | FP-vs-TN | true-MSI-vs-false-alarm (non-circular) |
|---|---|---|
| virchow2_mean | 0.796 | 0.684 |
| conch_v1.5_mean | 0.786 | 0.644 |
| uni2_mean | 0.784 | **0.744** |
| conch_v1.5_titan | 0.765 | 0.615 |
| ctranspath_mean_harmony | 0.759 | 0.713 |
| conch_mean_harmony | 0.782 | 0.709 |
| virchow2_prism | 0.559 | 0.650 |
Replicates across CONCH/UNI2/Virchow2/CtransPath => a GENERAL property of the H&E representation,
not one model's artifact. (PRISM slide-aggregator compresses it away.)

## Finding 2 — CORRECTED: the FP-axis is NOT distinct from the MSI-axis out-of-sample
Initial (LEAKY) analysis fit the FP-axis on all MSS then evaluated on the overlapping predicted-
positive set, giving true-MSI-vs-false-alarm AUROC 0.615 (conch_titan) - 0.744 (uni2). This was
train/test overlap on the FP side. Recomputed with OUT-OF-FOLD FP prediction (5-fold on MSS) and
apply-to-disjoint-MSI-H, the separation COLLAPSES TO CHANCE across all FMs:
| FM | true-MSI-vs-false-alarm (out-of-fold, corrected) |
|---|---|
| conch_v1.5_mean_harmony | 0.600 |
| ctranspath_mean_harmony | 0.575 |
| conch_v1.5_mean | 0.562 |
| uni2_mean | 0.543 |
| conch_v1.5_titan | 0.495 |
| virchow2_prism | 0.368 |
=> The FP-axis carries NO orthogonal information separating true MSI-H from false alarms. It is
largely the champion-score direction re-expressed (recovers "which MSS scored high", nothing more).
A second-stage rejector trained on it would suppress true MSI-H at the same rate as false alarms
=> kills sensitivity. THE NAIVE TWO-STAGE CASCADE PREMISE FAILS.
NB: the FP-vs-TN separability (Finding 1, 0.70-0.80) WAS properly cross-validated and stands — the
FPs are structured, but that structure is not a usable specificity lever.

## Finding 3 — RULED OUT: borderline biology
Are FPs sub-threshold-instability patients? No. Within MSS: FP cmo_msi_score median 2.43 vs TN 2.29
(Mann-Whitney p=0.13, n.s.). True MSI-H median 29.4 (min 10). FPs sit with the clean negatives at
~2.4. The FPs are true false alarms on genuinely stable tumors, not detected borderline instability.
(Champion continuous score does weakly track cmo_msi_score: Spearman rho=0.29, p=0.004.)

## Finding 4 — RULED OUT: Indeterminate-label contamination
cmo_msi_status: Stable 44, Indeterminate 31, Instable 22 (97 patients have a score; y for the other
83 comes from MMR-IHC). Are FPs enriched for Indeterminate? No: FP 44% indet vs TN 40% (Fisher
OR=1.19, p=0.45). Dropping all Indeterminate (definite-only, n=66): champion AUROC 0.679, spec@sens95
0.159 — NOT recovered. The specificity wall is real, not a label artifact.

## Consequence
The specificity wall is NOT fixable by any lever tested: top-heads (S5/attn/fusion), prefiltering
(D2, hurts), batch correction (A1/A2, falsified), per-site adaptation (uncalibratable, 2-4 pos/site),
OR a second-stage embedding rejector (Finding 2 corrected, collapses out-of-fold). It is also NOT an
artifact of borderline biology (Finding 3) or label noise (Finding 4). This exhaustive, rigorously-
falsified map of what does NOT work — plus the honest characterization that FPs are structured yet
the structure is champion-collinear — is the core negative-result contribution. Any remaining upside
must come from a fundamentally different representation or training objective (end-to-end task-trained
tiles, or operating-point / distributionally-robust training), evaluated leave-one-site-out — with a
power analysis for the N such a method would actually need.