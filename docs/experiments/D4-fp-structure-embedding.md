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

## Finding 2 — the FP-axis is partially DISTINCT from the MSI-axis (a specificity fix exists)
FP-axis (learned FP-vs-TN on MSS) score by group (conch_titan): true-neg -1.09, false-pos +0.98,
TRUE MSI-H +0.13. MSI-H sit between TN and FP, not on top of FP. Within champion-positive patients,
FP-axis separates true MSI-H from false alarms at AUROC 0.615 (conch_titan) up to 0.744 (uni2).
=> false alarms are partly separable from real MSI-H => a second-stage rejector can gain specificity.
NON-CIRCULAR: this number is measured within already-positive patients (champion score ~matched),
so it is not just re-recovering "who scored high."

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
The one non-dead-end: a TWO-STAGE CASCADE — high-sensitivity stage1 + a learned UNI2-embedding
false-alarm rejector (stage2) optimizing spec@sens>=0.95/0.96. Must be validated LEAVE-ONE-SITE-OUT
(the specificity gain has to generalize to an unseen site — the clinical claim). If the LOSO gain
evaporates, the honest paper is the exhaustive dead-end map + this FP-structure characterization.
Task: B1-cascade-rejector.
