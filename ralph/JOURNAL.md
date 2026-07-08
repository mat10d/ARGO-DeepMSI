# ARGO-DeepMSI autorun journal (append-only; one line per iteration)
# format: <UTC> <task-id> <status> <one-line-result-or-reason>
# terminal markers: LOOP-COMPLETE <UTC> | HALT-BLOCKED <UTC> <ids>

SCAFFOLD-INIT autorun-scaffold branch created: ralph/ harness + eval/screening.py + tests. E00 pending.
2026-07-07T21:17:41Z Q1-ecrf-qc done cohort_clean.csv v0: 509 slides / 198 patients clean (prev 0.218); eCRF drops 295 slides, 281 from OAUTHC.
2026-07-08T01:07:16Z Q2-artifact-qc done GrandQC artifact-QC merged: 507/509 clean scored, 392 flagged @floor0.5 (flag-only, site-skewed: OAUTHC 99%/UITH 100% vs MSK 20%; drop deferred to Q3/Q4).
2026-07-08T01:49:30Z Q3-tumor-filter done ctranspath TUM head, floor 0.01: 509->428 slides/181 pat; champion AUROC 0.7127 (>=0.710), spec@sens95 0.154->0.179.
