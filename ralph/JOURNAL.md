# ARGO-DeepMSI autorun journal (append-only; one line per iteration)
# format: <UTC> <task-id> <status> <one-line-result-or-reason>
# terminal markers: LOOP-COMPLETE <UTC> | HALT-BLOCKED <UTC> <ids>

SCAFFOLD-INIT autorun-scaffold branch created: ralph/ harness + eval/screening.py + tests. E00 pending.
2026-07-07T21:17:41Z Q1-ecrf-qc done cohort_clean.csv v0: 509 slides / 198 patients clean (prev 0.218); eCRF drops 295 slides, 281 from OAUTHC.
2026-07-08T01:07:16Z Q2-artifact-qc done GrandQC artifact-QC merged: 507/509 clean scored, 392 flagged @floor0.5 (flag-only, site-skewed: OAUTHC 99%/UITH 100% vs MSK 20%; drop deferred to Q3/Q4).
2026-07-08T01:49:30Z Q3-tumor-filter done ctranspath TUM head, floor 0.01: 509->428 slides/181 pat; champion AUROC 0.7127 (>=0.710), spec@sens95 0.154->0.179.
2026-07-08T02:52:28Z Q4-rebuild-manifest done manifest frozen v1-ecrf+artifact+tumor (428 slides/181 pat, prev 0.236); board re-raced on in_clean_set; no_regression_floor=0.7127 (calibrated_pool); verify green.
2026-07-08T03:19:09Z S1-slidefm-linearprobe done LR probe on TITAN/PRISM slide embeddings; best=TITAN patient AUROC 0.646 (< champ 0.713) but spec@sens95 0.229 > 0.179; few-shot near-chance at K<=16 (not low-N capable); floor holds.
2026-07-08T03:57:35Z S2-protonet-cluster done ProtoNet on cluster-agg CONCH tumor tiles (G=8, 6144-d): patient AUROC 0.521 (~chance, worst on board); few-shot flat ~0.51 all K; negative result, floor holds. FM benchmark recipe transcribed.
2026-07-08T04:25:01Z S3-tip-adapter done training-free Tip-Adapter + CoOp-lite on CONCH/TITAN: tip AUROC 0.444 (below chance, worst on board), coop_lite 0.502, zero_shot 0.438; slide-level text alignment doesn't transfer; few-shot flat; negative result, floor holds.
2026-07-08T06:51:14Z S4-clam-tilemil done ABMIL + multi-fidelity fusion on tumor CONCH bags: patient AUROC 0.607 (best tumor-tile learned recipe; < champ 0.713 & S1 0.646); beats champion on OAUTHC (0.629 vs 0.579), most site-uniform scorer; few-shot below chance <all; floor holds.
2026-07-08T08:41:25Z S5-setencoder-agg done Deep-Sets patient-bag aggregator: patient AUROC 0.516 (~chance); learned aggregator FAILS to beat champion's max/sqrtn on same Wagner scores; worst on single-slide (0.405) & 5+ (0.136) bags; op-point ok (spec95 0.164, npv 0.920). Closes S-phase: no learned recipe beats zero-param champ 0.713.
2026-07-08T09:32:31Z R1-stain-aug-tta done feature-space 5-crop+stain TTA on S4: inter_condition_kappa 0.927 (> MSIntuit 0.82) — S4 calls stable/not brittle to crop+stain; base AUROC ~0.59 (robust but modest). Diagnoses OAUTHC gap as domain-shift not TTA instability. Feature-space proxy (image re-extraction ~55h GPU infeasible).
2026-07-08T10:08:41Z R2-flex-bottleneck done FLEX site-adversarial+text-anchor bottleneck on TITAN: overall AUROC 0.623 (2nd-best learned, strong op-point spec@sens90 0.221) but FAILS site-invariance: every per-site delta vs champ negative (OAUTHC 0.514 vs 0.579, mean|d| 0.209); equalizes sites downward (regression to mean). Steers R-phase to R3/T1. Floor holds.
