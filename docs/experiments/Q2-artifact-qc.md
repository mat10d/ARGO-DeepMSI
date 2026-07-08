# Q2 — GrandQC artifact-QC layer

**Method.** Ran GrandQC artifact segmentation (`grandqc-artifact`, 7x variant) over the
509-slide clean cohort via a 3-way SLURM GPU array (`scripts/artifact_qc.sh`, job 10300717,
3×~3h on nvidia-A6000). Each shard writes `artifact_qc.part{0,1,2}.csv` with per-slide
`artifact_fraction` = polygon area of artifact masks / tissue area. Reduced with
`python -m argo_deepmsi.eval.cohort --artifact-qc-dir results/data/artifact_qc`, which merges
the shards, attaches `artifact_fraction`, and flags `passes_artifact_qc = artifact_fraction <=
0.5`. **Flag-only:** `in_clean_set` is unchanged; dropping is deferred to Q3/Q4.

**Headline.** 507/509 clean slides scored (2 retrospective_oau slides unreadable → NaN →
flagged). Not a modeling result — this is a QC-instrumentation layer, so there is no
spec@sens vs MSIntuit to report yet. No-regression floor unchanged (best clean AUROC 0.7105).

**Per-site flag counts (floor = 0.5):**

| site | n in clean set | scored | flagged artifact |
|---|---|---|---|
| LASUTH | 11 | 11 | 10 |
| LUTH | 21 | 21 | 19 |
| OAUTHC | 196 | 196 | 194 |
| UITH | 12 | 12 | 12 |
| retrospective_msk | 97 | 97 | 19 |
| retrospective_oau | 172 | 170 | 138 |
| **total** | **509** | **507** | **392** |

**Verdict.** Layer landed; `cohort_clean.csv` gains `artifact_fraction` + `passes_artifact_qc`,
manifest gains the `artifact_qc` block with per-site counts. **Caveat for Q3/Q4:** at floor 0.5
the flag rate is 77% (392/507) and is strongly site-skewed — the Nigerian-imaged sites
(OAUTHC 99%, UITH 100%) flag near-universally while retrospective_msk flags only 20%. This is
consistent with the known site/scanner covariate shift, not necessarily 77% genuinely
unusable tissue. Q3/Q4 must NOT drop on this flag naively at floor 0.5 — either recalibrate
the floor per the artifact_fraction distribution or treat the flag as a covariate rather than
an exclusion. Recorded flag-only so the floor decision is made with the tumor-filter smoke-off
in hand.
