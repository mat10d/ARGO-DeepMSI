# S2 — ProtoNet few-shot on cluster-aggregated CONCH tumor tiles

**Method.** The FM-MSI benchmark recipe ("Benchmarking Pathology Foundation Models for
Predicting Microsatellite Instability in Colorectal Cancer Histopathology", *Computerized
Medical Imaging and Graphics* 2025, PII S0895611125001892). Two stages:

1. **Cluster-aggregation** (`scripts/build_protonet_features.py`, CPU): for each in_clean_set
   slide, load cached CONCH tile features (`conch_v1.5_tiles`, 768-d) and keep only the Q3
   **tumor tiles**. A single global MiniBatchKMeans (G=8) is fit unsupervised on the pooled
   tumor tiles (70,896 sampled across 428 slides) — no labels, so it is a fixed feature
   transform, not leakage. Each slide's tumor tiles are assigned to the 8 clusters, the
   per-cluster mean is L2-normalised, and the 8 group means are concatenated → a 6144-d WSI
   embedding.
2. **ProtoNet** (`argo_deepmsi/scorers/protonet_cluster.py`): patient-grouped 5-fold CV; per
   fold a StandardScaler is fit train-only, class prototypes are the per-class training means,
   and each test WSI's p(MSI-H) is the softmax over negative squared distances to the two
   prototypes. Few-shot curve K∈{1,2,4,8,16,all} = K patients/class forming the prototypes
   (10 random draws averaged). Frozen features only.

**Headline vs MSIntuit** (sens 0.96–0.98 @ spec 0.46–0.47) and the FM-MSI benchmark (FM-MSI benchmark (CONCH; ScienceDirect PII S0895611125001892 -- closed-access, full text NOT obtained; operating points NOT transcribed, do not cite figures)). On our clean cohort (428 slides / 181
patients): patient AUROC **0.521**, AUPRC 0.264, spec@sens90 0.157, **spec@sens95 0.086**,
spec@sens96 0.014, NPV@sens95 0.857.

**Verdict — negative result.** ProtoNet on cluster-aggregated CONCH tumor tiles is
**essentially at chance** (0.521) on this cohort — the worst scorer on the board, far below
the zero-param champion (`calibrated_pool` 0.713), S1's linear probe (0.646), and the FM
benchmark (whose operating points we could not obtain from the closed-access paper). The few-shot curve is flat at ≈0.51 across every
K, i.e. adding shots does not help because the base feature space barely separates the
classes. The `_proto_scores` unit test confirms the head is correct on separable data, so
this reflects weak MSI signal in unsupervised cluster-mean features + a distance classifier,
not an implementation bug. Two compounding causes: (a) a distance-to-prototype head is weaker
than a learned linear head (S1 got 0.646 from the same CONCH via TITAN); (b) global k-means
cluster means average away the discriminative sub-population. The paper's stronger CONCH
comparator cohorts (TCGA/PAIP) lack our Nigerian site shift; we did not obtain its exact operating points.

**Few-shot learning curve (patient AUROC):**

| K / class | 1 | 2 | 4 | 8 | 16 | all (37) |
|-----------|--:|--:|--:|--:|---:|---------:|
| ProtoNet | 0.513 | 0.529 | 0.506 | 0.497 | 0.508 | **0.521** |

Flat — the recipe is not low-N capable here, echoing S1 (linear probe also near chance at
K≤16). Distance-based few-shot does not rescue the weak feature separation.

**Per-site (patient AUROC / spec@sens95):**

| site | n | prev | AUROC | spec@sens95 |
|------|--:|-----:|------:|------------:|
| LASUTH | 9 | 0.33 | 0.611 | 0.000 |
| LUTH | 15 | 0.13 | 0.923 | 0.923 |
| OAUTHC | 64 | 0.20 | 0.516 | 0.000 |
| UITH | 10 | 0.40 | 0.500 | 0.000 |
| retrospective_msk | 60 | 0.23 | 0.509 | 0.174 |
| retrospective_oau | 23 | 0.22 | 0.378 | 0.000 |

Site AUROCs straddle chance in both directions (LUTH 0.92 on n=15 is small-n noise;
retrospective_oau 0.378 is below chance). No coherent signal — consistent with the near-0.5
overall.

**Per-bag-size (patient AUROC):** 1→0.577, 2→0.526, 3–4→0.521, 5+→0.091 (n=13). No bag-size
lifts the score above chance.

**Note on cohort scope.** S2 is defined only on the tumor-filtered clean cohort (it needs Q3
tumor tiles, computed only for in_clean_set slides), so its "dirty" and "clean" passes are the
same 181 patients — unlike the zero-shot scorers that also score the full 217-patient dirty
set.

**Files.** `results/scorers/protonet_cluster/{cluster_features.npy, cluster_metadata.csv,
cluster_info.json, slide_scores.csv, metrics.json, few_shot_curve.csv}`. Leaderboard re-raced
to include S2. **Recommendation:** de-prioritise distance-based few-shot on unsupervised
cluster features; the S3 (Tip-Adapter on CONCH text prompts) and S4 (attention-MIL) recipes
carry a stronger inductive bias for the low-N regime.
