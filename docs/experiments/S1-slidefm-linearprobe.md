# S1 — linear probe on slide-FM embeddings (TITAN / PRISM)

**Method.** A class-balanced logistic-regression head fit 5-fold patient-grouped
(`StratifiedGroupKFold`, seed 42) on the already-extracted slide-level foundation-model
embeddings — TITAN (`conch_v1.5_titan`, 768-d) and PRISM (`virchow2_prism`, 1280-d).
Slide-FM embeddings are one vector per slide (no tile aggregation), so this is a genuine
low-N recipe: a linear head on frozen features, no FM finetuning, no external fitting.
Evaluated on the Q4 frozen clean cohort (`in_clean_set`, 428 slides / 181 patients);
patient scores via max/√n. Few-shot curve K ∈ {1,2,4,8,16,all}: for each K the head sees
only K patients per class per fold (10 random draws averaged), tested on the held-out
fold. Scorer `slidefm_linearprobe`; run via `scripts/run_slidefm_linearprobe.py` (CPU).

**Headline vs MSIntuit** (target sens 0.96–0.98 @ spec 0.46–0.47, κ 0.82). Best embedding
**TITAN**, full-shot: patient AUROC **0.646**, AUPRC 0.326, **spec@sens95 0.229**,
spec@sens90 0.243, spec@sens96 0.157, **NPV@sens95 0.941** (n=181). TITAN and PRISM are
near-identical (0.646 vs 0.643).

**Verdict vs champion.** Below the zero-param champion on AUROC (0.646 < `calibrated_pool`
0.713) — a supervised linear head on slide-FM vectors does **not** beat max/√n pooling of
Wagner scores here. But its **operating point is better**: spec@sens95 **0.229 vs 0.179**
and NPV 0.941 vs 0.926. For a rule-out screener the operating point is what matters, so S1
is not dominated — it trades ranking (AUROC) for a higher-specificity floor. Still far from
MSIntuit's spec 0.46; the tumor-only MIL / few-shot recipes (S2–S5) target that gap.

**Few-shot learning curve (patient AUROC):**

| K / class | TITAN | PRISM |
|----------:|------:|------:|
| 1 | 0.515 | 0.539 |
| 2 | 0.507 | 0.499 |
| 4 | 0.504 | 0.523 |
| 8 | 0.528 | 0.556 |
| 16 | 0.550 | 0.525 |
| all (37) | **0.646** | 0.643 |

At K ≤ 16 patients/class both embeddings sit near chance (0.50–0.56); the signal only
emerges with the full ~37 positives. A linear probe on slide-FM vectors is **not** few-shot
capable on this cohort — motivating the cluster/ProtoNet and Tip-Adapter recipes (S2, S3)
that are designed for the low-N regime.

**Per-site (patient AUROC / spec@sens95):**

| site | n | prev | AUROC | spec@sens95 |
|------|--:|-----:|------:|------------:|
| LASUTH | 9 | 0.33 | 0.611 | 0.333 |
| LUTH | 15 | 0.13 | 0.423 | 0.154 |
| OAUTHC | 64 | 0.20 | **0.608** | 0.118 |
| UITH | 10 | 0.40 | 0.458 | 0.167 |
| retrospective_msk | 60 | 0.23 | 0.696 | 0.217 |
| retrospective_oau | 23 | 0.22 | 0.789 | 0.556 |

OAUTHC is again weak (0.608), consistent with the diagnosed site-robustness problem the
R-phase targets. Small sites (LUTH, UITH, LASUTH) are noisy at n<16.

**Per-bag-size (patient AUROC):**

| slides/patient | n | AUROC |
|---------------:|--:|------:|
| 1 | 77 | 0.608 |
| 2 | 35 | 0.726 |
| 3–4 | 56 | 0.696 |
| 5+ | 13 | 0.364 |

Multi-slide patients (2–4) score best; the 5+ bucket collapses (n=13, noisy), and
single-slide patients (43 % of the cohort) sit at 0.608 — the bag-size dependence S5's
cardinality-calibrated aggregator is meant to fix.

**Files.** `results/scorers/slidefm_linearprobe/{slide_scores.csv, metrics.json,
few_shot_curve.csv, embedding_ranking.csv}`. Full metric block in `metrics.json`; the
leaderboard was re-raced (`qc_comparison --clean-csv`) to include S1.
