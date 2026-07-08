# S4 — attention-MIL on tumor tile bags with multi-fidelity fusion

**Method.** Gated attention-MIL (ABMIL, Ilse et al. 2018) on tumor-only CONCH tile bags,
with the **Multi-Fidelity Model Fusion** variance-reduction of Mammadov et al. (MICCAI 2025 /
arXiv 2507.00292):

1. **Bags** (`scripts/build_tilemil_bags.py`, CPU): each in_clean_set slide's cached
   `conch_v1.5_tiles` restricted to the Q3 tumor tiles, capped at 500 random tiles/slide
   (428 bags, 140,851 tiles, 768-d).
2. **ABMIL**: a small projection (768→192) + gated attention pooling + linear head, trained
   per bag with class-weighted BCE (Adam, dropout 0.25).
3. **Multi-fidelity fusion**: MIL runs vary by 10–15 AUC points across init/batch-order/lr, so
   within every patient-grouped outer fold we train M models on an inner-train split, select
   the top-k by inner-validation patient AUROC, and average their test predictions (full-shot
   M=3, top-2, 12 epochs). The reported OOF is thus a fused, lower-variance estimate.

Few-shot curve K∈{1,2,4,8,16,all}. CPU-only (ABMIL on frozen features).

**Headline vs MSIntuit** (sens 0.96–0.98 @ spec 0.46–0.47). On the clean cohort (181
patients): patient AUROC **0.607**, AUPRC 0.315, spec@sens90 0.086, **spec@sens95 0.071**,
NPV@sens95 0.833.

**Verdict — best tumor-tile learned recipe, still below the champion.** S4 (0.607) is the
strongest of the tumor-tile learned scorers — clearly above S2 (ProtoNet 0.521) and S3
(Tip-Adapter 0.444) — but remains below the zero-param champion `calibrated_pool` (0.713),
S1's slide-FM linear probe (0.646), and `simple_grid` (0.646). A learned tile-attention head
does extract more MSI signal from tumor tiles than distance/text/prototype heads, but not
enough to beat max/√n pooling of Wagner scores on this cohort.

**One encouraging signal — site uniformity.** S4 is the most site-balanced scorer yet:

| site | n | AUROC | (champion AUROC) |
|------|--:|------:|-----------------:|
| LASUTH | 9 | 0.556 | 0.889 |
| LUTH | 15 | 0.462 | 1.000 |
| **OAUTHC** | 64 | **0.629** | 0.579 |
| UITH | 10 | 0.625 | 0.750 |
| retrospective_msk | 60 | 0.716 | 0.793 |
| retrospective_oau | 23 | 0.511 | 0.800 |

Notably S4 **beats the champion on OAUTHC** (0.629 vs 0.579) — the site the R-phase targets. The
attention-MIL's signal is flatter across sites (no site above 0.72, none catastrophically
low on the large sites) where the champion is spiky (great on small clean sites, weak on
OAUTHC). This makes S4 a candidate base for the R-phase site-robustness work even though its
overall AUROC trails.

**Few-shot learning curve (patient AUROC):**

| K / class | 1 | 2 | 4 | 8 | 16 | all (37) |
|-----------|--:|--:|--:|--:|---:|---------:|
| ABMIL-fusion | 0.526 | 0.443 | 0.459 | 0.491 | 0.498 | **0.607** |

Below chance at small K (a fused ABMIL trained on 1–16 patients/class is unstable even with
the variance-reduction fusion); the signal only materialises at full-shot. Consistent with
S1–S3: none of these recipes is genuinely few-shot capable on this cohort.

**Per-bag-size (patient AUROC):** 1→0.609, 2→0.573, 3–4→0.719, 5+→0.500 (n=13). Multi-slide
patients (3–4) score best, as with S1.

**Files.** `results/scorers/clam_tilemil/{bags_concat.npy, bags_index.csv, bags_info.json,
slide_scores.csv, metrics.json, few_shot_curve.csv}`. `compute_batch` is cache-first (reads the
trained OOF; only the runner retrains) so the leaderboard re-race stays cheap. Leaderboard
re-raced to include S4.

**Recommendation.** S4 is the best learned MSI scorer so far and the most site-robust — carry
it (not the spiky champion) into the R-phase site-invariance experiments (R1 stain-aug, R2
FLEX bottleneck), where its flat per-site profile is the right starting point.
