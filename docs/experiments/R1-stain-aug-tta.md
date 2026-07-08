# R1 — 5-crop + stain-augmentation test-time robustness (κ vs MSIntuit)

**Goal.** MSIntuit's headline robustness claim is an inter-scanner Cohen's **κ = 0.82**: its
MSI call rarely flips when the same slide is re-imaged on a different scanner. R1 measures
the analogous stability of our best tile-level scorer (**S4 `clam_tilemil`**, the most
site-uniform) under test-time augmentation.

**Method — feature-space TTA (documented proxy).** True H&E stain augmentation perturbs raw
pixels and re-extracts the foundation model; on this cohort that is ~55 h of GPU
re-extraction (428 slides × conditions × an 11 h CONCH pass) — infeasible in the autorun
loop, and the pipeline retains only frozen CONCH tile features. So R1 augments in feature
space (`scripts/run_stainaug_tta.py`):

- **5-crop** → 5 tile-subset views per slide (each a 70 % bootstrap of the slide's tumor
  tiles) — spatial crop / field-of-view variation.
- **stain** → per-view feature perturbation: a per-dimension log-normal gain (σ=0.10) plus
  Gaussian jitter (0.05 × per-dim std) — the embedding shift a stain/scanner change induces.

Six conditions (identity + 5 crop-stain views). Each trained S4 fold-model scores each
held-out slide under every condition; slide scores → patient (max/√n) → binarized at the
identity condition's sens-95 threshold. **inter_condition_kappa** = mean pairwise Cohen's κ
across the six conditions.

**Headline result.** **inter_condition_kappa = 0.927** — *above* MSIntuit's inter-scanner
κ = 0.82. The attention-MIL's MSI calls are highly stable under crop + feature-space stain
perturbation; predictions almost never flip across conditions.

**Per-condition AUROC (stability of ranking, not just the binary call):**

| condition | identity | crop-stain 0–4 |
|-----------|---------:|:--------------:|
| patient AUROC | 0.588 | 0.584 / 0.587 / 0.593 / 0.591 / 0.597 |

Ranking is essentially invariant (spread 0.584–0.597). The identity AUROC (0.588) is within
noise of S4's headline 0.607 (R1 re-trains with identity-based inner-val model selection).

**Interpretation.** S4 clears MSIntuit's robustness bar (κ 0.927 > 0.82) — the property that
matters for a deployable rule-out screener that must survive scanner/stain drift across the
Nigerian sites. But this is robustness *of a modest base*: AUROC ≈ 0.59 and spec@sens95 0.086
are still far below the champion (0.713 / 0.179). **A stable-but-weak predictor is robustly
wrong as often as robustly right** — high κ is necessary, not sufficient. The value of R1 is
diagnostic: it confirms the attention-MIL is not brittle to crop/stain variation, so the
OAUTHC site gap (per-site AUROC OAUTHC 0.627, LUTH 0.385, UITH 0.417) is a *representation /
domain-shift* problem, not a test-time-instability problem — pointing the remaining R-phase
(R2 FLEX bottleneck, R3 fmMAP) at feature-space site-invariance rather than TTA.

**Caveat.** Feature-space stain augmentation is a proxy for true image-level H&E augmentation;
the κ would likely be *lower* under real stain normalization variance (which perturbs
features more structurally than per-dimension gain/jitter). Read κ = 0.927 as an
*upper bound* on robustness, and the qualitative conclusion (stable, not brittle) as the
finding — not the exact figure as directly interchangeable with MSIntuit's image-level κ.

**Per-site / per-bag-size (identity condition).** By site AUROC: LASUTH 0.500, LUTH 0.385,
OAUTHC 0.627, UITH 0.417, retrospective_msk 0.663, retrospective_oau 0.533. By bag size:
1→0.544, 2→0.590, 3–4→0.737, 5+→0.136. Consistent with S4.

**Files.** `results/scorers/clam_tilemil_stainaug/metrics.json` (full block +
inter_condition_kappa + per_condition_auroc). No leaderboard entry (R1 is a robustness
analysis, not a board scorer).
