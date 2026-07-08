# R3 — fmMAP: supervised-UMAP site-suppressing projection + linear probe

**Method.** R2's adversarial site-invariance (FLEX) equalized sites downward. R3 tries the
gentler, projection-based alternative (`argo_deepmsi/scorers/fmmap_probe.py`), per
patient-grouped fold:

1. **Site residualization** — subtract each site's train-set mean TITAN feature (first-order
   batch correction; test slides use their site's train mean, global fallback for unseen sites).
2. **Supervised UMAP (fmMAP)** — `umap.UMAP(n_components=15, target_metric='categorical', y=MSI)`
   fit on residualized train features → an MSI-guided manifold; `transform` test (out-of-sample,
   no test labels → no leakage).
3. **Linear probe** — class-balanced logistic regression on the UMAP coordinates.

Reports per-site AUROC and Δ vs champion. Frozen features only (CPU).

**Headline.** On the clean cohort (181 patients): patient AUROC **0.550**, AUPRC 0.273,
spec@sens90 0.071, spec@sens95 0.057, NPV@sens95 0.800. This is *below* R2 (0.623) and well
below the champion (0.713) — the UMAP projection discards even more MSI signal than the FLEX
bottleneck.

**Site-invariance verdict — fails, same as R2.** Per-site AUROC and Δ vs champion
(`calibrated_pool`):

| site | n | fmMAP AUROC | champion | Δ |
|------|--:|-----------:|---------:|---:|
| LASUTH | 9 | 0.778 | 0.889 | −0.111 |
| LUTH | 15 | 0.731 | 1.000 | −0.269 |
| **OAUTHC** | 64 | **0.499** | 0.579 | **−0.080** |
| UITH | 10 | 0.333 | 0.750 | −0.417 |
| retrospective_msk | 60 | 0.536 | 0.793 | −0.258 |
| retrospective_oau | 23 | 0.667 | 0.800 | −0.133 |

Every delta is negative (mean |Δ| = 0.211, essentially identical to R2's 0.209). On **OAUTHC —
the target — fmMAP is at chance (0.499)**, worse than the champion (0.579). The site-residualize
+ supervised-UMAP pipeline retains a little more on the tiny sites (LASUTH 0.778, LUTH 0.731)
but destroys signal on the large sites (OAUTHC, retrospective_msk 0.536) — the projection to 15
UMAP dims is a hard information bottleneck that, like R2's adversary, cannot separate site from
MSI at this cohort size.

**Few-shot curve (patient AUROC):** K=1→0.534, 2→nan (UMAP degenerate on 2 patients/class),
4→0.531, 8→0.521, 16→0.496, all→0.550. Flat around chance; UMAP is unstable at low K.

**Per-bag-size (patient AUROC):** 1→0.598, 2→0.598, 3–4→0.581, 5+→0.000 (n=13).

**Verdict — the R-phase site-invariance thesis is falsified on this cohort.** R2 (adversarial
bottleneck) and R3 (projection) are two independent feature-space site-suppression methods, and
**both fail identically**: neither closes the OAUTHC gap, both regress every site vs the champion,
both land at mean |Δ| ≈ 0.21. The consistent conclusion: **site and MSI signal are entangled in
these frozen features at n=181, so any method that removes site removes MSI too.** Combined with
R1 (the base scorer is robust, not brittle), the OAUTHC failure is not fixable by making the
model site-invariant — the MSI signal on OAUTHC is genuinely weak in CONCH space.

**This is the pivot to the T-phase.** If you cannot make the model right on OAUTHC, the correct
deployment behaviour is to **know when it is wrong and abstain** — a selective/conformal screener
(T1) driven by an OOD score (T2), evaluated for fairness (T3). That is where a
MSIntuit-competitive *rule-out* story can still be told: high sensitivity on the sites the model
understands, principled abstention on the OOD (OAUTHC-type) slides, with a coverage guarantee.

**Files.** `results/scorers/fmmap_probe/{slide_scores.csv, metrics.json, few_shot_curve.csv}`.
Leaderboard re-raced to include R3.
