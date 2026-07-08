# S5 — Deep-Sets patient-bag aggregator (cardinality-calibrated)

**Method.** The champion aggregates a patient's per-slide Wagner scores with a fixed
max/√n heuristic. S5 replaces that with a *learned*, permutation-invariant Deep-Sets
aggregator (Zaheer et al. 2017) to test whether a cardinality-calibrated set function can
beat the heuristic. Per patient, each slide contributes [Wagner p(MSI-H) ‖ TITAN slide
embedding] (1+768-d); a shared encoder φ maps each slide, the bag is pooled mean⊕max, the
cardinality features [n, √n, log(1+n)] are appended, and an MLP head ρ produces a patient
logit. A 3-model ensemble is averaged; class-weighted BCE, patient-grouped CV; few-shot
curve K∈{1,2,4,8,16,all}. Frozen features only (CPU).

**Headline vs MSIntuit** (sens 0.96–0.98 @ spec 0.46–0.47). On the clean cohort (181
patients): patient AUROC **0.516**, AUPRC 0.238, spec@sens90 0.164, **spec@sens95 0.164**,
NPV@sens95 **0.920**.

**Verdict — negative result: the learned aggregator does not beat max/√n.** S5's AUROC
(0.516) is near chance and far below the champion `calibrated_pool` (0.713), which uses the
*same* Wagner per-slide scores with a hand-picked max/√n. A learned set aggregator has too
little to work with here — ~181 patients is not enough to fit a Deep-Sets that improves on a
good closed-form cardinality correction; it overfits the training bags and generalises no
better than chance on ranking.

**The irony — it fails exactly where it should help.** By bag size:

| slides/patient | n | AUROC |
|---------------:|--:|------:|
| 1 | 77 | **0.405** |
| 2 | 35 | 0.654 |
| 3–4 | 56 | 0.621 |
| 5+ | 13 | **0.136** |

The aggregator built to fix bag-size inflation scores *below chance* on single-slide
patients (43 % of the cohort — where there is no set to aggregate, just the raw Wagner score
passed through the net) and collapses on the 5+ bucket. Multi-slide patients (2–4) are the
only ones above chance. The learned cardinality calibration did not materialise.

**One redeeming feature — the operating point survives.** Despite chance-level AUROC, S5's
rule-out operating point is nearly champion-grade: **spec@sens95 0.164** (champion 0.179) and
**NPV 0.920** (champion 0.926). The learned head still ranks the most-confident negatives
well enough for a comparable high-sensitivity threshold, even though global ordering is poor.

**Few-shot curve (patient AUROC):** K=1→0.504, 2→0.463, 4→0.502, 8→0.494, 16→0.552,
all→0.516. Flat around chance — no few-shot benefit, consistent with S1–S4.

**Per-site (patient AUROC):** LASUTH 0.444, LUTH 0.538, OAUTHC 0.376, UITH 0.375,
retrospective_msk 0.616, retrospective_oau 0.578. Weak everywhere, notably poor on OAUTHC
(0.376) — the opposite of S4, which was the OAUTHC standout.

**S-phase verdict.** S5 closes the low-N scorer phase. Across all five (S1 0.646, S2 0.521,
S3 0.444, S4 0.607, S5 0.516), **none beats the zero-param champion (0.713)** on this
198→181-patient cohort. The learned/few-shot FM recipes borrowed from the literature do not
transfer to our small, site-shifted Nigerian cohort. The two signals worth carrying forward:
(1) **S4** (attention-MIL) is the best learned scorer and the most site-uniform — the base for
the R-phase; (2) the **operating-point-vs-AUROC gap** (S1 spec@sens95 0.229 > champion 0.179;
S5 NPV 0.920) suggests the fusion (F1) and abstention (T1) phases, not raw AUROC, are where
the MSIntuit spec@sens gap might still close.

**Files.** `results/scorers/setencoder_agg/{patient_scores.csv, metrics.json,
few_shot_curve.csv}`. Leaderboard re-raced to include S5.
