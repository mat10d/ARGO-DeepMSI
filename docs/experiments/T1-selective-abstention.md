# T1 — selective / conformal abstention screener

**Motivation.** The R-phase showed the model cannot be made *right* on OAUTHC (site and MSI
signal are entangled; both site-invariance methods failed). The correct deployment behaviour
for a rule-out screener is then to *know when it is unsure and abstain* — the MSIntuit-style
value proposition: rule out (skip molecular testing on) the confidently-negative patients,
refer the rest, and control the dangerous error (missing an MSI-H patient).

**Method** (`argo_deepmsi/scorers/selective_abstention.py`). Wrap the zero-param champion
(`calibrated_pool` = max/√n over Wagner per-slide p_msih) in a selective rule-out screener:

- **Rule-out** = call the lowest-scoring patients MSS-negative; **coverage** = fraction ruled
  out (workload saved); **risk = false-omission rate (FOR)** = P(MSI-H | ruled out) = 1 − NPV
  on the covered set — the dangerous miss.
- **Risk–coverage curve**: rule out the lowest scores first; coverage ↑ ⇒ FOR ↑.
- **Split-conformal guarantee**: calibrate a rule-out threshold τ on half the patients so
  calibration FOR ≤ target (10%), evaluate coverage + FOR on the other half (50 repeats).
- **Group-conditional (per-site) coverage**: the fairness view.

Board score = champion score (this is the champion *plus* an abstention overlay); no new
ranking. Patient-resolution, CPU, no training.

**Headline — the rule-out works.** At a **target FOR of 10%**, split-conformal confidently rules
out **48.6% of patients** (empirical covered FOR 9.3%, within target). Roughly *half* the cohort
can skip molecular MSI testing while the missed-MSI rate stays ≈ 9%. AURC(FOR) = 0.130; champion
NPV@sens95 = 0.926.

**Risk–coverage curve (rule-out FOR):**

| coverage | 0.25 | 0.50 | 0.75 | 1.00 |
|----------|-----:|-----:|-----:|-----:|
| FOR | 0.111 | 0.089 | 0.169 | 0.227 |

Note the curve is **not perfectly monotone** (FOR at 25% coverage 0.111 > at 50% coverage
0.089): the champion's ranking is imperfect at the very bottom, so a few MSI-H patients receive
among the lowest scores and are ruled out early. This is itself a safety caveat — the most
confident rule-outs are not the safest.

**The fairness problem — global conformal is unfair (the T3 hook).** At the global τ (target FOR
10%), per-site rule-out coverage and FOR:

| site | n | rule-out coverage | FOR |
|------|--:|------------------:|----:|
| LASUTH | 9 | 0.360 | 0.000 |
| LUTH | 15 | 0.525 | 0.000 |
| retrospective_oau | 23 | 0.472 | 0.047 |
| retrospective_msk | 60 | 0.553 | 0.078 |
| **OAUTHC** | 64 | 0.443 | **0.134** |
| **UITH** | 10 | 0.514 | **0.288** |

A **global** score threshold does *not* abstain more on the weak sites — it rules out OAUTHC
(44%) and UITH (51%) at rates similar to the good sites, but with **much higher miss rates**
(FOR 0.134 and 0.288 vs ≤ 0.078 elsewhere). The screener is therefore **systematically more
dangerous for exactly the sites the model understands least** — it rules out OAUTHC/UITH patients
it should be referring. The threshold is blind to OOD because a low champion score on OAUTHC does
not mean "confidently MSS", it means "the model has no signal here."

**Verdict.** Selective rule-out is the right deployment frame and delivers a real, MSIntuit-style
operating point (≈ 49% workload saved at ≈ 9% global FOR). But a *global* conformal threshold
violates per-site safety: it must be replaced by **group-conditional (per-site) conformal
calibration** so each site meets the FOR target independently — which will *lower* OAUTHC/UITH
coverage (abstain more there) in exchange for equal safety. That group-conditional guarantee and
its coverage-adjusted fairness gap is exactly **T3 (fairness gate)**; the per-site coverage
signal it needs is sharpened by **T2 (OOD score)**, which gives a site-aware confidence instead
of the raw champion score.

**Files.** `results/scorers/selective_abstention/{patient_scores.csv, metrics.json}` —
metrics.json carries `coverage_risk`, `conformal`, and `per_site_coverage`. No board regen
(patient-resolution; board score ties the champion).
