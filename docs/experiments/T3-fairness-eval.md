# T3 — fairness evaluation & the group-conditional fairness gate

**Goal.** T1 exposed unequal per-site safety under a global rule-out threshold; T2 showed the
conditioning variable must be *site* (perfectly identifiable) not OOD distance. T3
(`argo_deepmsi/eval/fairness.py`) implements group-conditional (per-site) conformal calibration
and evaluates the honest fairness of the champion-based rule-out screener, ending in a pass/fail
**gate** for the loop stop bar.

**Per-site champion AUROC (the raw disparity):**

| site | n | AUROC |
|------|--:|------:|
| LUTH | 15 | 1.000 |
| LASUTH | 9 | 0.889 |
| retrospective_oau | 23 | 0.800 |
| retrospective_msk | 60 | 0.793 |
| UITH | 10 | 0.750 |
| **OAUTHC** | 64 | **0.579** |

**AUROC gap (max−min) = 0.421** — a large disparity, with OAUTHC (the biggest prospective
Nigerian site) the clear floor. At a **global** rule-out threshold the false-omission rate
ranges **0.000–0.333** across sites: the global screener is dangerously unequal.

**Group-conditional conformal — equalize safety, expose the coverage cost.** Give each site its
own rule-out threshold so its FOR ≤ 10% target, then read off the coverage (fraction safely
ruled out) it can afford:

| site | n | safe rule-out coverage | out-of-sample FOR |
|------|--:|-----------------------:|------------------:|
| LUTH | 15 | 0.743 | 0.000 |
| retrospective_msk | 60 | 0.653 | 0.101 |
| retrospective_oau | 23 | 0.630 | 0.097 |
| **OAUTHC** | 64 | 0.344 | **0.198** |
| LASUTH | 9 | — | too small to calibrate |
| UITH | 10 | — | too small to calibrate |

For the signal-bearing sites the fix **works**: retrospective_oau/msk and LUTH meet the FOR
target out-of-sample and still safely rule out 63–74 % of patients. **Coverage-adjusted gap =
0.399** — even at equalized safety, the benefit is very unevenly distributed.

**Fairness gate: FAIL — OAUTHC is a disqualifying outlier.** OAUTHC is the critical case: even
with its *own* conformal threshold, its out-of-sample FOR is **0.198**, ~2× the 10 % target.
Group-conditional calibration cannot make it safe because **there is no MSI signal on OAUTHC to
calibrate on** — the calibration threshold does not generalize to held-out OAUTHC patients. Two
further sites (LASUTH n=9, UITH n=10) are too small to calibrate or gate at all. The gate
therefore **FAILS**: the champion-plus-abstention screener cannot be fairly deployed on this
cohort as-is.

**What this means for the mission.** This is the honest capstone of the S+R+T arc:

1. The zero-param champion (patient AUROC 0.713) is the best available scorer; no learned FM
   recipe beat it (S-phase).
2. It is robust to crop/stain (R1) but site-spiky; its weakness on OAUTHC is a genuine
   *absence of recoverable MSI signal* in CONCH features at n=181, not test-time instability
   (R1) or fixable domain shift (R2/R3 both failed identically).
3. Framed as a selective rule-out screener it delivers real value on the sites it understands
   (~49 % coverage at ~9 % FOR overall; 63–74 % per-site at equal safety for 3 sites), but
4. it **cannot fairly serve OAUTHC** — the fairness gate disqualifies as-is deployment.

The productive path is therefore **not** "beat 0.713 on the whole cohort" but "deploy the
selective screener on the sites it can serve safely, and treat OAUTHC as requiring either more
data or a different (non-CONCH) signal." The fairness gate makes that boundary explicit and
auditable.

**Gate wiring.** `fairness_report()` returns `fairness_gate.pass` (bool) + the disqualifying
outlier list; the loop stop bar's "no disqualifying per-site outlier" clause reads this. Here it
is **False**, so the champion path to LOOP-COMPLETE is (correctly) not open — consistent with the
champion also being far from the spec 0.47 @ sens 0.96 operating target.

**Files.** `argo_deepmsi/eval/fairness.py`, `tests/test_fairness.py`,
`results/analysis/fairness/fairness_report.json`. Analysis module, no board entry, no new deps.
