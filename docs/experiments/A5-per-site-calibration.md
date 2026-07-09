# A5 — OAUTHC-specific operating-point calibration

## Method
A pooled rule-out threshold set for sensitivity 0.95/0.96 on the WHOLE cohort need not hold
that sensitivity ON OAUTHC. A5 fits an OAUTHC-specific operating point on the **best A-phase
scorer (A0 Harmony, OAUTHC AUROC 0.683)** and reports the honest OAUTHC spec@sens / NPV vs
the global-threshold baseline. Patient-level (Harmony max/√n), patient-grouped 5-fold CV over
OAUTHC to avoid threshold-selection optimism. Three operating points per target sensitivity:
`global` (threshold fit on the full cohort), `per_site_threshold` (CV-fit on OAUTHC),
`per_site_isotonic` (CV isotonic recalibration + threshold on OAUTHC). No new data.
`argo_deepmsi/eval/per_site_calibration.py` (+ tests).

## Headline (OAUTHC operating points, n=64 pt, 13 MSI-H)
| target sens | method | threshold | achieved sens | spec | NPV |
|---|---|---|---|---|---|
| 0.95 | **global** | 0.033 | **1.000** | 0.275 | **1.000** |
| 0.95 | per_site_threshold | 0.043 | 0.923 | 0.294 | 0.938 |
| 0.95 | per_site_isotonic | 0.163 | 0.923 | 0.294 | 0.938 |
| 0.96 | **global** | 0.027 | **1.000** | 0.275 | **1.000** |
| 0.96 | per_site_threshold | 0.043 | 0.923 | 0.294 | 0.938 |
| 0.96 | per_site_isotonic | 0.163 | 0.923 | 0.294 | 0.938 |

## Interpretation — the global threshold is already honest (even conservative) on OAUTHC
1. **No miscalibration to fix.** At the pooled rule-out threshold, OAUTHC achieves **sensitivity
   1.000 and NPV 1.000** (all 13 OAUTHC MSI-H patients are caught; every ruled-out OAUTHC patient
   is truly MSS), at spec 0.275. The feared failure mode — a global threshold silently
   under-catching OAUTHC positives — **does not occur** after Harmony correction.
2. **Per-site recalibration does not help.** CV-fitting the threshold (or isotonic) on OAUTHC
   trades sensitivity **down** (1.000 → 0.923, i.e. it starts missing ~1 of 13 positives) for a
   negligible specificity gain (0.275 → 0.294) and a worse NPV (1.000 → 0.938). For a rule-out
   screener, giving up sensitivity/NPV for +0.02 specificity is the wrong trade.
3. **The residual OAUTHC limitation is DISCRIMINATION, not calibration.** OAUTHC's ceiling is
   set by Harmony's AUROC (0.683) — the score *ranks* OAUTHC patients only moderately — not by a
   mis-placed operating point. Better OAUTHC rule-out needs better OAUTHC discrimination (A0–A3
   showed how hard that is at n=64), not threshold tuning.

## Verdict vs targets
- **vs MSIntuit (spec 0.46 @ sens 0.96):** OAUTHC holds sens 1.0 / NPV 1.0 at the global
  threshold but only **spec 0.275** — a *safe but low-yield* rule-out on OAUTHC (rules out ~28%
  of OAUTHC MSS patients with zero missed MSI-H in this cohort), still short of MSIntuit's spec.
- **Practical read:** deploy the GLOBAL Harmony threshold on OAUTHC — it is already the honest,
  safe operating point (sens/NPV 1.0). Do not per-site-recalibrate; it only costs sensitivity.
- Feeds A6: the OAUTHC rule-out is safe but low-specificity; the gap to MSIntuit is discrimination.

Output: `results/analysis/per_site_calibration/oauthc_operating_points.csv`.
