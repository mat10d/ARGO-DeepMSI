# T2 — OOD (covariate-shift) characterization

**Goal.** T1 abstains on low champion confidence but a low score on OAUTHC means "no signal",
not "confidently MSS". T2 builds a *site-aware* OOD score (how far each slide is from the
in-distribution feature manifold) to test whether covariate shift is the actionable signal the
abstention gate should key on. Two OOD scores on frozen TITAN (CONCH v1.5) slide features
(`argo_deepmsi/eval/ood.py`): **Mahalanobis** (parametric, shrunk covariance) and **kNN** (mean
distance to k=10 nearest in-distribution neighbours). Leave-site-out: each site treated as OOD
vs the rest; OOD-AUROC = how separable that site is.

**Finding 1 — covariate shift is pervasive, not OAUTHC-specific.**

| site | n | kNN OOD-AUROC | Mahalanobis OOD-AUROC |
|------|--:|--------------:|----------------------:|
| retrospective_msk | 61 | 0.976 | 1.000 |
| retrospective_oau | 146 | 0.917 | 0.999 |
| LUTH | 18 | 0.865 | 1.000 |
| LASUTH | 9 | 0.863 | 1.000 |
| **OAUTHC** | 182 | **0.839** | **1.000** |
| UITH | 12 | 0.561 | 0.983 |
| **mean** | | **0.837** | **0.998** |

*Every* site is a strongly detectable covariate shift (kNN mean 0.837; Mahalanobis essentially
perfect at 0.998 — in 768-d TITAN space each site occupies a distinct region). **OAUTHC is not
uniquely OOD** — on kNN it is the *most in-distribution* of the large sites (0.839), and on
Mahalanobis every site including OAUTHC separates perfectly. The Nigerian-site failure is
therefore *not* "OAUTHC is the most out-of-distribution slide set." All six sites carry heavy,
distinct site signatures in the FM features.

**Finding 2 — OOD-ness does NOT predict error (the actionable negative).** Does a higher OOD
score flag the patients the champion gets wrong?

| method | OOD→error AUROC | Spearman ρ |
|--------|----------------:|-----------:|
| kNN | 0.552 | +0.087 |
| Mahalanobis | 0.545 | +0.074 |

Both are **at chance**. A slide being far from the in-distribution manifold tells you almost
nothing about whether the champion's MSI call is wrong. This is the key result: **an OOD-gated
abstention rule would not work** — the OOD score cannot find the errors it is supposed to route
to abstention. It reflects R2/R3: site and MSI are entangled, but the *magnitude* of the site
shift (OOD distance) does not track the *loss* of MSI signal.

**Implication for T3.** The T1 fairness problem (per-site FOR 0.00–0.288) is real, but T2 shows
the fix is **not** a continuous OOD-distance gate. Instead:

1. **Site membership**, not OOD distance, is the reliable conditioning variable — sites are
   perfectly identifiable (Mahalanobis 1.000) even though OOD distance doesn't track error. So
   the abstention/calibration should be **group-conditional on site** (each site gets its own
   conformal threshold, T3), not gated on a scalar OOD score.
2. A continuous OOD score can still *rank* how novel a truly-new site is at deployment (e.g., a
   7th site), so it is kept in `eval/ood.py` for T1/T3 as a deployment-time novelty monitor, but
   it is **not** used to gate abstention on the seen cohort.

**Files.** `argo_deepmsi/eval/ood.py`, `tests/test_ood.py`, `results/analysis/ood/ood_report.json`.
No board entry (analysis module). No new dependencies (scipy/sklearn already present).
