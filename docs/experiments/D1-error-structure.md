# D1 — Error-structure & complementarity diagnostic

## Question (the "better top layer" gate)
Every prior top-layer attempt (S5 Deep-Sets, slide-attention, F1 fusion) was built blind and
failed. Before designing another, test: do the scorers fail on DIFFERENT patients (=> a
case-conditioned readout has headroom) or the SAME patients (=> representational ceiling, no top
layer helps)? Cohort: 180 clean patients, 41 MSI-H, 15 usable scorers, max/sqrt(n) patient agg.

## Result 1 — errors are largely DISJOINT (complementarity is real)
Champion (wagner_zeroshot) errs on 34% of patients (rank>0.5 call). Pairwise, when champion errs
a partner usually does not:
| pair | both err | either err |
|---|---|---|
| champion & batch_corrected | 0.133 | 0.622 |
| champion & transductive    | 0.139 | 0.661 |
| champion & harmony         | 0.161 | 0.606 |
| champion & clam_tilemil    | 0.178 | 0.589 |
Both-err ~0.13-0.18 vs either-err ~0.59-0.66 => the errors are mostly non-overlapping. If they
coincided, no top layer could help. They don't. Direct examples on champion's worst OAUTHC calls:
P_0076 (MSI-H) champion 0.03 WRONG, harmony 0.61 + clam 0.68 RIGHT; P_0170 (MSS) champion 0.89
WRONG, harmony 0.12 + batch_corr 0.05 RIGHT.

## Result 2 — per-site specialization (different scorers win different sites)
wagner_zeroshot best overall 0.731 & on good sites (incl retro-OAU 0.822), but OAUTHC only 0.614.
harmony/simple_grid best on OAUTHC 0.683. transductive 1.000 on LASUTH, protonet 0.923 on LUTH 
(where wagner is 0.923 too) — i.e. small-site winners differ from the overall champion. No single 
scorer dominates all sites; OAUTHC's best (harmony 0.683) is not the overall champion.

## Result 3 — oracle ceiling is an UPPER BOUND, not achievable (stated honestly)
Label-aware oracle (per patient pick the scorer that happened to be right): all-15 -> AUROC 1.000
(+0.269 over champion); champ+harmony+batch_corr -> 0.959 (+0.228). THIS IS NOT A RESULT — it uses
the true label to route. It only proves the complementarity ceiling is high. The achievable gain
is a FRACTION of this gap and must be measured on held-out patients.

## Result 4 — the signal is PATIENT-conditional, not SITE-conditional
On OAUTHC, harmony moves the score correctly vs champion on only 32/63 patients (0.51 — a coin
flip). So a fixed site-router ("trust harmony on OAUTHC") barely helps. The complementarity is at
the PATIENT level, not the site level => a naive site-switch won't capture it.

## Result 5 — what champion failure tracks
Champion error vs features (Pearson r): tumor_fraction +0.159 (weak), n_slides -0.03, n_tiles
-0.03, artifact_fraction -0.01. By site, mean error: OAUTHC 0.475 (worst), LUTH 0.326 (best).
Failure is only weakly explained by bag size / QC features; it is mostly site + patient-idiosyncratic.

## VERDICT
A better top layer IS justified — disjoint-error headroom is real (both-err 0.13-0.18 << either-err
0.6). BUT it must be a LEARNED, CASE-CONDITIONED gate keyed on patient-level features + the score
vector (agreement pattern), NOT a fixed site-router (Result 4 kills that). Validate on held-out
patients; expect a fraction of the 0.27 oracle gap, not the gap. This is the B-phase method:
learned per-patient scorer-gating / mixture-of-experts with honest held-out CV.

## Caveat carried forward
Achievable gain unknown until a real gate is CV-tested. Do NOT report the oracle number as a result.
