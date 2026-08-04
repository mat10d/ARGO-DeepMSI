# V1 — Nested patient-grouped validation

The previous grid searched encoder, representation, classifier, correction, and aggregation
using the same OOF predictions reported as final performance. V1 introduces an outer/inner
patient-grouped evaluator (`argo_deepmsi.eval.validation.nested_grouped_oof`). Only inner-fold
patient AUROC selects the candidate; outer predictions are untouched. Scaling and every model
fit occur within their fold.

On the v2 primary cohort (217 patients), a pre-specified class-balanced LR race across four
mean-pooled encoders yields **patient AUROC 0.530, 95% bootstrap CI 0.429–0.636** and
spec@sens95 0.035. All 15 outer folds have zero patient overlap. No encoder dominates inner
selection, and OAUTHC is 0.510.

This scorer is the valid cohort-trained reference going forward. Historical scorer rows remain
useful as experiments, but configurations selected on their own final OOF predictions are not
confirmatory estimates. The unified leaderboard marks those rows `confirmatory_valid=false`;
they cannot set the frozen champion or no-regression floor even if their apparent AUROC is high.
