# Q4 — freeze the manifest & re-race the board on the clean cohort

**Method.** Q1–Q3 built `cohort_clean.csv` through three QC layers (eCRF exclusion →
GrandQC artifact flag → CTransPath tumor-fraction drop, floor 0.01), landing
`in_clean_set` at **428 slides / 181 patients** (prevalence 0.236). But the leaderboard
was still raced against the eCRF-only pathologist list (`problem_slides.csv`, 198
patients) — it never saw the artifact/tumor drops. Q4 closes that gap and freezes the
cohort:

1. **Wire `in_clean_set` into the board re-race.** Added `load_clean_exclusion()`
   (`eval/cohort.py`) — the complement of the clean set (380 slides) — and a `--clean-csv`
   flag to `eval/qc_comparison`. The board now excludes every non-clean slide, not just the
   eCRF layer.
2. **Regenerate** the leaderboard on the clean cohort
   (`qc_comparison --clean-csv results/data/cohort_clean.csv`, CPU job 10302514, ~18 min).
3. **Freeze** `cohort_manifest.json` (`cohort.py --freeze`): bump `cohort_version` to
   `v1-ecrf+artifact+tumor`, record the final layer counts + per-site clean composition,
   and set `no_regression_floor` = best clean patient AUROC read from the fresh board.
   `ralph/no_regression.py` already reads this key, so the gate now ratchets from the true
   clean-cohort champion.

**Headline vs MSIntuit** (target sens 0.96–0.98 @ spec 0.46–0.47, κ 0.82). Champion
`calibrated_pool` (max/√n, zero params) on the frozen clean cohort (428 slides / 181
patients): patient AUROC **0.7127**, spec@sens90 **0.193**, spec@sens95 **0.179**,
spec@sens96 **0.079**, NPV@sens95 **0.926**. Racing on the true clean set (not the eCRF
198) confirms the Q3 purity gain sticks (spec@sens95 0.154 → 0.179) and leaves the floor
unchanged at 0.7127. The spec@sens gap to MSIntuit is untouched — Q4 is a bookkeeping
freeze, not a modeling step; the S-phase scorers are where that gap must close.

**Leaderboard on the clean cohort (patient AUROC):**

| scorer | AUROC | spec@sens95 | NPV@sens95 | trained on ours |
|--------|------:|------------:|-----------:|:---------------:|
| calibrated_pool | **0.713** | 0.179 | 0.926 | no |
| wagner_zeroshot | 0.713 | 0.179 | 0.926 | no |
| slide_attention_mil† | 0.669 | 0.038 | 0.750 | yes |
| simple_grid | 0.646 | 0.229 | 0.941 | yes |
| score_fusion | 0.608 | 0.129 | 0.900 | yes |
| vl_text_cosine | 0.563 | 0.064 | 0.818 | no |
| transductive_smoothing | 0.551 | 0.093 | 0.867 | no |
| nuclear_morphology | 0.542 | 0.043 | 0.750 | yes |

†`slide_attention_mil` still races on 198 patients — its `compute_batch` does not honor
`clean_slide_ids`. It sits well below champion, so it does not affect the no-regression
floor. Flagged as a follow-up scorer fix (out of Q4 scope; editing it here would touch
another scorer's file).

**Champion per-site AUROC (clean board):**

| site | n_pat | prevalence | AUROC |
|------|------:|-----------:|------:|
| LASUTH | 9 | 0.33 | 0.889 |
| LUTH | 15 | 0.13 | 1.000 |
| OAUTHC | 64 | 0.20 | **0.579** |
| UITH | 10 | 0.40 | 0.750 |
| retrospective_msk | 60 | 0.23 | 0.793 |
| retrospective_oau | 23 | 0.22 | 0.800 |

OAUTHC remains the weak site (0.579) — the diagnosed site-robustness problem the R-phase
targets. Every other site is ≥ 0.75.

**Clean cohort by bag size (patients per slide count):** 1 slide → 77 pat, 2 → 35,
3–4 → 56, 5+ → 13. Single-slide patients dominate (43 %), which is why the bag-size
inflation fix (S5) and cardinality-calibrated aggregation matter.

**Verdict.** Manifest frozen at `v1-ecrf+artifact+tumor` (428 slides / 181 patients,
prevalence 0.236). No-regression floor set to **0.7127** (calibrated_pool), read live by
the gate. Board re-raced on the true clean set; champion held. The Q-phase (clean
tumor-only dataset foundation) is complete — the S-phase low-N scorers build on this
frozen cohort.
