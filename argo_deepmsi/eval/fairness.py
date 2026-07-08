"""T3 — fairness evaluation + the group-conditional fairness gate.

T1 showed a *global* rule-out threshold is unfair: per-site false-omission rate ranged
0.00–0.288, so the weak sites are ruled out unsafely. T2 showed the fix must condition on
*site* (perfectly identifiable) rather than a scalar OOD distance. T3 implements that fix and
evaluates it:

- **per-site metrics** — AUROC, spec@sens95, and the false-omission rate (FOR) at the global
  rule-out threshold; the raw disparity.
- **group-conditional conformal** — each site gets its *own* rule-out threshold so its FOR ≤
  target. This equalizes *safety* (every site meets the FOR target) and exposes the true cost:
  the *coverage* (fraction safely ruled out) each site can afford at equal safety.
- **coverage-adjusted gap** — the disparity in benefit (rule-out coverage) across sites once
  safety is equalized; the honest fairness metric for a rule-out screener.
- **fairness gate** — a pass/fail verdict for the loop stop bar: FAIL if any adequately-sized
  site cannot be served safely (safe coverage ≈ 0 at the FOR target) — a disqualifying outlier.

Frozen inputs (champion patient scores). No training, no external data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .screening import screening_block

TARGET_FOR = 0.10
TARGET_SENS = 0.95
MIN_SITE_N = 15  # below this a site is too small to calibrate / gate on
DISQUALIFY_COVERAGE = 0.05  # a serviceable site should safely rule out > 5% at the FOR target
FOR_SLACK = 0.05  # out-of-sample FOR may exceed target by at most this (else calibration failed)


def _for_at(y, score, tau) -> float:
    keep = score <= tau
    return float(np.mean(y[keep])) if keep.any() else float("nan")


def _max_safe_tau(y, score, target_for) -> float:
    """Highest rule-out threshold τ (rule out score ≤ τ) keeping FOR ≤ target_for."""
    order = np.argsort(score)
    y_o, s_o = y[order], score[order]
    tau = float(score.min()) - 1e-6
    for k in range(len(y), 0, -1):
        if np.mean(y_o[:k]) <= target_for:
            tau = float(s_o[k - 1])
            break
    return tau


def per_site_metrics(pat: pd.DataFrame, global_tau: float) -> dict:
    out = {}
    for site, sub in pat.groupby("site"):
        y = sub["y"].to_numpy()
        s = sub["p_msih"].to_numpy()
        blk = screening_block(y, s, (0.95,)) if sub["y"].nunique() == 2 else {"spec_at_sens95": float("nan")}
        out[str(site)] = {
            "n": int(len(sub)), "prevalence": float(y.mean()),
            "auroc": float(roc_auc_score(y, s)) if sub["y"].nunique() == 2 else float("nan"),
            "spec_at_sens95": float(blk["spec_at_sens95"]),
            "for_at_global_tau": _for_at(y, s, global_tau),
            "coverage_at_global_tau": float(np.mean(s <= global_tau)),
        }
    return out


def group_conditional_conformal(pat: pd.DataFrame, target_for: float = TARGET_FOR,
                                seed: int = 42) -> dict:
    """Per-site split-conformal rule-out: each site's own τ so its FOR ≤ target. Report the
    coverage each site can safely afford (averaged over calibration splits)."""
    rng = np.random.default_rng(seed)
    out = {}
    for site, sub in pat.groupby("site"):
        y = sub["y"].to_numpy()
        s = sub["p_msih"].to_numpy()
        n = len(sub)
        if n < MIN_SITE_N or len(np.unique(y)) < 2:
            out[str(site)] = {"n": int(n), "coverage": float("nan"), "for": float("nan"),
                              "note": "too small to calibrate"}
            continue
        covs, fors = [], []
        for _ in range(50):
            idx = rng.permutation(n)
            cal, te = idx[: n // 2], idx[n // 2:]
            tau = _max_safe_tau(y[cal], s[cal], target_for)
            keep = s[te] <= tau
            covs.append(float(keep.mean()))
            fors.append(float(np.mean(y[te][keep])) if keep.any() else float("nan"))
        out[str(site)] = {"n": int(n), "coverage": float(np.nanmean(covs)),
                          "for": float(np.nanmean(fors))}
    return out


def fairness_report(pat: pd.DataFrame, target_for: float = TARGET_FOR) -> dict:
    y = pat["y"].to_numpy()
    s = pat["p_msih"].to_numpy()
    global_tau = _max_safe_tau(y, s, target_for)
    per_site = per_site_metrics(pat, global_tau)
    gcc = group_conditional_conformal(pat, target_for)

    aurocs = [d["auroc"] for d in per_site.values() if not np.isnan(d["auroc"])]
    global_fors = [d["for_at_global_tau"] for d in per_site.values() if not np.isnan(d["for_at_global_tau"])]
    servable = {k: v for k, v in gcc.items() if v["n"] >= MIN_SITE_N and not np.isnan(v["coverage"])}
    covs = [v["coverage"] for v in servable.values()]

    # fairness gate: a serviceable site is a disqualifying outlier if, even with its OWN
    # conformal threshold, it either cannot rule out a useful fraction (coverage too low) or
    # cannot be made safe (out-of-sample FOR blows past the target — no signal to calibrate on).
    outliers = sorted(
        k for k, v in servable.items()
        if v["coverage"] < DISQUALIFY_COVERAGE
        or (not np.isnan(v["for"]) and v["for"] > target_for + FOR_SLACK)
    )
    gate_pass = len(outliers) == 0 and len(servable) >= 2

    return {
        "target_for": target_for, "global_tau": float(global_tau),
        "n_patients": int(len(pat)),
        "per_site": per_site,
        "group_conditional_conformal": gcc,
        "auroc_gap_max_minus_min": float(max(aurocs) - min(aurocs)) if aurocs else float("nan"),
        "worst_site_auroc": float(min(aurocs)) if aurocs else float("nan"),
        "global_tau_for_disparity": {
            "min": float(min(global_fors)) if global_fors else float("nan"),
            "max": float(max(global_fors)) if global_fors else float("nan"),
        },
        "coverage_adjusted_gap": float(max(covs) - min(covs)) if covs else float("nan"),
        "min_safe_coverage_servable": float(min(covs)) if covs else float("nan"),
        "fairness_gate": {
            "pass": bool(gate_pass),
            "disqualifying_outliers": outliers,
            "min_site_n": MIN_SITE_N,
            "disqualify_coverage_below": DISQUALIFY_COVERAGE,
            "for_slack": FOR_SLACK,
            "rationale": (
                "PASS: every serviceable site (n>=%d) meets the FOR target out-of-sample and "
                "still rules out >%.0f%% of patients." % (MIN_SITE_N, DISQUALIFY_COVERAGE * 100)
                if gate_pass else
                "FAIL: site(s) %s cannot be safely served under group-conditional conformal — "
                "either safe coverage < %.0f%% or out-of-sample FOR exceeds target+%.2f (the "
                "site has no MSI signal to calibrate on). A disqualifying per-site outlier."
                % (outliers, DISQUALIFY_COVERAGE * 100, FOR_SLACK)
            ),
        },
    }


def main() -> None:
    import argparse
    import json

    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--patient-scores",
                   default="results/scorers/selective_abstention/patient_scores.csv", type=Path)
    p.add_argument("--outdir", default="results/analysis/fairness", type=Path)
    p.add_argument("--target-for", default=TARGET_FOR, type=float)
    a = p.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)

    pat = pd.read_csv(a.patient_scores)
    rep = fairness_report(pat, target_for=a.target_for)
    (a.outdir / "fairness_report.json").write_text(json.dumps(rep, indent=2))

    print(f"Fairness gate: {'PASS' if rep['fairness_gate']['pass'] else 'FAIL'}")
    print(f"  {rep['fairness_gate']['rationale']}")
    print(f"AUROC gap (max-min) {rep['auroc_gap_max_minus_min']:.3f}; "
          f"coverage-adjusted gap {rep['coverage_adjusted_gap']:.3f}")
    print("group-conditional safe rule-out coverage per site (FOR-equalized):")
    for site, d in sorted(rep["group_conditional_conformal"].items(),
                          key=lambda kv: (np.isnan(kv[1]["coverage"]), kv[1]["coverage"])):
        cov = d["coverage"]
        print(f"  {site:<20s} coverage {cov:.3f}  FOR {d['for']:.3f}  (n={d['n']})"
              if not np.isnan(cov) else f"  {site:<20s} {d.get('note','')}  (n={d['n']})")


if __name__ == "__main__":
    main()
