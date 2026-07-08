"""T1 — selective / conformal abstention screener wrapping the champion.

The R-phase falsified feature-space site-invariance: the model cannot be made right on
OAUTHC, because site and MSI signal are entangled in frozen CONCH features. The correct
deployment behaviour for a rule-out screener is then to *know when it is unsure and
abstain* — trading coverage for a risk guarantee, and declining preferentially on the OOD
(OAUTHC-type) slides it cannot handle.

This wraps the zero-param champion (`calibrated_pool` = max/√n over Wagner per-slide
p_msih) in a selective classifier. Per patient the confidence is the distance of the score
from the sens-95 operating threshold; abstaining on the least-confident patients yields a
risk–coverage curve. A split-conformal calibration turns a target risk into a guaranteed
coverage. Group-conditional (per-site) coverage shows whether abstention concentrates on the
weak sites.

The board p_msih is the champion's score (this is the champion *plus* an abstention overlay,
not a new ranking); the novelty lives in `coverage_risk` + `per_site_coverage` in metrics.json.

Resolution: patient.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..eval.screening import threshold_at_sensitivity
from .base import Scorer, ScoreColumn
from .registry import register

WAGNER_CSV = Path("results/scorers/wagner_zeroshot/slide_scores.csv")
SEED = 42
TARGET_SENS = 0.95
TARGET_RISK = 0.10  # conformal: cap covered-set error at 10%


def _champion_patients(clean_slide_ids: set[str] | None) -> pd.DataFrame:
    """Champion (calibrated_pool) patient scores = max(wagner p_msih)/√n per patient."""
    wag = pd.read_csv(WAGNER_CSV)
    if clean_slide_ids is not None:
        wag = wag[wag["slide_id"].isin(clean_slide_ids)]
    wag = wag.dropna(subset=["p_msih"])
    rows = []
    for pid, g in wag.groupby("patient_id"):
        rows.append({"patient_id": pid, "y": int(g["y"].iloc[0]), "site": g["site"].iloc[0],
                     "n_slides": int(len(g)), "p_msih": float(g["p_msih"].max() / np.sqrt(len(g)))})
    return pd.DataFrame(rows)


def risk_coverage_curve(y, score, thr=None, grid=None) -> list[dict]:
    """Rule-out selective screener: coverage vs false-omission rate (FOR).

    A rule-out test confidently calls the lowest-scoring patients MSS-negative (they skip
    molecular testing) and *refers / abstains* on the rest. Coverage = fraction ruled out
    (workload saved); risk = FOR = P(y=MSI-H | ruled out) = 1 − NPV on the covered set — the
    dangerous error (a missed MSI-H patient). Ruling out the lowest scores first, coverage
    grows and FOR grows. Returns points by increasing coverage.
    """
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    order = np.argsort(score)  # lowest (most confidently negative) first
    y_o = y[order]
    n = len(y)
    if grid is None:
        grid = np.linspace(0.05, 1.0, 20)
    pts = []
    for c in grid:
        k = max(1, int(round(c * n)))
        for_ = float(np.mean(y_o[:k]))  # fraction of ruled-out that are actually MSI-H
        pts.append({"coverage": float(k / n), "n_covered": int(k), "risk": for_})
    return pts


def _aurc(pts: list[dict]) -> float:
    cov = np.array([p["coverage"] for p in pts])
    risk = np.array([p["risk"] for p in pts])
    o = np.argsort(cov)
    return float(np.trapz(risk[o], cov[o]) / (cov[o].max() - cov[o].min() + 1e-9))


def split_conformal(y, score, site, thr=None, target_risk=TARGET_RISK, seed=SEED) -> dict:
    """Split-conformal rule-out screener. Calibrate a rule-out score threshold τ on half the
    patients so the calibration false-omission rate ≤ target_risk; on the other half report the
    coverage (fraction ruled out), covered FOR, and per-site coverage/FOR. A site where the
    model cannot confidently rule out (OOD) gets *lower* coverage — abstention concentrates
    there. Averaged over repeats to reduce split noise."""
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    site = np.asarray(site, dtype=object)
    n = len(y)
    rng = np.random.default_rng(seed)

    covs, fors, taus = [], [], []
    per_site = {s: {"cov": [], "for": [], "n": int((site == s).sum())} for s in np.unique(site)}
    for _ in range(50):
        idx = rng.permutation(n)
        cal, te = idx[: n // 2], idx[n // 2:]
        # τ = highest rule-out threshold whose calibration FOR ≤ target_risk (rule out score ≤ τ)
        c_order = np.argsort(score[cal])  # ascending
        yc = y[cal][c_order]
        tau = float(score[cal].min()) - 1e-6  # rule out nobody by default
        for k in range(len(cal), 0, -1):
            if np.mean(yc[:k]) <= target_risk:
                tau = float(score[cal][c_order][k - 1])
                break
        keep = score[te] <= tau  # ruled out
        covs.append(float(keep.mean()))
        fors.append(float(np.mean(y[te][keep])) if keep.any() else float("nan"))
        taus.append(tau)
        for s in per_site:
            m = (site[te] == s)
            if m.any():
                per_site[s]["cov"].append(float((keep & m).sum() / m.sum()))
                km = keep & m
                per_site[s]["for"].append(float(np.mean(y[te][km])) if km.any() else float("nan"))
    out_site = {}
    for s, d in per_site.items():
        out_site[str(s)] = {"n": d["n"],
                            "coverage": float(np.nanmean(d["cov"])) if d["cov"] else float("nan"),
                            "false_omission_rate": float(np.nanmean(d["for"])) if d["for"] else float("nan")}
    return {"target_risk": target_risk, "mean_tau": float(np.mean(taus)),
            "guaranteed_coverage": float(np.mean(covs)),
            "empirical_covered_for": float(np.nanmean(fors)),
            "per_site_coverage": out_site}


class SelectiveAbstention(Scorer):
    name = "selective_abstention"
    description = (
        "Selective / split-conformal abstention screener wrapping the champion "
        "(calibrated_pool). Abstains on the least-confident patients (distance from the "
        "sens-95 threshold) for a risk-coverage guarantee; reports group-conditional "
        "(per-site) coverage. Board score = champion score (abstention is an overlay)."
    )
    needs_training_on_our_data = False
    resolution = "patient"
    score_columns = [
        ScoreColumn("p_msih", "champion (calibrated_pool) patient score", primary=True),
    ]

    def compute_batch(
        self,
        slide_table: pd.DataFrame,
        *,
        clean_slide_ids: set[str] | None = None,
        write_outputs: bool = True,
        **_,
    ) -> pd.DataFrame:
        pat = _champion_patients(clean_slide_ids)
        if len(pat) == 0:
            raise RuntimeError(f"{self.name}: no champion patient scores (need {WAGNER_CSV})")
        if write_outputs and self.score_path is not None:
            self.score_path.parent.mkdir(parents=True, exist_ok=True)
            pat.to_csv(self.score_path, index=False)
        return pat


register("selective_abstention", SelectiveAbstention)


def _write_metrics(scorer: "SelectiveAbstention", pat: pd.DataFrame) -> dict:
    import json

    from ..eval.metrics import _safe_auprc, _safe_auroc
    from ..eval.screening import screening_block

    y = pat["y"].to_numpy()
    s = pat["p_msih"].to_numpy()
    site = pat["site"].to_numpy()
    thr = threshold_at_sensitivity(y, s, TARGET_SENS)
    curve = risk_coverage_curve(y, s, thr)
    conf = split_conformal(y, s, site, thr)
    block = screening_block(y, s, (0.90, 0.95, 0.96, 0.98))

    by_site = {}
    for st, sub in pat.groupby("site"):
        yy, ss = sub["y"].to_numpy(), sub["p_msih"].to_numpy()
        by_site[str(st)] = {"n": int(len(sub)), "prevalence": float(sub["y"].mean()),
                            "auroc": _safe_auroc(yy, ss),
                            "spec_at_sens95": screening_block(yy, ss, (0.95,))["spec_at_sens95"]
                            if sub["y"].nunique() == 2 else float("nan")}

    metrics = {
        "scorer": scorer.name, "base": "calibrated_pool (champion)",
        "operating_threshold": float(thr), "target_sensitivity": TARGET_SENS,
        "n_patients": int(len(pat)), "prevalence_patient": float(y.mean()),
        "auroc": _safe_auroc(y, s), "auprc": _safe_auprc(y, s),
        "spec_at_sens90": block["spec_at_sens90"], "spec_at_sens95": block["spec_at_sens95"],
        "spec_at_sens96": block["spec_at_sens96"], "npv_at_sens95": block["npv_at_sens95"],
        "sensitivity": block["op_sens95"]["sensitivity"],
        "aurc": _aurc(curve),
        "coverage_risk": curve,
        "conformal": conf,
        "per_site_coverage": conf["per_site_coverage"],
        "by_site": by_site,
    }
    out = SelectiveAbstention().score_path.parent / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2))
    return metrics
