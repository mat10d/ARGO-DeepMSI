"""F1 — stacked fusion of the top-3 distinct-signal scorers.

Stacks the three best-performing, distinct-signal scorers on the clean cohort:

  - `calibrated_pool`     (0.713) — Wagner zero-shot CONCH, max/√n patient pooling
  - `slidefm_linearprobe` (0.646) — logistic probe on TITAN slide embeddings
  - `flex_bottleneck`     (0.623) — FLEX site-adversarial bottleneck on TITAN

(`wagner_zeroshot` is excluded as an exact duplicate of `calibrated_pool`; `simple_grid`
as a near-duplicate of the TITAN probe; the deep few-shot recipes S2/S3/S5 as
below-chance/uninformative — they would only add noise.)

Each base scorer's per-slide p_msih is aggregated to the patient with max/√n (identity for
the already-patient-level champion), aligned by patient_id, and a logistic-regression
meta-learner is fit over the three patient scores under patient-grouped CV (the base scores
are already OOF, so the stack is a clean stacked-generalization). Reported vs the champion and
the MSIntuit operating point.

Resolution: patient.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from ..eval.screening import screening_block
from .base import Scorer, ScoreColumn
from .registry import register

WAGNER_CSV = Path("results/scorers/wagner_zeroshot/slide_scores.csv")
BASE_SLIDE = {
    "slidefm_linearprobe": Path("results/scorers/slidefm_linearprobe/slide_scores.csv"),
    "flex_bottleneck": Path("results/scorers/flex_bottleneck/slide_scores.csv"),
}
SEED = 42
N_SPLITS = 5


def _agg_patient(slide_df: pd.DataFrame, col: str = "p_msih") -> pd.DataFrame:
    rows = []
    for pid, g in slide_df.groupby("patient_id"):
        rows.append({"patient_id": pid, "y": int(g["y"].iloc[0]), "site": g["site"].iloc[0],
                     "n_slides": int(len(g)), "score": float(g[col].max() / np.sqrt(len(g)))})
    return pd.DataFrame(rows)


def _base_patient_scores(clean_slide_ids: set[str] | None):
    """Return a wide frame patient_id,y,site,n_slides + one column per base scorer."""
    wag = pd.read_csv(WAGNER_CSV)
    if clean_slide_ids is not None:
        wag = wag[wag["slide_id"].isin(clean_slide_ids)]
    champ = _agg_patient(wag.dropna(subset=["p_msih"])).rename(columns={"score": "calibrated_pool"})
    wide = champ
    for name, path in BASE_SLIDE.items():
        if not path.exists():
            continue
        sd = pd.read_csv(path)
        if clean_slide_ids is not None:
            sd = sd[sd["slide_id"].isin(clean_slide_ids)]
        pat = _agg_patient(sd.dropna(subset=["p_msih"]))[["patient_id", "score"]].rename(columns={"score": name})
        wide = wide.merge(pat, on="patient_id", how="inner")
    return wide


def _feature_cols(wide: pd.DataFrame) -> list[str]:
    return [c for c in ("calibrated_pool", *BASE_SLIDE) if c in wide.columns]


def _stack_oof(wide: pd.DataFrame) -> np.ndarray:
    cols = _feature_cols(wide)
    X = wide[cols].to_numpy()
    y = wide["y"].to_numpy()
    groups = wide["patient_id"].to_numpy()
    oof = np.full(len(y), np.nan, dtype=np.float32)
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    pipe = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
    scaler = StandardScaler()
    for tr, te in cv.split(X, y, groups=groups):
        sc = clone(scaler).fit(X[tr])
        clf = clone(pipe).fit(sc.transform(X[tr]), y[tr])
        oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return oof


class FusionTop3(Scorer):
    name = "fusion_top3"
    description = (
        "Stacked logistic-regression fusion of the top-3 distinct-signal scorers "
        "(calibrated_pool + slidefm_linearprobe + flex_bottleneck) at patient level, "
        "patient-grouped CV over their OOF scores. Reported vs champion + MSIntuit."
    )
    needs_training_on_our_data = True
    resolution = "patient"
    score_columns = [
        ScoreColumn("p_msih", "stacked fusion patient p(MSI-H)", primary=True),
    ]

    def compute_batch(
        self,
        slide_table: pd.DataFrame,
        *,
        clean_slide_ids: set[str] | None = None,
        write_outputs: bool = True,
        retrain: bool = False,
        **_,
    ) -> pd.DataFrame:
        if not retrain and self.score_path is not None and self.score_path.exists():
            df = pd.read_csv(self.score_path)
            if clean_slide_ids is not None:
                wag = pd.read_csv(WAGNER_CSV)
                keep = set(wag[wag["slide_id"].isin(clean_slide_ids)]["patient_id"])
                df = df[df["patient_id"].isin(keep)].reset_index(drop=True)
            return df
        wide = _base_patient_scores(clean_slide_ids)
        if len(wide) == 0 or len(_feature_cols(wide)) < 2:
            raise RuntimeError(f"{self.name}: insufficient base scorers to fuse")
        wide = wide.assign(p_msih=_stack_oof(wide))
        self._last = {"wide": wide}
        pat = wide[["patient_id", "y", "site", "n_slides", "p_msih"]].dropna(subset=["p_msih"]).reset_index(drop=True)
        if write_outputs and self.score_path is not None:
            self.score_path.parent.mkdir(parents=True, exist_ok=True)
            pat.to_csv(self.score_path, index=False)
        return pat


register("fusion_top3", FusionTop3)


def _write_metrics(scorer: "FusionTop3", pat: pd.DataFrame, wide: pd.DataFrame) -> dict:
    import json

    from ..eval.metrics import _safe_auprc, _safe_auroc
    from ..eval.screening import MSINTUIT_TARGET

    y, s = pat["y"].to_numpy(), pat["p_msih"].to_numpy()
    block = screening_block(y, s, (0.90, 0.95, 0.96, 0.98))
    fusion_auroc = _safe_auroc(y, s)

    # component AUROCs on the same patients + champion baseline
    comp = {}
    for c in _feature_cols(wide):
        comp[c] = _safe_auroc(wide["y"].to_numpy(), wide[c].to_numpy())
    champ_auroc = comp.get("calibrated_pool", float("nan"))

    by_site = {}
    for st, sub in pat.groupby("site"):
        yy, ss = sub["y"].to_numpy(), sub["p_msih"].to_numpy()
        by_site[str(st)] = {"n": int(len(sub)), "prevalence": float(sub["y"].mean()),
                            "auroc": _safe_auroc(yy, ss),
                            "spec_at_sens95": screening_block(yy, ss, (0.95,))["spec_at_sens95"]
                            if sub["y"].nunique() == 2 else float("nan")}
    pat_b = pat.assign(_b=pd.cut(pat["n_slides"], [0, 1, 2, 4, np.inf], labels=["1", "2", "3-4", "5+"]))
    by_bagsize = {str(b): {"n": int(len(g)), "auroc": _safe_auroc(g["y"].to_numpy(), g["p_msih"].to_numpy())}
                  for b, g in pat_b.groupby("_b", observed=True)}

    metrics = {
        "scorer": scorer.name, "components": _feature_cols(wide),
        "component_auroc": comp, "champion_auroc": champ_auroc,
        "fusion_minus_champion_auroc": (fusion_auroc - champ_auroc) if not np.isnan(champ_auroc) else None,
        "n_patients": int(len(pat)), "prevalence_patient": float(y.mean()),
        "auroc": fusion_auroc, "auprc": _safe_auprc(y, s),
        "sensitivity": block["op_sens95"]["sensitivity"],
        "spec_at_sens90": block["spec_at_sens90"], "spec_at_sens95": block["spec_at_sens95"],
        "spec_at_sens96": block["spec_at_sens96"], "npv_at_sens95": block["npv_at_sens95"],
        "operating_threshold": block["op_sens95"]["threshold"],
        "msintuit_target": MSINTUIT_TARGET,
        "by_site": by_site, "by_bagsize": by_bagsize, "screening_clean": block,
    }
    out = FusionTop3().score_path.parent / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2))
    return metrics
