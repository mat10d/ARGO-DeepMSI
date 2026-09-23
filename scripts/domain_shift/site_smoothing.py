"""Label-free site smoothing: Wagner under input moment matching, score normalisation,
and shift-invariant bag descriptors (see ``argo_deepmsi/eval/site_smoothing.py``).

    python scripts/domain_shift/site_smoothing.py wagner   # GPU: re-scores Wagner per variant
    python scripts/domain_shift/site_smoothing.py bags     # CPU: descriptors + nested LR

Outputs under ``results/analysis/domain_shift/site_smoothing/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from argo_deepmsi.eval.domain_shift import load_cohort
from argo_deepmsi.eval.metrics import aggregate_to_patient, stratified_patient_bootstrap
from argo_deepmsi.eval.screening import specificity_at_sensitivity
from argo_deepmsi.eval.site_smoothing import (
    MomentAccumulator,
    invariant_bag_descriptor,
    moment_match,
    site_score_normalise,
)
from argo_deepmsi.eval.validation import nested_grouped_oof

OUT = Path("results/analysis/domain_shift/site_smoothing")
META = Path("results/embeddings/ctranspath_mean/metadata.csv")
REF = "retrospective_msk"


def _slides() -> pd.DataFrame:
    meta = pd.read_csv(META)[["slide_id", "zarr_path"]]
    return load_cohort().merge(meta, on="slide_id")


def _tiles(zarr_path: str, model: str) -> np.ndarray:
    return np.asarray(ad.read_zarr(f"{zarr_path}/tables/{model}_tiles").X, dtype=np.float32)


def evaluate(patients: pd.DataFrame, label: str) -> dict:
    ci = stratified_patient_bootstrap(patients)
    op = specificity_at_sensitivity(patients["y"], patients["score"], 0.95)
    row = {"variant": label, "auroc": ci["estimate"], "ci_low": ci["ci_low"],
           "ci_high": ci["ci_high"], "spec_at_sens95": op.get("specificity")}
    for site, g in patients.groupby("site"):
        if g["y"].nunique() == 2:
            row[f"auroc_{site}"] = roc_auc_score(g["y"], g["score"])
    return row


def run_wagner(device: str) -> None:
    import torch

    from argo_deepmsi.models.wagner import load_wagner

    slides = _slides()
    acc = MomentAccumulator(768, full_cov=True)
    for _, r in slides.iterrows():
        acc.add(r["site"], _tiles(r["zarr_path"], "ctranspath"))
    ref = acc.moments(REF)
    pooled = MomentAccumulator(768, full_cov=True)
    for g in acc.n:
        pooled.n.setdefault("all", 0)
        pooled.n["all"] += acc.n[g]
        pooled.s["all"] = pooled.s.get("all", 0) + acc.s[g]
        pooled.ss["all"] = pooled.ss.get("all", 0) + acc.ss[g]
    ref_all = pooled.moments("all")

    variants = {
        "identity": None,
        "site-diag→MSK": ("diag", "site", ref, 1.0),
        "site-diag→MSK (α=0.5)": ("diag", "site", ref, 0.5),
        "site-diag→pooled": ("diag", "site", ref_all, 1.0),
        "site-CORAL→MSK": ("coral", "site", ref, 1.0),
        "bag-diag→MSK (instance norm)": ("diag", "bag", ref, 1.0),
    }
    model = load_wagner(device=device)
    rows = []
    with torch.no_grad():
        for _, r in slides.iterrows():
            X = _tiles(r["zarr_path"], "ctranspath")
            out = {"slide_id": r["slide_id"]}
            for name, spec in variants.items():
                if spec is None:
                    Y = X
                else:
                    mode, level, tgt, alpha = spec
                    if level == "site":
                        src = acc.moments(r["site"])
                    else:
                        src = (X.mean(0), X.var(0))
                        mode = "diag"
                    Y = moment_match(X, src, tgt, mode=mode, alpha=alpha)
                x = torch.from_numpy(Y).unsqueeze(0).to(device)
                out[name] = torch.sigmoid(model(x).squeeze()).item()
            rows.append(out)
            print(r["slide_id"], flush=True)
    scores = slides.merge(pd.DataFrame(rows), on="slide_id")
    OUT.mkdir(parents=True, exist_ok=True)
    scores.drop(columns="zarr_path").to_csv(OUT / "wagner_variant_slide_scores.csv", index=False)

    table = []
    for name in variants:
        pts = aggregate_to_patient(scores, name)
        table.append(evaluate(pts, f"input: {name}"))
        if name == "identity":
            for method in ("znorm", "rank"):
                table.append(evaluate(pts.assign(score=site_score_normalise(pts, method)),
                                      f"score: site {method}"))
    pd.DataFrame(table).to_csv(OUT / "wagner_site_smoothing.csv", index=False)
    print(pd.DataFrame(table).round(3).to_string())


def run_bags(encoders: list[str]) -> None:
    slides = _slides()
    OUT.mkdir(parents=True, exist_ok=True)
    table = []
    for enc in encoders:
        mean_rows, inv_rows = [], []
        for _, r in slides.iterrows():
            X = _tiles(r["zarr_path"], enc)
            mean_rows.append(X.mean(0))
            inv_rows.append(invariant_bag_descriptor(X))
        candidates = {"mean": np.stack(mean_rows), "invariant": np.stack(inv_rows),
                      "mean+invariant": np.hstack([np.stack(mean_rows), np.stack(inv_rows)])}
        for name, M in candidates.items():
            factories = {f"C={c}": (lambda c=c: make_pipeline(
                StandardScaler(), LogisticRegression(C=c, max_iter=5000,
                                                     class_weight="balanced")))
                         for c in (0.001, 0.01, 0.1)}
            oof, _ = nested_grouped_oof(slides.drop(columns="zarr_path"),
                                        {k: M for k in factories}, factories)
            pts = aggregate_to_patient(oof, "p_msih")
            row = evaluate(pts, f"{enc}: {name}")
            table.append(row)
            print(row, flush=True)
        pd.DataFrame(table).to_csv(OUT / "bag_descriptors_nested.csv", index=False)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("step", choices=["wagner", "bags"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--encoders", nargs="*", default=["mascaret", "phaet", "ctranspath"])
    a = p.parse_args()
    if a.step == "wagner":
        run_wagner(a.device)
    else:
        run_bags(a.encoders)


if __name__ == "__main__":
    main()
