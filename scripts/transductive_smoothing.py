"""C6 / A2 — Histo-TransCLIP-style transductive refinement of A1 tile scores.

A1 produced per-tile (MSI - MSS) cosine deltas using the TITAN text encoder
against L2-normalised CONCH v1.5 patch features. A2 smooths those tile scores
through a per-slide kNN affinity graph on the same patch features and then
re-aggregates to the slide and patient level.

Intuition (Zanella et al. 2024 — Histo-TransCLIP): tiles that are visually
similar should agree on their VL score. Smoothing kills isolated high-confidence
mistakes (a stray necrosis tile that happens to land near "lymphocytic
infiltrate" in CLIP space) and amplifies regional consensus (a true MSI-H tumor
field where many neighbouring tiles all score positive). No training.

Algorithm (per slide):
    1. Load N×768 ``conch_v1.5_tiles`` features X.
    2. Compute per-tile raw VL score s_i = cos(x_i, MSI) - cos(x_i, MSS).
    3. Build kNN graph (k=10) on L2-normalised X (cosine kNN).
    4. Form row-stochastic transition matrix W (mean over neighbours).
    5. Iterate s ← (1-α)·s_0 + α·W·s for T iterations (label propagation).
    6. Aggregate smoothed s to slide score (mean, top-k mean, max).

Patient aggregation = max/√n (matches C5 best baseline).

Outputs (under ``results/analysis/c6_a2_transductive/``):
    slide_scores.csv
    patient_scores.csv
    metrics.json
    roc.png
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import zarr
from scipy import sparse
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from c6_a1_vl_zeroshot import (
    MSI_PROMPTS,
    MSS_PROMPTS,
    build_class_prototypes,
    load_titan_text_encoder,
    read_conch_v15_tiles,
    zarr_to_svs,
)


def smooth_scores(
    X: np.ndarray, s0: np.ndarray, k: int, alpha: float, n_iter: int
) -> np.ndarray:
    """Row-stochastic kNN graph label propagation in cosine space."""
    n = len(X)
    if n <= 2:
        return s0
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    k_eff = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric="cosine", algorithm="brute")
    nn.fit(Xn)
    _, idx = nn.kneighbors(Xn)
    idx = idx[:, 1:]  # drop self
    rows = np.repeat(np.arange(n), k_eff)
    cols = idx.reshape(-1)
    data = np.full(rows.shape, 1.0 / k_eff, dtype=np.float32)
    W = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    s = s0.astype(np.float32, copy=True)
    for _ in range(n_iter):
        s = (1.0 - alpha) * s0 + alpha * W.dot(s)
    return s


def score_slide_smoothed(
    tile_feats: np.ndarray,
    protos: torch.Tensor,
    k: int,
    alpha: float,
    n_iter: int,
    topk: int,
) -> dict:
    Xt = torch.from_numpy(tile_feats)
    Xt = F.normalize(Xt, dim=-1)
    sims = (Xt @ protos.T).numpy()  # (n, 2)
    s0 = sims[:, 0] - sims[:, 1]
    s = smooth_scores(tile_feats, s0, k=k, alpha=alpha, n_iter=n_iter)
    n = len(s)
    return {
        "n_tiles": int(n),
        "smooth_mean": float(s.mean()),
        "smooth_topk": float(np.sort(s)[-min(topk, n) :].mean()),
        "smooth_max": float(s.max()),
        # raw (no smoothing) reference
        "raw_topk": float(np.sort(s0)[-min(topk, n) :].mean()),
        "raw_max": float(s0.max()),
    }


def evaluate_per_site(slide_df: pd.DataFrame, score_col: str) -> dict:
    out = {}
    for s, sub in slide_df.groupby("site"):
        if sub["y"].nunique() == 2:
            out[s] = {
                "n": int(len(sub)),
                "prevalence": float(sub["y"].mean()),
                "auroc": float(roc_auc_score(sub["y"], sub[score_col])),
                "auprc": float(average_precision_score(sub["y"], sub[score_col])),
            }
        else:
            out[s] = {
                "n": int(len(sub)),
                "prevalence": float(sub["y"].mean()),
                "note": "single-class",
            }
    return out


def run(
    slide_table: Path,
    clinical: Path,
    outdir: Path,
    device: str,
    k: int,
    alpha: float,
    n_iter: int,
    topk: int,
    limit: int | None,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    st = pd.read_csv(slide_table)
    cl = pd.read_csv(clinical)[["PATIENT", "isMSIH"]]
    cl["y"] = (cl["isMSIH"] == "MSI-H").astype(int)
    st = st.merge(cl[["PATIENT", "y"]], on="PATIENT", how="inner")
    if limit is not None:
        st = st.head(limit)

    print("Loading TITAN text encoder...")
    model, tok = load_titan_text_encoder(device)
    protos, info = build_class_prototypes(model, tok, device)
    info.update({"k": k, "alpha": alpha, "n_iter": n_iter, "topk": topk})
    (outdir / "config.json").write_text(json.dumps(info, indent=2))

    rows = []
    for _, r in tqdm(st.iterrows(), total=len(st), desc="A2 transductive"):
        svs = r["FILENAME"]
        zp = zarr_to_svs(svs)
        X = read_conch_v15_tiles(zp)
        if X is None or X.shape[0] == 0:
            continue
        slide_id = Path(svs).stem
        scores = score_slide_smoothed(X, protos, k=k, alpha=alpha, n_iter=n_iter, topk=topk)
        rows.append(
            {
                "slide_id": slide_id,
                "patient_id": r["PATIENT"],
                "site": r["SITE"],
                "stain_location": r.get("stain_location"),
                "y": int(r["y"]),
                **scores,
            }
        )

    slide_df = pd.DataFrame(rows)
    slide_df.to_csv(outdir / "slide_scores.csv", index=False)

    score_cols = ["smooth_mean", "smooth_topk", "smooth_max", "raw_topk", "raw_max"]

    pat_rows = []
    for pid, g in slide_df.groupby("patient_id"):
        row = {
            "patient_id": pid,
            "y": int(g["y"].iloc[0]),
            "n_slides": int(len(g)),
            "site": g["site"].iloc[0],
        }
        for c in score_cols:
            sub = g.dropna(subset=[c])
            row[c] = (
                float(sub[c].max() / np.sqrt(len(sub))) if len(sub) else float("nan")
            )
        pat_rows.append(row)
    patient_df = pd.DataFrame(pat_rows)
    patient_df.to_csv(outdir / "patient_scores.csv", index=False)

    metrics = {
        "n_slides": int(len(slide_df)),
        "n_patients": int(len(patient_df)),
        "config": info,
        "variants": {},
    }
    for c in score_cols:
        sd = slide_df.dropna(subset=[c])
        pd_ = patient_df.dropna(subset=[c])
        s_auroc = (
            float(roc_auc_score(sd["y"], sd[c])) if sd["y"].nunique() == 2 else float("nan")
        )
        p_auroc = (
            float(roc_auc_score(pd_["y"], pd_[c]))
            if pd_["y"].nunique() == 2
            else float("nan")
        )
        metrics["variants"][c] = {
            "slide_auroc": s_auroc,
            "patient_auroc": p_auroc,
            "per_site": evaluate_per_site(sd, c),
        }
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    for c in score_cols:
        sub = slide_df.dropna(subset=[c])
        if sub["y"].nunique() != 2:
            continue
        fpr, tpr, _ = roc_curve(sub["y"], sub[c])
        auc = roc_auc_score(sub["y"], sub[c])
        ax.plot(fpr, tpr, label=f"{c} (AUROC={auc:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", c="gray", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"C6/A2 — transductive (k={k}, α={alpha}, T={n_iter})")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "roc.png", bbox_inches="tight")
    plt.close(fig)

    print("\n=== A2 results ===")
    for c in score_cols:
        v = metrics["variants"][c]
        print(
            f"  {c:<14s}  slide AUROC={v['slide_auroc']:.3f}  patient AUROC={v['patient_auroc']:.3f}"
        )
    print("\nPer-site (slide-level, smooth_max):")
    for s, m in metrics["variants"]["smooth_max"]["per_site"].items():
        if "auroc" in m:
            print(
                f"  {s:<22s}  n={m['n']:<4d} prev={m['prevalence']:.2f}  AUROC={m['auroc']:.3f}"
            )
        else:
            print(f"  {s:<22s}  n={m['n']:<4d} prev={m['prevalence']:.2f}  ({m.get('note','')})")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--slide-table", default="results/data/slide_table_pyramidal.csv")
    p.add_argument("--clinical", default="results/data/clinical_table.csv")
    p.add_argument("--outdir", default="results/analysis/c6_a2_transductive")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--n-iter", type=int, default=5)
    p.add_argument("--topk", type=int, default=64)
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args()
    run(
        Path(a.slide_table),
        Path(a.clinical),
        Path(a.outdir),
        a.device,
        a.k,
        a.alpha,
        a.n_iter,
        a.topk,
        a.limit,
    )


if __name__ == "__main__":
    main()
