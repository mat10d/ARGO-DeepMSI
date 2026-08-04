"""A3 — Wagner et al. (HistoBistro, Cancer Cell 2023) zero-shot MSI evaluation.

Runs the published transformer MSI classifier (trained on ~13k Western CRC
slides using CTransPath features) on our 803-slide Nigerian cohort. No
fine-tuning — purely measures Western → Africa generalization.

Model source:
    https://github.com/peng-lab/HistoBistro/tree/main/CancerCellCRCTransformer
    trained_models/MSI_high_CRC_model.pth (pytorch-lightning state_dict,
    architecture = Transformer(input_dim=768, dim=512, depth=2, heads=8,
    mlp_dim=512, pool='cls')).

The canonical project-local implementation lives in
``argo_deepmsi.models.wagner``. The external checkpoint is staged and
checksum-verified under ``artifacts/checkpoints/wagner``. Runtime code never
imports or reads the archived ``old/`` tree.

Inputs:
    - Per-slide tile features from <slide>.zarr/tables/ctranspath_tiles.
      Shape (n_tiles, 768). LazySlide-extracted.
    - Clinical table for ground-truth MSI labels.
    - Pretrained weights at artifacts/checkpoints/wagner/MSI_high_CRC_model.pth.

Output:
    results/analysis/wagner_zeroshot/
        slide_scores.csv       — slide_id, patient_id, site, stain_location, y, p_msih
        patient_scores.csv     — patient-level mean probability + label
        metrics.json           — slide/patient AUROC, AUPRC, per-site AUROC
        roc.png                — slide and patient ROC curves
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from tqdm import tqdm
from wsidata import open_wsi

from argo_deepmsi.models.wagner import DEFAULT_WAGNER_CHECKPOINT, load_wagner

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def read_ctranspath_tiles(svs_path: Path) -> np.ndarray | None:
    zarr_path = svs_path.with_suffix(".zarr")
    if not zarr_path.exists():
        return None
    try:
        wsi = open_wsi(str(svs_path), store=str(svs_path.parent), attach_thumbnail=False)
    except Exception as e:
        print(f"  skip (open failed): {svs_path.name}: {e}")
        return None
    if "ctranspath_tiles" not in wsi.tables:
        return None
    X = np.asarray(wsi.tables["ctranspath_tiles"].X, dtype=np.float32)
    return X


def run(slide_table: Path, clinical: Path, weights: Path, outdir: Path, device: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    st = pd.read_csv(slide_table)
    cl = pd.read_csv(clinical)[["PATIENT", "isMSIH"]]
    cl["y"] = (cl["isMSIH"] == "MSI-H").astype(int)
    st = st.merge(cl[["PATIENT", "y"]], on="PATIENT", how="inner")

    model = load_wagner(weights, device=device)

    rows = []
    with torch.no_grad():
        for _, r in tqdm(st.iterrows(), total=len(st), desc="wagner"):
            svs = Path(r["FILENAME"])
            X = read_ctranspath_tiles(svs)
            if X is None or X.shape[0] == 0:
                continue
            try:
                x = torch.from_numpy(X).unsqueeze(0).to(device)
                logit = model(x).squeeze()
                prob = torch.sigmoid(logit).item()
            except torch.cuda.OutOfMemoryError:
                # fallback to CPU for pathologically large slides
                torch.cuda.empty_cache()
                x = torch.from_numpy(X).unsqueeze(0)
                model_cpu = model.to("cpu")
                logit = model_cpu(x).squeeze()
                prob = torch.sigmoid(logit).item()
                model.to(device)
                print(f"  OOM on {svs.name} (n_tiles={X.shape[0]}); used CPU fallback")
            rows.append(
                {
                    "slide_id": svs.stem,
                    "patient_id": r["PATIENT"],
                    "site": r["SITE"],
                    "stain_location": r.get("stain_location"),
                    "n_tiles": int(X.shape[0]),
                    "y": int(r["y"]),
                    "p_msih": float(prob),
                }
            )

    slide_df = pd.DataFrame(rows)
    slide_df.to_csv(outdir / "slide_scores.csv", index=False)

    # Patient-level mean
    patient_df = (
        slide_df.groupby("patient_id")
        .agg(p_msih=("p_msih", "mean"), y=("y", "first"), n_slides=("slide_id", "count"))
        .reset_index()
    )
    patient_df.to_csv(outdir / "patient_scores.csv", index=False)

    # Metrics
    slide_auroc = float(roc_auc_score(slide_df["y"], slide_df["p_msih"])) if slide_df["y"].nunique() == 2 else float("nan")
    slide_auprc = float(average_precision_score(slide_df["y"], slide_df["p_msih"])) if slide_df["y"].nunique() == 2 else float("nan")
    pat_auroc = float(roc_auc_score(patient_df["y"], patient_df["p_msih"])) if patient_df["y"].nunique() == 2 else float("nan")
    pat_auprc = float(average_precision_score(patient_df["y"], patient_df["p_msih"])) if patient_df["y"].nunique() == 2 else float("nan")

    per_site = {}
    for s, sub in slide_df.groupby("site"):
        if sub["y"].nunique() == 2:
            per_site[s] = {
                "n": int(len(sub)),
                "prevalence": float(sub["y"].mean()),
                "auroc": float(roc_auc_score(sub["y"], sub["p_msih"])),
                "auprc": float(average_precision_score(sub["y"], sub["p_msih"])),
            }
        else:
            per_site[s] = {"n": int(len(sub)), "prevalence": float(sub["y"].mean()), "note": "single-class"}

    metrics = {
        "n_slides": int(len(slide_df)),
        "n_patients": int(len(patient_df)),
        "prevalence_slide": float(slide_df["y"].mean()),
        "prevalence_patient": float(patient_df["y"].mean()),
        "slide_auroc": slide_auroc,
        "slide_auprc": slide_auprc,
        "patient_auroc": pat_auroc,
        "patient_auprc": pat_auprc,
        "per_site": per_site,
    }
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ROC plot
    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    for label, sub in [("slide", slide_df), ("patient", patient_df)]:
        if sub["y"].nunique() == 2:
            fpr, tpr, _ = roc_curve(sub["y"], sub["p_msih"])
            auc = roc_auc_score(sub["y"], sub["p_msih"])
            ax.plot(fpr, tpr, label=f"{label}-level (AUROC={auc:.3f}, N={len(sub)})")
    ax.plot([0, 1], [0, 1], ls="--", c="gray", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("A3 — Wagner zero-shot on Nigerian cohort")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(outdir / "roc.png", bbox_inches="tight")
    plt.close(fig)

    print(f"\nSlide-level   AUROC={slide_auroc:.3f}  AUPRC={slide_auprc:.3f}  (n={metrics['n_slides']}, prev={metrics['prevalence_slide']:.2f})")
    print(f"Patient-level AUROC={pat_auroc:.3f}  AUPRC={pat_auprc:.3f}  (n={metrics['n_patients']}, prev={metrics['prevalence_patient']:.2f})")
    print("\nPer-site slide AUROC:")
    for s, m in per_site.items():
        if "auroc" in m:
            print(f"  {s:20s}  n={m['n']:4d}  prev={m['prevalence']:.2f}  AUROC={m['auroc']:.3f}")
        else:
            print(f"  {s:20s}  n={m['n']:4d}  {m.get('note','')}")

    print(f"\nSaved to {outdir}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--slide-table", default="results/data/slide_table_pyramidal.csv")
    p.add_argument("--clinical", default="results/data/clinical_table.csv")
    p.add_argument("--weights", default=DEFAULT_WAGNER_CHECKPOINT)
    p.add_argument("--outdir", default="results/analysis/wagner_zeroshot")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    run(Path(args.slide_table), Path(args.clinical), Path(args.weights), Path(args.outdir), args.device)


if __name__ == "__main__":
    main()
