"""Cohort-wide VL tumor calibration — generalize the per-case forensic across all patients.

For every slide, compute a tumor-restricted vision-language MSI score (CONCH v1.5 tile embeddings
vs curated MSI-H/MSS prompts), aggregate to patient, then:

1. **Validate the instrument.** Does the VL-tumor axis separate *true* MSI-H from *true* MSS
   patients at all (AUROC)? If not, it cannot adjudicate false positives and we say so.
2. **Place the false positives.** For Wagner's confident MSS->MSI-H false positives, does their
   tumor read MSI-H-like (genuine morphology mimicry) or MSS-like (Wagner-specific error)?
   Reported as a z-position between the true-MSS and true-MSI-H reference means.

Reuses the TITAN/CONCH-v1.5 text encoder + prompts from ``scripts/vl_text_cosine.py`` and the
quality prompts from ``scripts/case_forensic.py``. CPU-only (cached ``conch_v1.5_tiles``).
Outputs under ``results/analysis/error_anatomy/vl_tumor/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import argo_deepmsi  # noqa: F401  (import triggers .env autoload -> HF_TOKEN for gated TITAN)
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from case_forensic import QUALITY_BAD, QUALITY_GOOD  # noqa: E402
from vl_text_cosine import (  # noqa: E402
    build_class_prototypes,
    encode_prompts,
    load_titan_text_encoder,
)


def _tumor_vl(conch, tumor_idx, proto_msi, proto_mss, qg, qb, topk=64):
    x = F.normalize(torch.from_numpy(conch), dim=-1)
    vl = (x @ proto_msi).numpy() - (x @ proto_mss).numpy()
    q = (x @ qg).numpy() - (x @ qb).numpy()
    n = conch.shape[0]
    mask = np.zeros(n, dtype=bool)
    ti = tumor_idx[(tumor_idx >= 0) & (tumor_idx < n)]
    mask[ti] = True
    tum = vl[mask]
    return {
        "n_tiles": int(n), "n_tumor": int(mask.sum()),
        "all_vl_msi": float(vl.mean()),
        "tumor_vl_msi": float(tum.mean()) if mask.any() else float("nan"),
        "tumor_vl_msi_topk": float(np.sort(tum)[-min(topk, len(tum)):].mean()) if mask.any() else float("nan"),
        "tumor_vl_quality": float(q[mask].mean()) if mask.any() else float("nan"),
    }


def run(device: str, outdir: Path) -> dict:
    from wsidata import open_wsi

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    st = pd.read_csv("results/data/slide_table_pyramidal.csv")
    st["slide_id"] = st["FILENAME"].map(lambda p: Path(p).stem)
    cl = pd.read_csv("results/data/clinical_table.csv")[["PATIENT", "isMSIH"]]
    cl["y"] = (cl["isMSIH"] == "MSI-H").astype(int)
    st = st.merge(cl[["PATIENT", "y"]], left_on="PATIENT", right_on="PATIENT", how="inner")

    model, tok = load_titan_text_encoder(device)
    protos, _ = build_class_prototypes(model, tok, device)
    qg = F.normalize(encode_prompts(model, tok, QUALITY_GOOD, device).mean(0), dim=-1)
    qb = F.normalize(encode_prompts(model, tok, QUALITY_BAD, device).mean(0), dim=-1)

    tumor_dir = Path("results/data/tumor_tiles")
    rows = []
    for _, r in tqdm(st.iterrows(), total=len(st), desc="vl-tumor"):
        svs = Path(r["FILENAME"])
        sid = r["slide_id"]
        tmask = tumor_dir / f"{sid}.npy"
        if not tmask.exists():
            continue
        try:
            wsi = open_wsi(str(svs), store=str(svs.parent), attach_thumbnail=False)
            if "conch_v1.5_tiles" not in wsi.tables:
                continue
            conch = np.asarray(wsi.tables["conch_v1.5_tiles"].X, dtype=np.float32)
            if conch.shape[0] == 0:
                continue
            m = _tumor_vl(conch, np.load(tmask), protos[0], protos[1], qg, qb)
            rows.append({"slide_id": sid, "patient_id": r["PATIENT"], "site": r["SITE"],
                         "y": int(r["y"]), **m})
        except Exception as e:
            print(f"  skip {sid}: {e}")
            continue

    sl = pd.DataFrame(rows)
    sl.to_csv(out / "slide_vl_tumor.csv", index=False)

    # patient-level: mean tumor VL over the patient's slides
    pat = (sl.groupby("patient_id")
             .agg(y=("y", "max"), site=("site", "first"),
                  tumor_vl_msi=("tumor_vl_msi", "mean"),
                  tumor_vl_msi_topk=("tumor_vl_msi_topk", "mean"),
                  tumor_vl_quality=("tumor_vl_quality", "mean")).reset_index())

    # join Wagner error class from the Stage-A ledger
    ledger = pd.read_csv("results/analysis/error_anatomy/A_error_ledger.csv")
    pat = pat.merge(ledger[["patient_id", "error_class", "confident_wrong", "attributed_cause"]],
                    on="patient_id", how="left")
    pat.to_csv(out / "patient_vl_tumor.csv", index=False)

    # 1) instrument validity: does tumor VL separate true MSI-H vs true MSS?
    valid = pat.dropna(subset=["tumor_vl_msi"])
    auroc = float(roc_auc_score(valid["y"], valid["tumor_vl_msi"])) if valid["y"].nunique() == 2 else float("nan")

    # 2) place the confident FPs between MSS and MSI-H references
    ref_pos = valid[valid.y == 1]["tumor_vl_msi"]
    ref_neg = valid[(valid.y == 0) & (valid.error_class == "TN")]["tumor_vl_msi"]
    fp = valid[(valid.error_class == "FP") & (valid.confident_wrong)]["tumor_vl_msi"]
    mu_pos, mu_neg = float(ref_pos.mean()), float(ref_neg.mean())
    denom = (mu_pos - mu_neg)
    fp_position = float((fp.mean() - mu_neg) / denom) if denom else float("nan")  # 0=MSS-like,1=MSI-H-like

    summary = {
        "n_patients_scored": int(len(valid)),
        "tumor_vl_msi_auroc_true_labels": auroc,
        "mean_tumor_vl_msi": {"true_MSIH": mu_pos, "true_MSS_TN": mu_neg,
                              "confident_FP": float(fp.mean())},
        "fp_position_0MSS_1MSIH": fp_position,
        "tumor_vl_quality_by_group": {
            "true_MSIH": float(valid[valid.y == 1]["tumor_vl_quality"].mean()),
            "confident_FP": float(valid[(valid.error_class == "FP") & valid.confident_wrong]["tumor_vl_quality"].mean()),
            "TN": float(valid[valid.error_class == "TN"]["tumor_vl_quality"].mean()),
        },
    }
    (out / "vl_tumor_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--outdir", default="results/analysis/error_anatomy/vl_tumor")
    a = p.parse_args()
    run(a.device, Path(a.outdir))


if __name__ == "__main__":
    main()
