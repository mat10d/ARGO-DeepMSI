"""Per-case failure forensic — disentangle morphology / quality / label at the tile level.

For one slide, cross three tile-level signals over the tumor region:

- **Wagner attention** — where the frozen champion looks (capture_attention hook).
- **VL MSI-vs-MSS axis** — CONCH v1.5 (TITAN text encoder) cosine to curated MSI-H vs MSS
  morphology prompts, per cached ``conch_v1.5_tiles`` embedding (no image re-encode).
- **VL quality axis** — cosine to sharp/well-stained vs blurry/folded/faded prompts.

The question: on a confident false positive, does the natural-language encoder *also* read the
tumor region Wagner attends to as MSI-H-like (genuine morphology mimicry), or does it read it as
degraded (a quality artifact), or neither (Wagner-specific)?

Reuses prompt sets + TITAN text encoder from ``scripts/vl_text_cosine.py``.
Outputs under ``results/analysis/error_anatomy/cases/<slide_id>/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vl_text_cosine import (  # noqa: E402
    build_class_prototypes,
    encode_prompts,
    load_titan_text_encoder,
)

QUALITY_GOOD = [
    "a sharp, well-focused hematoxylin and eosin histopathology image",
    "high quality H&E stained tissue with crisp nuclear detail",
]
QUALITY_BAD = [
    "a blurry, out-of-focus histopathology image",
    "histology tissue with folds and scanning artifacts",
    "faded, poorly stained H&E tissue",
]


def _read_table(wsi, name):
    return np.asarray(wsi.tables[name].X, dtype=np.float32) if name in wsi.tables else None


def _axis(protos_pos, protos_neg, X):
    x = F.normalize(torch.from_numpy(X), dim=-1)
    return (x @ protos_pos).numpy() - (x @ protos_neg).numpy()


def run(slide_id: str, patient_id: str, device: str, outdir: Path, topk: int = 100) -> dict:
    from wsidata import open_wsi

    from argo_deepmsi.models.wagner import capture_attention, load_wagner

    st = pd.read_csv("results/data/slide_table_pyramidal.csv")
    st["slide_id"] = st["FILENAME"].map(lambda p: Path(p).stem)
    svs = Path(st[st["slide_id"] == slide_id].iloc[0]["FILENAME"])
    wsi = open_wsi(str(svs), store=str(svs.parent), attach_thumbnail=False)

    conch = _read_table(wsi, "conch_v1.5_tiles")
    ctp = _read_table(wsi, "ctranspath_tiles")
    if conch is None or ctp is None:
        raise RuntimeError(f"{slide_id}: missing conch/ctranspath tiles")
    n = conch.shape[0]
    assert ctp.shape[0] == n, f"tile count mismatch conch {n} vs ctranspath {ctp.shape[0]}"

    tmask_path = Path(f"results/data/tumor_tiles/{slide_id}.npy")
    tumor_idx = np.load(tmask_path) if tmask_path.exists() else np.array([], dtype=int)
    is_tumor = np.zeros(n, dtype=bool)
    is_tumor[tumor_idx[(tumor_idx >= 0) & (tumor_idx < n)]] = True

    # Wagner per-tile attention (frozen; capture hook keeps CLS->tile vector)
    model = load_wagner(device=device)
    with torch.no_grad(), capture_attention(model):
        _ = model(torch.from_numpy(ctp).unsqueeze(0).to(device))
    attn = model.last_cls_attn.cpu().numpy()
    assert attn.shape[0] == n

    # VL axes on cached conch tiles
    tmodel, tok = load_titan_text_encoder(device)
    protos, _ = build_class_prototypes(tmodel, tok, device)              # (2,768): [msi, mss]
    zq_good = F.normalize(encode_prompts(tmodel, tok, QUALITY_GOOD, device).mean(0), dim=-1)
    zq_bad = F.normalize(encode_prompts(tmodel, tok, QUALITY_BAD, device).mean(0), dim=-1)
    vl_msi = _axis(protos[0], protos[1], conch)                          # MSI - MSS
    vl_qual = _axis(zq_good, zq_bad, conch)                              # good - bad

    df = pd.DataFrame({"tile": np.arange(n), "is_tumor": is_tumor,
                       "wagner_attn": attn, "vl_msi_minus_mss": vl_msi, "vl_quality": vl_qual})
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "tile_forensic.csv", index=False)

    top = df.sort_values("wagner_attn", ascending=False).head(topk)
    tum = df[df.is_tumor]

    def m(s):
        return float(s.mean()) if len(s) else float("nan")

    summary = {
        "slide_id": slide_id, "patient_id": patient_id, "n_tiles": int(n),
        "n_tumor_tiles": int(is_tumor.sum()),
        "wagner_attn_corr_vl_msi": float(np.corrcoef(df.wagner_attn, df.vl_msi_minus_mss)[0, 1]),
        "wagner_attn_corr_vl_quality": float(np.corrcoef(df.wagner_attn, df.vl_quality)[0, 1]),
        "vl_msi_all": m(df.vl_msi_minus_mss), "vl_msi_tumor": m(tum.vl_msi_minus_mss),
        "vl_msi_top_attn": m(top.vl_msi_minus_mss),
        "vl_quality_all": m(df.vl_quality), "vl_quality_tumor": m(tum.vl_quality),
        "vl_quality_top_attn": m(top.vl_quality),
        "frac_top_attn_on_tumor": float(top.is_tumor.mean()),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--slide-id", required=True)
    p.add_argument("--patient-id", default="")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--outdir", default=None)
    a = p.parse_args()
    outdir = Path(a.outdir or f"results/analysis/error_anatomy/cases/{a.slide_id}")
    run(a.slide_id, a.patient_id, a.device, outdir)


if __name__ == "__main__":
    main()
