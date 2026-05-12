"""C6 / A1 — Vision-language zero-shot MSI scoring with TITAN's CONCH v1.5
text encoder.

For each slide we score MSI-likeness as

    score(t) = mean_i cos(tile_t,i, MSI_proto) - mean_i cos(tile_t,i, MSS_proto)

where MSI_proto / MSS_proto are L2-mean text embeddings over a curated set of
pathology prompts. We evaluate three aggregation variants:

- ``tile_mean``  : average tile-level (MSI - MSS) similarity per slide
- ``tile_topk``  : mean of top-k tile (MSI - MSS) similarities (default k=64)
- ``slide``      : same score computed against the existing TITAN slide
                   embedding at ``results/embeddings/conch_v1.5_titan/``

Patient-level aggregation uses max/√n on slide scores (the C5 best baseline).

The TITAN text encoder produces 768-d embeddings that share the patch-level
hidden dim with ``conch_v1.5_tiles``. The text↔slide alignment is the one
TITAN was trained for; the text↔patch alignment is heuristic but is the
hypothesis being tested in C6 (Strategy A).

Outputs (under ``results/analysis/c6_a1_vl_zeroshot/``):
    prompts.json               — prompt sets used
    text_embeddings.npy        — class prototypes (2, 768)
    slide_scores.csv           — per-slide scores for all variants
    patient_scores.csv         — per-patient scores (max/sqrt(n))
    metrics.json               — overall + per-site AUROC, comparison vs Wagner
    roc.png                    — ROC curves per variant
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
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from tqdm import tqdm


MSI_PROMPTS = [
    "colorectal adenocarcinoma with dense tumor-infiltrating lymphocytes",
    "poorly differentiated colorectal carcinoma with mucinous features",
    "medullary-type colorectal carcinoma with prominent lymphoid response",
    "colorectal tumor with Crohn's-like peritumoral lymphoid reaction",
    "signet ring cell features in colorectal adenocarcinoma",
    "microsatellite-unstable colorectal cancer with abundant intraepithelial lymphocytes",
    "colorectal carcinoma with mucin pools and poor gland formation",
]

MSS_PROMPTS = [
    "well-differentiated colorectal adenocarcinoma",
    "moderately differentiated tubular adenocarcinoma",
    "colorectal adenocarcinoma without significant lymphocytic infiltrate",
    "conventional colorectal adenocarcinoma with glandular architecture",
    "microsatellite-stable colorectal cancer with low lymphocyte infiltration",
    "colorectal carcinoma with well-formed glands and minimal stromal lymphocytes",
]


def load_titan_text_encoder(device: str):
    from huggingface_hub import login
    from transformers import AutoModel, AutoTokenizer

    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token, add_to_git_credential=False)
    model = (
        AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)
        .to(device)
        .eval()
    )
    tok = AutoTokenizer.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)
    return model, tok


@torch.no_grad()
def encode_prompts(
    model, tok, prompts: list[str], device: str, max_length: int = 128
) -> torch.Tensor:
    ids = tok(
        prompts,
        return_tensors="pt",
        padding="max_length",
        max_length=max_length,
        truncation=True,
    ).input_ids.to(device)
    z = model.encode_text(ids, normalize=True)
    return z.float().cpu()


def build_class_prototypes(
    model, tok, device: str
) -> tuple[torch.Tensor, dict]:
    z_msi = encode_prompts(model, tok, MSI_PROMPTS, device)
    z_mss = encode_prompts(model, tok, MSS_PROMPTS, device)
    proto_msi = F.normalize(z_msi.mean(0, keepdim=True), dim=-1).squeeze(0)
    proto_mss = F.normalize(z_mss.mean(0, keepdim=True), dim=-1).squeeze(0)
    protos = torch.stack([proto_msi, proto_mss])  # (2, 768)
    info = {
        "msi_prompts": MSI_PROMPTS,
        "mss_prompts": MSS_PROMPTS,
        "n_msi": len(MSI_PROMPTS),
        "n_mss": len(MSS_PROMPTS),
    }
    return protos, info


def zarr_to_svs(svs_path: str) -> str:
    s = str(svs_path)
    if s.endswith(".pyramidal.tiff"):
        return s.replace(".pyramidal.tiff", ".pyramidal.zarr")
    return str(Path(s).with_suffix(".zarr"))


def read_conch_v15_tiles(zarr_path: str) -> np.ndarray | None:
    if not Path(zarr_path).exists():
        return None
    g = zarr.open(zarr_path, mode="r")
    if "tables" not in g or "conch_v1.5_tiles" not in g["tables"]:
        return None
    return np.asarray(g["tables"]["conch_v1.5_tiles"]["X"][...], dtype=np.float32)


def score_slide(
    tile_feats: np.ndarray, protos: torch.Tensor, topk: int = 64
) -> dict:
    # L2-normalize tile features, cosine via dot
    X = torch.from_numpy(tile_feats)
    X = F.normalize(X, dim=-1)
    sims = X @ protos.T  # (n_tiles, 2)  cols = [msi, mss]
    delta = (sims[:, 0] - sims[:, 1]).numpy()  # per-tile MSI-MSS
    n = len(delta)
    return {
        "n_tiles": int(n),
        "tile_mean": float(delta.mean()),
        "tile_max": float(delta.max()),
        "tile_topk": float(np.sort(delta)[-min(topk, n) :].mean()),
        "msi_mean": float(sims[:, 0].mean()),
        "mss_mean": float(sims[:, 1].mean()),
    }


def score_titan_slide_emb(
    slide_emb: np.ndarray, protos: torch.Tensor
) -> float:
    s = torch.from_numpy(slide_emb)
    s = F.normalize(s, dim=-1)
    sims = (s @ protos.T).numpy()  # (2,)
    return float(sims[0] - sims[1])


def patient_max_sqrtn(group: pd.DataFrame, score_col: str) -> float:
    n = len(group)
    return float(group[score_col].max() / np.sqrt(n))


def evaluate(
    slide_df: pd.DataFrame, score_col: str
) -> tuple[float, float, dict]:
    if slide_df["y"].nunique() != 2:
        return float("nan"), float("nan"), {}
    slide_auroc = float(roc_auc_score(slide_df["y"], slide_df[score_col]))
    slide_auprc = float(average_precision_score(slide_df["y"], slide_df[score_col]))
    per_site = {}
    for s, sub in slide_df.groupby("site"):
        if sub["y"].nunique() == 2:
            per_site[s] = {
                "n": int(len(sub)),
                "prevalence": float(sub["y"].mean()),
                "auroc": float(roc_auc_score(sub["y"], sub[score_col])),
                "auprc": float(average_precision_score(sub["y"], sub[score_col])),
            }
        else:
            per_site[s] = {
                "n": int(len(sub)),
                "prevalence": float(sub["y"].mean()),
                "note": "single-class",
            }
    return slide_auroc, slide_auprc, per_site


def run(
    slide_table: Path,
    clinical: Path,
    titan_dir: Path,
    outdir: Path,
    device: str,
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

    # Pre-loaded TITAN slide embeddings keyed by slide_id
    titan_emb = np.load(titan_dir / "embeddings.npy")
    titan_meta = pd.read_csv(titan_dir / "metadata.csv")
    titan_idx = {r["slide_id"]: i for i, r in titan_meta.iterrows()}

    print("Loading TITAN text encoder...")
    model, tok = load_titan_text_encoder(device)
    protos, info = build_class_prototypes(model, tok, device)
    np.save(outdir / "text_embeddings.npy", protos.numpy())
    (outdir / "prompts.json").write_text(json.dumps(info, indent=2))

    rows = []
    for _, r in tqdm(st.iterrows(), total=len(st), desc="A1 VL zero-shot"):
        svs = r["FILENAME"]
        zp = zarr_to_svs(svs)
        X = read_conch_v15_tiles(zp)
        slide_id = Path(svs).stem.replace(".pyramidal", "") + (
            ".pyramidal" if str(svs).endswith(".pyramidal.tiff") else ""
        )
        # match how TITAN metadata stores it (Path.stem of the source path)
        slide_id_meta = Path(svs).stem
        if X is None or X.shape[0] == 0:
            continue

        tile_scores = score_slide(X, protos, topk=topk)
        # Slide-level via existing TITAN embedding if available
        s_idx = titan_idx.get(slide_id_meta)
        if s_idx is None:
            slide_score = float("nan")
        else:
            slide_score = score_titan_slide_emb(titan_emb[s_idx], protos)

        rows.append(
            {
                "slide_id": slide_id_meta,
                "patient_id": r["PATIENT"],
                "site": r["SITE"],
                "stain_location": r.get("stain_location"),
                "y": int(r["y"]),
                **tile_scores,
                "slide_score": slide_score,
            }
        )

    slide_df = pd.DataFrame(rows)
    slide_df.to_csv(outdir / "slide_scores.csv", index=False)

    # Patient-level: max/sqrt(n) on each score variant
    score_cols = ["tile_mean", "tile_topk", "tile_max", "slide_score"]
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
            row[c] = float(sub[c].max() / np.sqrt(len(sub))) if len(sub) else float("nan")
        pat_rows.append(row)
    patient_df = pd.DataFrame(pat_rows)
    patient_df.to_csv(outdir / "patient_scores.csv", index=False)

    # Metrics for each variant
    metrics = {
        "n_slides": int(len(slide_df)),
        "n_patients": int(len(patient_df)),
        "prevalence_slide": float(slide_df["y"].mean()),
        "prevalence_patient": float(patient_df["y"].mean()),
        "topk": topk,
        "variants": {},
    }
    for c in score_cols:
        sd = slide_df.dropna(subset=[c])
        pd_ = patient_df.dropna(subset=[c])
        s_auroc, s_auprc, per_site = evaluate(sd, c)
        p_auroc = (
            float(roc_auc_score(pd_["y"], pd_[c]))
            if pd_["y"].nunique() == 2
            else float("nan")
        )
        p_auprc = (
            float(average_precision_score(pd_["y"], pd_[c]))
            if pd_["y"].nunique() == 2
            else float("nan")
        )
        metrics["variants"][c] = {
            "slide_auroc": s_auroc,
            "slide_auprc": s_auprc,
            "patient_auroc": p_auroc,
            "patient_auprc": p_auprc,
            "per_site": per_site,
        }

    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ROC plot (slide level, all variants)
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
    ax.set_title("C6/A1 — TITAN VL zero-shot MSI (slide-level)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "roc.png", bbox_inches="tight")
    plt.close(fig)

    print("\n=== A1 results ===")
    for c in score_cols:
        v = metrics["variants"][c]
        print(
            f"  {c:<12s}  slide AUROC={v['slide_auroc']:.3f}  patient AUROC={v['patient_auroc']:.3f}"
        )
    print("\nPer-site (slide-level, slide_score variant):")
    for s, m in metrics["variants"]["slide_score"]["per_site"].items():
        if "auroc" in m:
            print(f"  {s:<8s}  n={m['n']:<4d} prev={m['prevalence']:.2f}  AUROC={m['auroc']:.3f}")
        else:
            print(f"  {s:<8s}  n={m['n']:<4d} prev={m['prevalence']:.2f}  ({m.get('note','')})")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--slide-table", default="results/data/slide_table_pyramidal.csv")
    p.add_argument("--clinical", default="results/data/clinical_table.csv")
    p.add_argument("--titan-dir", default="results/embeddings/conch_v1.5_titan")
    p.add_argument("--outdir", default="results/analysis/c6_a1_vl_zeroshot")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--topk", type=int, default=64)
    p.add_argument("--limit", type=int, default=None, help="debug: cap to N slides")
    a = p.parse_args()
    run(
        Path(a.slide_table),
        Path(a.clinical),
        Path(a.titan_dir),
        Path(a.outdir),
        a.device,
        a.topk,
        a.limit,
    )


if __name__ == "__main__":
    main()
