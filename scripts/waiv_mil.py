"""Waiv under a supervised aggregator — the decisive test (build/eval split).

Same attention-MIL recipe that gave CONCH-ABMIL 0.607 (S4), swapping the encoder for the Waiv
models. Tumor-only tile bags (<=500/slide), patient-grouped 5-fold OOF with multi-fidelity fusion
(M=3, 12 epochs), patient aggregation max/sqrt(n). Variants: phaet (1024), mascaret (1536),
waiv_concat (2560 = phaet||mascaret per tile).

Two modes so the slow batch-1 training can be parallelised and is restartable:
    python scripts/waiv_mil.py build                 # read zarrs once, save bags per variant
    python scripts/waiv_mil.py eval <phaet|mascaret|waiv_concat>   # OOF on a saved variant

Output: results/analysis/error_anatomy/waiv_mil/{<variant>_bags.npz, <variant>.json}. CPU-only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import anndata as ad
import argo_deepmsi  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from argo_deepmsi.scorers.clam_tilemil import (
    EPOCHS_FULL,
    M_FULL,
    TOPK_FULL,
    _fused_oof,
    _patient_max_sqrtn,
)

SEED = 42
MAX_TILES = 500
OUT = Path("results/analysis/error_anatomy/waiv_mil")
VARIANTS = ("phaet", "mascaret", "waiv_concat")


def _log(msg):
    print(msg, flush=True)


def _read_tiles(zpath: Path, model: str):
    try:
        return np.asarray(ad.read_zarr(str(zpath / "tables" / f"{model}_tiles")).X, dtype=np.float32)
    except Exception:  # noqa: BLE001
        return None


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    coh = pd.read_csv("results/data/cohort_clean.csv")
    coh = coh[coh["in_clean_set"] == 1]
    st = pd.read_csv("results/data/slide_table_pyramidal.csv")
    st["slide_id"] = st["FILENAME"].apply(lambda p: Path(p).stem)
    df = coh[["slide_id", "patient_id", "site"]].merge(
        st[["slide_id", "FILENAME"]].drop_duplicates("slide_id"), on="slide_id", how="left")
    cl = pd.read_csv("results/data/clinical_table.csv")[["PATIENT", "isMSIH"]]
    cl["y"] = (cl["isMSIH"] == "MSI-H").astype(int)
    df = df.merge(cl[["PATIENT", "y"]], left_on="patient_id", right_on="PATIENT", how="inner")

    tumor_dir = Path("results/data/tumor_tiles")
    rng = np.random.default_rng(SEED)
    store = {v: [] for v in VARIANTS}
    rows = []
    for k, (_, r) in enumerate(df.iterrows()):
        if k % 100 == 0:
            _log(f"  building {k}/{len(df)}")
        sid = r["slide_id"]
        tnpy = tumor_dir / f"{sid}.npy"
        if pd.isna(r["FILENAME"]) or not tnpy.exists():
            continue
        z = Path(r["FILENAME"]).with_suffix(".zarr")
        ph, ma = _read_tiles(z, "phaet"), _read_tiles(z, "mascaret")
        if ph is None or ma is None or ph.shape[0] == 0 or ph.shape[0] != ma.shape[0]:
            continue
        tidx = np.load(tnpy)
        tidx = tidx[tidx < ph.shape[0]]
        if len(tidx) == 0:
            continue
        if len(tidx) > MAX_TILES:
            tidx = rng.choice(tidx, MAX_TILES, replace=False)
        store["phaet"].append(ph[tidx])
        store["mascaret"].append(ma[tidx])
        store["waiv_concat"].append(np.hstack([ph[tidx], ma[tidx]]).astype(np.float32))
        rows.append({"slide_id": sid, "patient_id": r["patient_id"], "site": r["site"], "y": int(r["y"])})
    idx = pd.DataFrame(rows).reset_index(drop=True)
    idx.to_csv(OUT / "bags_index.csv", index=False)
    for v in VARIANTS:
        concat = np.concatenate(store[v], axis=0).astype(np.float32)
        lengths = np.array([len(b) for b in store[v]], dtype=np.int64)
        np.savez(OUT / f"{v}_bags.npz", concat=concat, lengths=lengths)
        _log(f"saved {v}: {len(idx)} slides, dim={concat.shape[1]}, tiles={len(concat)}")
    _log(f"built bags: {len(idx)} slides, {idx['patient_id'].nunique()} patients, "
         f"{int(idx['y'].sum())} MSI-H")


def _load(variant):
    idx = pd.read_csv(OUT / "bags_index.csv")
    d = np.load(OUT / f"{variant}_bags.npz")
    concat, lengths = d["concat"], d["lengths"]
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    bag_list = [concat[s:s + n] for s, n in zip(starts, lengths)]
    return bag_list, idx


def evaluate(variant):
    bag_list, idx = _load(variant)
    _log(f"{variant}: {len(idx)} slides, dim={bag_list[0].shape[1]} — running OOF")
    oof = _fused_oof(bag_list, idx, None, M_FULL, EPOCHS_FULL, TOPK_FULL)
    pat = _patient_max_sqrtn(idx.assign(p_msih=oof))
    overall = float(roc_auc_score(pat["y"], pat["score"])) if pat["y"].nunique() == 2 else float("nan")
    by_site = {s: (float(roc_auc_score(g["y"], g["score"])) if g["y"].nunique() == 2 else None)
               for s, g in pat.groupby("site")}
    res = {"overall": overall, "by_site": by_site, "n_slides": int(len(idx))}
    (OUT / f"{variant}.json").write_text(json.dumps(res, indent=2))
    oau = by_site.get("OAUTHC")
    _log(f"{variant:12s} overall={overall:.3f} OAUTHC={oau if oau is None else round(oau, 3)}")
    _log("Refs: CONCH-ABMIL 0.607 | Wagner 0.717/OAUTHC 0.616 | Waiv frozen mean-pool 0.619")
    return res


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "build"
    if mode == "build":
        build()
    elif mode == "eval":
        evaluate(sys.argv[2])
    else:
        raise SystemExit("usage: waiv_mil.py build | eval <variant>")


if __name__ == "__main__":
    main()
