"""Waiv under a Wagner-style aggregator (from scratch) — the apples-to-apples test.

Trains a Wagner-style slide transformer (same architecture as the champion: projection -> CLS +
2 transformer blocks -> head) FROM SCRATCH on Waiv all-tile bags, using the frozen W1 fold contract
(803 slides / 217 patients / 5 outer folds). Patient-grouped outer OOF with inner early-stopping;
patient aggregation max/sqrt(n).

This has NO pretrained aggregator (unlike Wagner, whose head is pretrained on ~13K Western slides),
so the honest reference is **W1-3 = 0.653** (CTransPath cohort-trained aggregator). Wagner 0.717 is
context, not a fair target. Beating W1-3 isolates the encoder under a real aggregator.

Usage: python scripts/waiv_wagner.py <phaet|mascaret> [--tile-cap 2048] [--epochs 20]
Output: results/analysis/error_anatomy/waiv_wagner/<model>.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import argo_deepmsi  # noqa: F401
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import zarr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from argo_deepmsi.models.wagner import WagnerTransformer
from argo_deepmsi.w1 import _stable_seed

INDEX = "results/experiments/w1_ctranspath/slide_index.csv"
OUT = Path("results/analysis/error_anatomy/waiv_wagner")
SEED = 42


class WaivBagStore:
    def __init__(self, index, model, max_tiles, seed=SEED):
        self.index = index.reset_index(drop=True)
        self.model = model
        self.max_tiles = int(max_tiles)
        self.seed = seed

    def _path(self, row):
        return str(row["ctranspath_zarr"]).replace("ctranspath_tiles", f"{self.model}_tiles")

    def read(self, row_index, epoch=0):
        row = self.index.iloc[int(row_index)]
        arr = zarr.open_array(self._path(row), mode="r")
        n = int(arr.shape[0])
        idx = np.arange(n)
        if n > self.max_tiles:
            rng = np.random.default_rng(_stable_seed(str(row["slide_id"]), self.seed, int(epoch)))
            idx = np.sort(rng.choice(idx, self.max_tiles, replace=False))
        return np.asarray(arr.oindex[idx, :], dtype=np.float32)


def _pos_weight(index, rows):
    y = index.iloc[rows].drop_duplicates("patient_id")["y"].to_numpy()
    return float((y == 0).sum() / max(1, (y == 1).sum()))


def _train_epoch(model, store, index, rows, opt, lossf, device, epoch, rng):
    model.train()
    order = list(rows)
    rng.shuffle(order)
    for i in order:
        opt.zero_grad()
        bag = torch.from_numpy(store.read(i, epoch=epoch)).unsqueeze(0).to(device)
        logit = model(bag).reshape(-1)
        loss = lossf(logit, torch.tensor([float(index.iloc[i]["y"])], device=device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


@torch.no_grad()
def _predict(model, store, index, rows, device):
    model.eval()
    out = {}
    for i in rows:
        bag = torch.from_numpy(store.read(i, epoch=-1)).unsqueeze(0).to(device)
        out[int(i)] = float(torch.sigmoid(model(bag).reshape(-1)).item())
    return out


def _patient_auroc(sl, by_site=False):
    rows = []
    for pid, g in sl.groupby("patient_id"):
        rows.append({"patient_id": pid, "y": int(g["y"].iloc[0]), "site": g["site"].iloc[0],
                     "score": float(g["p_msih"].max() / np.sqrt(len(g)))})
    pat = pd.DataFrame(rows)
    overall = float(roc_auc_score(pat["y"], pat["score"])) if pat["y"].nunique() == 2 else float("nan")
    if not by_site:
        return overall, pat
    bs = {s: (float(roc_auc_score(gg["y"], gg["score"])) if gg["y"].nunique() == 2 else None)
          for s, gg in pat.groupby("site")}
    return overall, pat, bs


def run(model_name, tile_cap, epochs, device):
    index = pd.read_csv(INDEX)
    dim = {"phaet": 1024, "mascaret": 1536}[model_name]
    store = WaivBagStore(index, model_name, tile_cap)
    dev = torch.device(device)
    oof = {}
    for fold in sorted(index["outer_fold"].unique()):
        tr_all = np.flatnonzero(index["outer_fold"].to_numpy() != fold)
        te = np.flatnonzero(index["outer_fold"].to_numpy() == fold)
        # patient-grouped inner split for early stopping
        gkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED + fold)
        y_tr = index.iloc[tr_all]["y"].to_numpy()
        g_tr = index.iloc[tr_all]["patient_id"].to_numpy()
        itr_rel, ival_rel = next(gkf.split(np.zeros(len(tr_all)), y_tr, groups=g_tr))
        itr, ival = tr_all[itr_rel], tr_all[ival_rel]

        torch.manual_seed(SEED + 10_000 + fold)
        model = WagnerTransformer(input_dim=dim).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
        lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([_pos_weight(index, itr)], device=dev))
        rng = np.random.default_rng(SEED + fold)

        best_auc, best_test = -1.0, None
        for ep in range(1, epochs + 1):
            _train_epoch(model, store, index, itr, opt, lossf, dev, ep, rng)
            vp = _predict(model, store, index, ival, dev)
            vsl = index.iloc[ival][["patient_id", "y", "site"]].copy()
            vsl["p_msih"] = [vp[int(i)] for i in ival]
            vauc, _ = _patient_auroc(vsl)
            if not np.isnan(vauc) and vauc > best_auc:
                best_auc = vauc
                best_test = _predict(model, store, index, te, dev)
            print(f"  fold{fold} ep{ep} inner_auc={vauc:.3f} best={best_auc:.3f}", flush=True)
        if best_test is None:
            best_test = _predict(model, store, index, te, dev)
        oof.update(best_test)
        print(f"fold {fold} done (best inner {best_auc:.3f})", flush=True)

    sl = index[["slide_id", "patient_id", "site", "y"]].copy()
    sl["p_msih"] = [oof.get(i, np.nan) for i in range(len(index))]
    sl = sl.dropna(subset=["p_msih"])
    overall, pat, by_site = _patient_auroc(sl, by_site=True)
    # patient bootstrap CI
    rng = np.random.default_rng(SEED)
    aucs = []
    for _ in range(1000):
        s = pat.sample(len(pat), replace=True, random_state=int(rng.integers(1 << 31)))
        if s["y"].nunique() == 2:
            aucs.append(roc_auc_score(s["y"], s["score"]))
    res = {"model": model_name, "tile_cap": tile_cap, "epochs": epochs,
           "overall": overall, "ci95": [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))],
           "by_site": by_site, "n_slides": int(len(sl))}
    OUT.mkdir(parents=True, exist_ok=True)
    sl.to_csv(OUT / f"{model_name}_slide_scores.csv", index=False)
    (OUT / f"{model_name}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2), flush=True)
    print("Refs: W1-3 CTransPath cohort-trained aggregator 0.653 | Wagner (pretrained) 0.717", flush=True)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model", choices=["phaet", "mascaret"])
    p.add_argument("--tile-cap", type=int, default=2048)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    run(a.model, a.tile_cap, a.epochs, a.device)


if __name__ == "__main__":
    main()
