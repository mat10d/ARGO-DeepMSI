"""S5 — Deep-Sets patient-bag aggregator (cardinality-calibrated).

The champion aggregates a patient's per-slide scores with a fixed max/√n heuristic.
That heuristic is a hand-picked cardinality correction; S5 replaces it with a learned,
permutation-invariant Deep-Sets aggregator (Zaheer et al. 2017) that sees the whole
slide bag and is given explicit cardinality features, so it can calibrate for bag size
rather than assume √n.

Per patient, each slide contributes a feature vector [Wagner p(MSI-H) ‖ TITAN slide
embedding] (1 + 768 d). A shared encoder φ maps each slide, the bag is pooled by
mean ⊕ max, the cardinality features [n, √n, log(1+n)] are appended, and a head ρ maps
to a patient logit. A small M-model ensemble (averaged) reduces run-to-run variance.

Trained with class-weighted BCE under patient-grouped CV; few-shot curve K∈{1,2,4,8,16,all}
subsamples K patients per class. Frozen features only.

Resolution: patient.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from ..eval.screening import screening_block
from .base import Scorer, ScoreColumn
from .registry import register

WAGNER_CSV = Path("results/scorers/wagner_zeroshot/slide_scores.csv")
TITAN_DIR = Path("results/embeddings/conch_v1.5_titan")
SEED = 42
N_SPLITS = 5
M_ENSEMBLE = 3
EPOCHS = 60
HIDDEN = 128
LR = 1e-3
FEWSHOT_K: tuple[int | str, ...] = (1, 2, 4, 8, 16, "all")


def _torch():
    import torch
    torch.manual_seed(SEED)
    try:
        torch.set_num_threads(4)
    except Exception:  # noqa: BLE001
        pass
    return torch


def _load_bags(clean_slide_ids: set[str] | None):
    if not WAGNER_CSV.exists() or not (TITAN_DIR / "embeddings.npy").exists():
        return None
    wag = pd.read_csv(WAGNER_CSV)
    if clean_slide_ids is not None:
        wag = wag[wag["slide_id"].isin(clean_slide_ids)]
    wag = wag.dropna(subset=["p_msih"]).reset_index(drop=True)
    if len(wag) == 0:
        return None
    emb = np.load(TITAN_DIR / "embeddings.npy").astype(np.float32)
    meta = pd.read_csv(TITAN_DIR / "metadata.csv")
    row_of = {sid: i for i, sid in enumerate(meta["slide_id"])}
    # per-slide feature [wagner_p, titan_emb]; slides missing a TITAN row → wagner + zeros
    dim = emb.shape[1]
    feats, keep = [], []
    for r in wag.itertuples():
        e = emb[row_of[r.slide_id]] if r.slide_id in row_of else np.zeros(dim, np.float32)
        feats.append(np.concatenate([[np.float32(r.p_msih)], e]))
        keep.append(True)
    wag = wag[keep].reset_index(drop=True)
    X = np.vstack(feats).astype(np.float32)
    # group rows into patient bags
    bags, y, sites, pids = [], [], [], []
    for pid, g in wag.groupby("patient_id"):
        idxs = g.index.to_numpy()
        bags.append(X[idxs])
        y.append(int(g["y"].iloc[0]))
        sites.append(g["site"].iloc[0])
        pids.append(pid)
    return bags, np.array(y), np.array(sites, dtype=object), np.array(pids, dtype=object), X.shape[1]


def _make_model(dim: int, seed: int):
    torch = _torch()
    import torch.nn as nn
    torch.manual_seed(seed)

    class DeepSets(nn.Module):
        def __init__(self):
            super().__init__()
            self.phi = nn.Sequential(nn.Linear(dim, HIDDEN), nn.ReLU(),
                                     nn.Dropout(0.2), nn.Linear(HIDDEN, HIDDEN), nn.ReLU())
            self.rho = nn.Sequential(nn.Linear(2 * HIDDEN + 3, 64), nn.ReLU(),
                                     nn.Dropout(0.2), nn.Linear(64, 1))

        def forward(self, bag):  # bag: (n_slides, dim)
            h = self.phi(bag)
            pooled = torch.cat([h.mean(0), h.amax(0)])  # (2*HIDDEN,)
            n = bag.shape[0]
            card = torch.tensor([n, n ** 0.5, float(np.log1p(n))], dtype=torch.float32)
            return self.rho(torch.cat([pooled, card]))  # (1,)

    return DeepSets()


def _train(bags, y, tr, dim, seed, scaler, pos_weight):
    torch = _torch()
    import torch.nn as nn
    torch.manual_seed(seed)
    model = _make_model(dim, seed)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
    order = list(tr)
    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(EPOCHS):
        rng.shuffle(order)
        for i in order:
            opt.zero_grad()
            bag = torch.from_numpy(scaler.transform(bags[i]).astype(np.float32))
            logit = model(bag)
            lossf(logit, torch.tensor([float(y[i])])).backward()
            opt.step()
    model.eval()
    return model


def _predict(model, bags, idxs, scaler) -> np.ndarray:
    torch = _torch()
    out = np.zeros(len(idxs), dtype=np.float32)
    with torch.no_grad():
        for j, i in enumerate(idxs):
            bag = torch.from_numpy(scaler.transform(bags[i]).astype(np.float32))
            out[j] = torch.sigmoid(model(bag)).item()
    return out


def _oof(bags, y, sites, support_fn=None) -> np.ndarray:
    dim = bags[0].shape[1]
    pos_weight = float((y == 0).sum() / max(1, (y == 1).sum()))
    oof = np.full(len(y), np.nan, dtype=np.float32)
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    groups = np.arange(len(y))  # each patient is its own group → stratified k-fold on patients
    for fold, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y, groups=groups)):
        sup = tr if support_fn is None else support_fn(tr)
        if sup is None or len(np.unique(y[sup])) < 2:
            continue
        scaler = StandardScaler().fit(np.vstack([bags[i] for i in sup]))
        preds = []
        for m in range(M_ENSEMBLE):
            model = _train(bags, y, sup, dim, SEED + 100 * fold + m, scaler, pos_weight)
            preds.append(_predict(model, bags, te, scaler))
        oof[te] = np.mean(preds, axis=0)
    return oof


def _auroc(y, s) -> float:
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


def _fewshot_curve(bags, y, sites) -> list[dict]:
    curve = []
    for k in FEWSHOT_K:
        if k == "all":
            oof = _oof(bags, y, sites, None)
            mask = ~np.isnan(oof)
            curve.append({"K": "all", "n_train_per_class": int(min((y == 1).sum(), (y == 0).sum())),
                          "patient_auroc": _auroc(y[mask], oof[mask])})
        else:
            aurocs = []
            for rep in range(3):
                rng = np.random.default_rng(SEED + rep)

                def pick(tr, _rng=rng, _k=int(k)):
                    pos = tr[y[tr] == 1]
                    neg = tr[y[tr] == 0]
                    if len(pos) < _k or len(neg) < _k:
                        return None
                    return np.concatenate([_rng.choice(pos, _k, replace=False),
                                           _rng.choice(neg, _k, replace=False)])

                oof = _oof(bags, y, sites, pick)
                mask = ~np.isnan(oof)
                if len(np.unique(y[mask])) == 2:
                    aurocs.append(_auroc(y[mask], oof[mask]))
            curve.append({"K": int(k), "n_train_per_class": int(k),
                          "patient_auroc": float(np.nanmean(aurocs)) if aurocs else float("nan")})
    return curve


class SetEncoderAgg(Scorer):
    name = "setencoder_agg"
    description = (
        "Deep-Sets patient-bag aggregator: a permutation-invariant encoder over each "
        "patient's [Wagner p(MSI-H) ‖ TITAN slide-emb] slide set, pooled mean⊕max with "
        "explicit cardinality features and an MLP head — a learned, cardinality-calibrated "
        "replacement for the champion's max/√n. M-model ensemble, patient-grouped CV, "
        "few-shot curve K∈{1,2,4,8,16,all}."
    )
    needs_training_on_our_data = True
    resolution = "patient"
    score_columns = [
        ScoreColumn("p_msih", "Deep-Sets patient p(MSI-H)", primary=True),
    ]

    def compute_batch(
        self,
        slide_table: pd.DataFrame,
        *,
        clean_slide_ids: set[str] | None = None,
        full_curve: bool = False,
        write_outputs: bool = True,
        retrain: bool = False,
        **_,
    ) -> pd.DataFrame:
        # Cache-first: training the Deep-Sets ensemble under CV is a few minutes, and
        # torch is not bit-reproducible across processes, so the leaderboard re-race
        # reads the cached patient scores (keeping metrics.json == board). Only the
        # runner (retrain=True) trains.
        if not retrain and self.score_path is not None and self.score_path.exists():
            df = pd.read_csv(self.score_path)
            if clean_slide_ids is not None:
                keep = pd.read_csv(WAGNER_CSV)
                keep = set(keep[keep["slide_id"].isin(clean_slide_ids)]["patient_id"])
                df = df[df["patient_id"].isin(keep)].reset_index(drop=True)
            return df

        loaded = _load_bags(clean_slide_ids)
        if loaded is None:
            raise RuntimeError(
                f"{self.name}: Wagner scores / TITAN embeddings missing "
                f"({WAGNER_CSV}, {TITAN_DIR})"
            )
        bags, y, sites, pids, _ = loaded
        oof = _oof(bags, y, sites, None)
        pat_df = pd.DataFrame({
            "patient_id": pids, "y": y, "site": sites, "n_slides": [len(b) for b in bags],
            "p_msih": oof,
        }).dropna(subset=["p_msih"]).reset_index(drop=True)
        self._last = {}
        if full_curve:
            self._last["curve"] = _fewshot_curve(bags, y, sites)

        if write_outputs and self.score_path is not None:
            self.score_path.parent.mkdir(parents=True, exist_ok=True)
            pat_df.to_csv(self.score_path, index=False)
            if full_curve:
                pd.DataFrame(self._last["curve"]).to_csv(
                    self.score_path.parent / "few_shot_curve.csv", index=False)
        return pat_df


register("setencoder_agg", SetEncoderAgg)


def _write_metrics(scorer: "SetEncoderAgg", pat_df: pd.DataFrame) -> dict:
    import json

    from ..eval.metrics import _safe_auprc, _safe_auroc

    y, s = pat_df["y"].to_numpy(), pat_df["p_msih"].to_numpy()
    block = screening_block(y, s, (0.90, 0.95, 0.96, 0.98))

    by_site = {}
    for site, sub in pat_df.groupby("site"):
        yy, ss = sub["y"].to_numpy(), sub["p_msih"].to_numpy()
        by_site[str(site)] = {
            "n": int(len(sub)), "prevalence": float(sub["y"].mean()),
            "auroc": _safe_auroc(yy, ss),
            "spec_at_sens95": screening_block(yy, ss, (0.95,))["spec_at_sens95"]
            if sub["y"].nunique() == 2 else float("nan"),
        }

    pat = pat_df.assign(_bucket=pd.cut(pat_df["n_slides"], [0, 1, 2, 4, np.inf], labels=["1", "2", "3-4", "5+"]))
    by_bagsize = {str(b): {"n": int(len(sub)), "auroc": _safe_auroc(sub["y"].to_numpy(), sub["p_msih"].to_numpy())}
                  for b, sub in pat.groupby("_bucket", observed=True)}

    metrics = {
        "scorer": scorer.name, "aggregator": "DeepSets(mean+max, cardinality-feat)",
        "n_patients": int(len(pat_df)), "prevalence_patient": float(pat_df["y"].mean()),
        "auroc": _safe_auroc(y, s), "auprc": _safe_auprc(y, s),
        "sensitivity": block["op_sens95"]["sensitivity"],
        "spec_at_sens90": block["spec_at_sens90"],
        "spec_at_sens95": block["spec_at_sens95"],
        "spec_at_sens96": block["spec_at_sens96"],
        "npv_at_sens95": block["npv_at_sens95"],
        "operating_threshold": block["op_sens95"]["threshold"],
        "by_site": by_site, "by_bagsize": by_bagsize,
        "few_shot_curve": scorer._last.get("curve", []),
        "screening_clean": block,
    }
    out = SetEncoderAgg().score_path.parent / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2))
    return metrics
