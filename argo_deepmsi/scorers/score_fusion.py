"""Late-average fusion of per-embedding OOF MSI predictions.

Reads the per-model OOF table written by ``scripts/score_fusion.py`` and emits
slide-level p_msih = mean(top-2 single-model OOFs). Default top-2 picks
the best single-model AUROCs from ``fusion_results.csv`` under the
``patient_cv`` regime: ``conch_v1.5_mean`` + ``virchow2_mean``.

Resolution: slide.

The single-model OOFs are themselves LR heads trained on our cohort, so
this scorer composes our-cohort-trained predictions (``needs_training_on_our_data``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from .base import Scorer, ScoreColumn
from .registry import register

EMB_ROOT = Path("results/embeddings")
SLIDE_TABLE_CSV = Path("results/data/slide_table_pyramidal.csv")
CLINICAL_CSV = Path("results/data/clinical_table.csv")
DEFAULT_MEMBERS = ("conch_v1.5_mean", "virchow2_mean")
SEED = 42


def _patient_cv_oof(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=np.float32)
    for tr, te in cv.split(X, y, groups=groups):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
        clf.fit(sc.transform(X[tr]), y[tr])
        oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return oof


class ScoreFusion(Scorer):
    name = "score_fusion"
    description = (
        "Late-average fusion of two single-embedding LR OOF heads "
        f"({' + '.join(DEFAULT_MEMBERS)}). The component heads are refit "
        "5-fold patient-grouped on whatever slide set the harness hands "
        "in; the fusion itself is parameter-free (mean of OOF probs)."
    )
    needs_training_on_our_data = True
    resolution = "slide"
    score_columns = [
        ScoreColumn("p_msih", "late-avg p(MSI-H) over 2 embedding LR heads", primary=True),
    ]

    def compute_batch(
        self,
        slide_table: pd.DataFrame,
        *,
        clean_slide_ids: set[str] | None = None,
        **_,
    ) -> pd.DataFrame:
        # Build aligned slide table + labels
        cl = pd.read_csv(CLINICAL_CSV)[["PATIENT", "isMSIH"]]
        cl["y"] = (cl["isMSIH"] == "MSI-H").astype(int)
        st = pd.read_csv(SLIDE_TABLE_CSV)
        st["slide_id"] = st["FILENAME"].apply(lambda p: Path(p).stem)
        st = st[["slide_id", "PATIENT", "SITE"]].rename(
            columns={"PATIENT": "patient_id", "SITE": "site"}
        )
        st = st.merge(cl[["PATIENT", "y"]], left_on="patient_id", right_on="PATIENT", how="inner")

        # Load + intersect per-embedding features
        mats: dict[str, np.ndarray] = {}
        meta_frames: dict[str, pd.DataFrame] = {}
        for name in DEFAULT_MEMBERS:
            emb_dir = EMB_ROOT / name
            X = np.load(emb_dir / "embeddings.npy")
            meta = pd.read_csv(emb_dir / "metadata.csv")
            if len(meta) != len(X):
                raise ValueError(f"{name}: metadata/embeddings row mismatch")
            meta = meta.copy()
            meta["_row"] = np.arange(len(meta))
            mats[name] = X
            meta_frames[name] = meta

        shared = set(meta_frames[DEFAULT_MEMBERS[0]]["slide_id"])
        for name in DEFAULT_MEMBERS[1:]:
            shared &= set(meta_frames[name]["slide_id"])
        if clean_slide_ids is not None:
            shared &= clean_slide_ids
        shared = sorted(shared)

        base = st[st["slide_id"].isin(shared)].drop_duplicates("slide_id").reset_index(drop=True)
        base = base.set_index("slide_id").loc[shared].reset_index()
        base = base[base["y"].notna() & base["site"].notna()].reset_index(drop=True)

        # Reorder each embedding matrix to match `base`
        aligned: dict[str, np.ndarray] = {}
        for name in DEFAULT_MEMBERS:
            m = meta_frames[name].set_index("slide_id")
            idx = m.loc[base["slide_id"], "_row"].to_numpy()
            aligned[name] = mats[name][idx]

        y = base["y"].to_numpy()
        groups = base["patient_id"].to_numpy()

        oofs = []
        for name in DEFAULT_MEMBERS:
            oofs.append(_patient_cv_oof(aligned[name], y, groups))
        oof_mat = np.stack(oofs, axis=1)

        return pd.DataFrame({
            "slide_id": base["slide_id"],
            "patient_id": base["patient_id"],
            "site": base["site"],
            "y": base["y"],
            "p_msih": oof_mat.mean(axis=1),
        })


register("score_fusion", ScoreFusion)
