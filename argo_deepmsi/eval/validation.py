"""Leakage-resistant nested validation for cohort-trained scorers.

Every model/configuration choice is made in an inner patient-grouped loop. The
outer loop is used only for held-out prediction. Preprocessing belongs inside
the estimator factory so PCA/scaling is fit on training rows only.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from .metrics import aggregate_to_patient

EstimatorFactory = Callable[[], BaseEstimator]


def _patient_auc(frame: pd.DataFrame, scores: np.ndarray, *, agg: str) -> float:
    scored = frame[["patient_id", "y", "site"]].copy()
    scored["score"] = scores
    patient = aggregate_to_patient(scored, "score", agg=agg)
    if patient["y"].nunique() != 2:
        return float("nan")
    return float(roc_auc_score(patient["y"], patient["score"]))


def nested_grouped_oof(
    frame: pd.DataFrame,
    candidate_matrices: Mapping[str, np.ndarray],
    estimator_factories: Mapping[str, EstimatorFactory],
    *,
    patient_agg: str = "max_sqrtn",
    outer_splits: int = 5,
    inner_splits: int = 4,
    repeats: int = 3,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return repeated nested-CV slide predictions and an outer-fold audit."""
    required = {"slide_id", "patient_id", "site", "y"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"frame missing required columns: {sorted(missing)}")
    if set(candidate_matrices) != set(estimator_factories):
        raise ValueError("candidate matrix and estimator-factory keys must match")
    if not candidate_matrices:
        raise ValueError("at least one candidate is required")
    n = len(frame)
    for name, matrix in candidate_matrices.items():
        if len(matrix) != n:
            raise ValueError(f"candidate {name!r}: {len(matrix)} rows != frame {n}")

    y = frame["y"].to_numpy()
    groups = frame["patient_id"].to_numpy()
    pred_sum = np.zeros(n, dtype=float)
    pred_count = np.zeros(n, dtype=int)
    fold_rows: list[dict] = []

    for repeat in range(repeats):
        outer = StratifiedGroupKFold(
            n_splits=outer_splits, shuffle=True, random_state=seed + repeat
        )
        for outer_fold, (train_idx, test_idx) in enumerate(
            outer.split(np.zeros(n), y, groups=groups)
        ):
            inner_scores: dict[str, float] = {}
            train_frame = frame.iloc[train_idx].reset_index(drop=True)
            train_y = y[train_idx]
            train_groups = groups[train_idx]

            for name in sorted(candidate_matrices):
                matrix = np.asarray(candidate_matrices[name])
                inner_oof = np.full(len(train_idx), np.nan, dtype=float)
                inner = StratifiedGroupKFold(
                    n_splits=inner_splits,
                    shuffle=True,
                    random_state=seed + 1000 + repeat * outer_splits + outer_fold,
                )
                for inner_train, inner_test in inner.split(
                    matrix[train_idx], train_y, groups=train_groups
                ):
                    estimator = estimator_factories[name]()
                    estimator.fit(matrix[train_idx][inner_train], train_y[inner_train])
                    inner_oof[inner_test] = estimator.predict_proba(
                        matrix[train_idx][inner_test]
                    )[:, 1]
                inner_scores[name] = _patient_auc(train_frame, inner_oof, agg=patient_agg)

            selected = max(
                sorted(inner_scores),
                key=lambda name: np.nan_to_num(inner_scores[name], nan=-np.inf),
            )
            selected_matrix = np.asarray(candidate_matrices[selected])
            estimator = estimator_factories[selected]()
            estimator.fit(selected_matrix[train_idx], y[train_idx])
            outer_pred = estimator.predict_proba(selected_matrix[test_idx])[:, 1]
            pred_sum[test_idx] += outer_pred
            pred_count[test_idx] += 1

            train_patients = set(groups[train_idx])
            test_patients = set(groups[test_idx])
            fold_rows.append({
                "repeat": repeat,
                "outer_fold": outer_fold,
                "selected_candidate": selected,
                "selected_inner_auroc": inner_scores[selected],
                "n_train_patients": len(train_patients),
                "n_test_patients": len(test_patients),
                "patient_overlap": len(train_patients & test_patients),
                "inner_scores": inner_scores,
            })

    if np.any(pred_count != repeats):
        raise RuntimeError("each row must receive exactly one outer prediction per repeat")
    scored = frame[["slide_id", "patient_id", "site", "y"]].copy()
    scored["p_msih"] = pred_sum / pred_count
    return scored, pd.DataFrame(fold_rows)
