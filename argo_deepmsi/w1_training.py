"""Leakage-safe outer-fold runner for W1 cached CTransPath candidates."""

from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from .eval.metrics import (
    aggregate_to_patient,
    evaluate_scorer,
    paired_bootstrap_auroc_delta,
    stratified_patient_bootstrap,
)
from .eval.screening import screening_block
from .models.w1_heads import build_cached_candidate
from .w1 import CachedCTransPathBagStore, DEFAULT_OUTDIR

REFERENCE_SCORES = Path("results/scorers/wagner_zeroshot/slide_scores.csv")


@dataclass
class FoldAudit:
    outer_fold: int
    n_train_patients: int
    n_validation_patients: int
    n_test_patients: int
    n_train_slides: int
    n_validation_slides: int
    n_test_slides: int
    patient_overlap: int
    selected_epochs: int
    best_inner_patient_auroc: float
    elapsed_seconds: float
    validation_history: list[dict[str, float]]


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _patient_split(
    index: pd.DataFrame,
    outer_train_rows: np.ndarray,
    *,
    seed: int,
    n_splits: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    outer = index.iloc[outer_train_rows]
    patient = (
        outer[["patient_id", "y"]]
        .drop_duplicates("patient_id")
        .sort_values("patient_id")
        .reset_index(drop=True)
    )
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_patient_rows, validation_patient_rows = next(
        cv.split(
            np.zeros(len(patient)),
            patient["y"].to_numpy(),
            groups=patient["patient_id"].to_numpy(),
        )
    )
    train_patients = set(patient.iloc[train_patient_rows]["patient_id"])
    validation_patients = set(patient.iloc[validation_patient_rows]["patient_id"])
    train_rows = outer_train_rows[
        index.iloc[outer_train_rows]["patient_id"].isin(train_patients).to_numpy()
    ]
    validation_rows = outer_train_rows[
        index.iloc[outer_train_rows]["patient_id"].isin(validation_patients).to_numpy()
    ]
    return train_rows, validation_rows


def _group_rows_by_patient(index: pd.DataFrame, rows: np.ndarray) -> list[np.ndarray]:
    selected = index.iloc[rows]
    return [
        group.index.to_numpy(dtype=int)
        for _, group in selected.groupby("patient_id", sort=True)
    ]


def _patient_class_weight(index: pd.DataFrame, rows: np.ndarray) -> float:
    labels = index.iloc[rows].drop_duplicates("patient_id")["y"].to_numpy()
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("training fold must contain both patient classes")
    return float(negatives / positives)


def _train_one_epoch(
    model: torch.nn.Module,
    store: CachedCTransPathBagStore,
    index: pd.DataFrame,
    rows: np.ndarray,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    epoch: int,
    seed: int,
) -> float:
    model.train()
    patient_groups = _group_rows_by_patient(index, rows)
    rng = np.random.default_rng(seed + epoch)
    rng.shuffle(patient_groups)
    pos_weight = torch.tensor(
        [_patient_class_weight(index, rows)], dtype=torch.float32, device=device
    )
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    total_loss = 0.0
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for patient_rows in patient_groups:
        optimizer.zero_grad(set_to_none=True)
        patient_loss = 0.0
        for row_index in patient_rows:
            bag = torch.from_numpy(store.read(int(row_index), epoch=epoch))
            bag = bag.unsqueeze(0).to(device, non_blocking=True)
            target = torch.tensor(
                [float(index.iloc[int(row_index)]["y"])],
                dtype=torch.float32,
                device=device,
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logit = model(bag).reshape(-1)
                loss = loss_fn(logit, target) / len(patient_rows)
            scaler.scale(loss).backward()
            patient_loss += float(loss.detach())
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            max_norm=1.0,
        )
        scaler.step(optimizer)
        scaler.update()
        total_loss += patient_loss
    return total_loss / max(1, len(patient_groups))


def _predict_rows(
    model: torch.nn.Module,
    store: CachedCTransPathBagStore,
    index: pd.DataFrame,
    rows: np.ndarray,
    *,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    probabilities = []
    with torch.no_grad():
        for row_index in rows:
            bag = torch.from_numpy(store.read(int(row_index), epoch=-1))
            bag = bag.unsqueeze(0).to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logit = model(bag).reshape(-1)
            probabilities.append(float(torch.sigmoid(logit).cpu().item()))
    output = index.iloc[rows][
        ["slide_id", "patient_id", "site", "y", "outer_fold"]
    ].copy()
    output["p_msih"] = probabilities
    return output.reset_index(drop=True)


def _patient_auroc(slides: pd.DataFrame) -> float:
    patient = aggregate_to_patient(slides, "p_msih", agg="max_sqrtn")
    if patient["y"].nunique() != 2:
        return float("nan")
    return float(roc_auc_score(patient["y"], patient["score"]))


def _optimizer(model: torch.nn.Module, learning_rate: float, weight_decay: float):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("candidate has no trainable parameters")
    return torch.optim.AdamW(
        parameters, lr=learning_rate, weight_decay=weight_decay
    )


def _select_epoch(
    candidate_id: str,
    train_store: CachedCTransPathBagStore,
    eval_store: CachedCTransPathBagStore,
    index: pd.DataFrame,
    train_rows: np.ndarray,
    validation_rows: np.ndarray,
    *,
    device: torch.device,
    seed: int,
) -> tuple[int, float, list[dict[str, float]]]:
    _seed_everything(seed)
    model, spec, _ = build_cached_candidate(candidate_id, device=device)
    optimizer = _optimizer(model, spec.learning_rate, spec.weight_decay)
    best_auroc = -np.inf
    best_epoch = 1
    stale = 0
    history = []
    for epoch in range(1, spec.max_epochs + 1):
        train_loss = _train_one_epoch(
            model,
            train_store,
            index,
            train_rows,
            optimizer,
            device=device,
            epoch=epoch,
            seed=seed,
        )
        validation = _predict_rows(
            model, eval_store, index, validation_rows, device=device
        )
        validation_auroc = _patient_auroc(validation)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "validation_patient_auroc": float(validation_auroc),
            }
        )
        if np.isfinite(validation_auroc) and validation_auroc > best_auroc + 1e-4:
            best_auroc = validation_auroc
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if epoch >= 4 and stale >= spec.patience:
            break
    return best_epoch, float(best_auroc), history


def _fit_outer_fold(
    candidate_id: str,
    train_store: CachedCTransPathBagStore,
    eval_store: CachedCTransPathBagStore,
    index: pd.DataFrame,
    outer_fold: int,
    *,
    device: torch.device,
    seed: int,
    checkpoint_dir: Path,
) -> tuple[pd.DataFrame, FoldAudit, dict[str, int]]:
    started = time.monotonic()
    outer_train_rows = np.flatnonzero(index["outer_fold"].to_numpy() != outer_fold)
    test_rows = np.flatnonzero(index["outer_fold"].to_numpy() == outer_fold)
    train_rows, validation_rows = _patient_split(
        index, outer_train_rows, seed=seed + outer_fold
    )
    selected_epochs, inner_auroc, history = _select_epoch(
        candidate_id,
        train_store,
        eval_store,
        index,
        train_rows,
        validation_rows,
        device=device,
        seed=seed + 100 * outer_fold,
    )

    _seed_everything(seed + 10_000 + outer_fold)
    model, spec, counts = build_cached_candidate(candidate_id, device=device)
    optimizer = _optimizer(model, spec.learning_rate, spec.weight_decay)
    for epoch in range(1, selected_epochs + 1):
        _train_one_epoch(
            model,
            train_store,
            index,
            outer_train_rows,
            optimizer,
            device=device,
            epoch=epoch,
            seed=seed + 10_000 + outer_fold,
        )
    predictions = _predict_rows(model, eval_store, index, test_rows, device=device)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "candidate_id": candidate_id,
            "outer_fold": outer_fold,
            "selected_epochs": selected_epochs,
            "spec": spec.to_dict(),
            "state_dict": copy.deepcopy(model.state_dict()),
        },
        checkpoint_dir / f"outer_fold_{outer_fold}.pt",
    )

    train_patients = set(index.iloc[outer_train_rows]["patient_id"])
    validation_patients = set(index.iloc[validation_rows]["patient_id"])
    test_patients = set(index.iloc[test_rows]["patient_id"])
    audit = FoldAudit(
        outer_fold=outer_fold,
        n_train_patients=len(train_patients),
        n_validation_patients=len(validation_patients),
        n_test_patients=len(test_patients),
        n_train_slides=int(len(outer_train_rows)),
        n_validation_slides=int(len(validation_rows)),
        n_test_slides=int(len(test_rows)),
        patient_overlap=len(train_patients & test_patients),
        selected_epochs=selected_epochs,
        best_inner_patient_auroc=inner_auroc,
        elapsed_seconds=float(time.monotonic() - started),
        validation_history=history,
    )
    return predictions, audit, counts


def _metrics(
    predictions: pd.DataFrame,
    *,
    candidate_id: str,
    reference_path: str | Path = REFERENCE_SCORES,
) -> tuple[dict, pd.DataFrame]:
    metrics = evaluate_scorer(predictions, "p_msih", agg="max_sqrtn")
    patient = aggregate_to_patient(predictions, "p_msih", agg="max_sqrtn")
    metrics["patient_bootstrap"] = stratified_patient_bootstrap(patient)
    metrics["screening"] = screening_block(
        patient["y"].to_numpy(), patient["score"].to_numpy()
    )
    metrics["candidate_id"] = candidate_id
    metrics["validation_design"] = (
        "fixed candidate; 5-fold patient-grouped outer OOF; fold-local inner "
        "early-stopping epoch; refit on complete outer training fold"
    )
    metrics["confirmatory_valid"] = False
    metrics["confirmatory_note"] = (
        "Each candidate has untouched outer predictions, but choosing the best "
        "W1 candidate on this development cohort makes the selected maximum exploratory."
    )

    reference = pd.read_csv(reference_path)
    reference = reference[
        reference["slide_id"].astype(str).isin(set(predictions["slide_id"].astype(str)))
    ].copy()
    reference_patient = aggregate_to_patient(reference, "p_msih", agg="max_sqrtn")
    metrics["paired_delta_vs_frozen_wagner"] = paired_bootstrap_auroc_delta(
        patient, reference_patient
    )
    return metrics, patient


def run_cached_candidate(
    candidate_id: str,
    *,
    w1_dir: str | Path = DEFAULT_OUTDIR,
    device: str | None = None,
    seed: int = 42,
) -> Path:
    """Run one W1 cached-feature candidate across the immutable outer folds."""
    w1_dir = Path(w1_dir)
    index = pd.read_csv(w1_dir / "slide_index.csv")
    cache_dir = w1_dir / "bag_cache_uniform_1024"
    _, spec, _ = build_cached_candidate(candidate_id, device="cpu")
    train_store = CachedCTransPathBagStore(
        cache_dir / "bags.npy",
        cache_dir / "bags_index.csv",
        index,
        max_tiles=spec.train_tile_cap,
        seed=seed,
    )
    eval_store = CachedCTransPathBagStore(
        cache_dir / "bags.npy",
        cache_dir / "bags_index.csv",
        index,
        max_tiles=spec.eval_tile_cap,
        seed=seed,
    )
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    try:
        torch.set_num_threads(4)
    except RuntimeError:
        pass

    outdir = w1_dir / "candidates" / candidate_id
    outdir.mkdir(parents=True, exist_ok=True)
    predictions = []
    audits = []
    parameter_count = None
    for outer_fold in sorted(index["outer_fold"].unique()):
        fold_predictions, audit, counts = _fit_outer_fold(
            candidate_id,
            train_store,
            eval_store,
            index,
            int(outer_fold),
            device=torch_device,
            seed=seed,
            checkpoint_dir=outdir / "checkpoints",
        )
        predictions.append(fold_predictions)
        audits.append(asdict(audit))
        parameter_count = counts
        print(
            f"{candidate_id} fold {int(outer_fold)}: "
            f"inner AUROC={audit.best_inner_patient_auroc:.3f}, "
            f"epochs={audit.selected_epochs}, seconds={audit.elapsed_seconds:.1f}",
            flush=True,
        )
    oof = pd.concat(predictions, ignore_index=True).sort_values("slide_id")
    if len(oof) != len(index) or oof["slide_id"].duplicated().any():
        raise RuntimeError("outer folds did not produce exactly one prediction per W1 slide")
    oof.to_csv(outdir / "slide_scores.csv", index=False)
    metrics, patient = _metrics(oof, candidate_id=candidate_id)
    patient.to_csv(outdir / "patient_scores.csv", index=False)
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (outdir / "fold_audit.json").write_text(json.dumps(audits, indent=2))
    metadata = {
        "candidate": spec.to_dict(),
        "parameter_count": parameter_count,
        "seed": seed,
        "device": str(torch_device),
        "fold_contract": json.loads((w1_dir / "fold_contract.json").read_text()),
        "runtime_dependencies": {
            "wagner": "argo_deepmsi.models.wagner",
            "ctranspath_features": "current Zarr tables/ctranspath_tiles",
            "archived_old_tree": False,
        },
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return outdir
