"""Patient-grouped W1 runner for true raw-tile CTransPath adaptation."""

from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .models.wagner import parameter_counts
from .w1 import DEFAULT_OUTDIR, _stable_seed
from .w1_peft import (
    FixedRawTileStore,
    MixedBagStore,
    MixedFeatureCTransPath,
    RawTileReader,
    fixed_raw_tile_cache_ready,
    transform_raw_tiles,
)
from .w1_training import _metrics, _patient_class_weight, _patient_split, _seed_everything


@dataclass(frozen=True)
class PEFTSpec:
    candidate_id: str
    ctranspath_mode: str
    description: str
    encoder_learning_rate: float
    head_learning_rate: float = 5e-5
    weight_decay: float = 1e-4
    max_epochs: int = 6
    patience: int = 2
    train_bag_tiles: int = 256
    eval_bag_tiles: int = 1024
    raw_tiles_per_slide: int = 8


PEFT_CANDIDATES = {
    "W1-7": PEFTSpec(
        candidate_id="W1-7",
        ctranspath_mode="bitfit_norm",
        description="CTransPath BitFit/normalization plus full warm-started Wagner",
        encoder_learning_rate=1e-5,
    ),
    "W1-9": PEFTSpec(
        candidate_id="W1-9",
        ctranspath_mode="last_stage",
        description="CTransPath final stage plus full warm-started Wagner",
        encoder_learning_rate=2e-6,
    ),
}


def _optimizer(model: MixedFeatureCTransPath, spec: PEFTSpec):
    encoder = [
        parameter
        for parameter in model.ctranspath.parameters()
        if parameter.requires_grad
    ]
    head = [
        parameter for parameter in model.wagner.parameters() if parameter.requires_grad
    ]
    if not encoder or not head:
        raise ValueError("PEFT candidate requires trainable encoder and Wagner parameters")
    return torch.optim.AdamW(
        [
            {"params": encoder, "lr": spec.encoder_learning_rate},
            {"params": head, "lr": spec.head_learning_rate},
        ],
        weight_decay=spec.weight_decay,
    )


def _one_slide_per_patient(
    index: pd.DataFrame, rows: np.ndarray, *, seed: int, epoch: int
) -> np.ndarray:
    selected = []
    for patient_id, group in index.iloc[rows].groupby("patient_id", sort=True):
        candidates = group.index.to_numpy(dtype=int)
        rng = np.random.default_rng(_stable_seed(str(patient_id), seed, epoch))
        selected.append(int(rng.choice(candidates)))
    return np.asarray(selected, dtype=int)


def _forward_row(
    model: MixedFeatureCTransPath,
    store: MixedBagStore | FixedRawTileStore,
    reader: RawTileReader | None,
    row_index: int,
    *,
    epoch: int,
    bag_tiles: int,
    raw_tiles: int,
    device: torch.device,
) -> torch.Tensor:
    sample = store.read(
        row_index,
        epoch=epoch,
        bag_tiles=bag_tiles,
        raw_tiles=raw_tiles,
    )
    if hasattr(sample, "raw_images"):
        images = sample.raw_images
    elif reader is not None:
        images = reader.read(row_index, sample.source_tile_indices)
    else:
        raise RuntimeError("on-demand PEFT store requires a RawTileReader")
    raw = transform_raw_tiles(images, model.transform).to(device, non_blocking=True)
    cached = torch.from_numpy(sample.cached_features).unsqueeze(0)
    cached = cached.to(device, non_blocking=True)
    positions = torch.from_numpy(sample.raw_positions).to(device=device, dtype=torch.long)
    return model(cached, raw, positions).reshape(-1)


def _train_one_epoch(
    model: MixedFeatureCTransPath,
    store: MixedBagStore | FixedRawTileStore,
    reader: RawTileReader | None,
    index: pd.DataFrame,
    rows: np.ndarray,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    spec: PEFTSpec,
    *,
    device: torch.device,
    seed: int,
    epoch: int,
) -> float:
    model.train()
    selected = _one_slide_per_patient(index, rows, seed=seed, epoch=epoch)
    rng = np.random.default_rng(seed + epoch)
    rng.shuffle(selected)
    pos_weight = torch.tensor(
        [_patient_class_weight(index, rows)], dtype=torch.float32, device=device
    )
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    total = 0.0
    for row_index in selected:
        optimizer.zero_grad(set_to_none=True)
        target = torch.tensor(
            [float(index.iloc[int(row_index)]["y"])], device=device
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logit = _forward_row(
                model,
                store,
                reader,
                int(row_index),
                epoch=epoch,
                bag_tiles=spec.train_bag_tiles,
                raw_tiles=spec.raw_tiles_per_slide,
                device=device,
            )
            loss = loss_fn(logit, target)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            max_norm=1.0,
        )
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.detach())
    return total / max(1, len(selected))


def _predict_rows(
    model: MixedFeatureCTransPath,
    store: MixedBagStore | FixedRawTileStore,
    reader: RawTileReader | None,
    index: pd.DataFrame,
    rows: np.ndarray,
    spec: PEFTSpec,
    *,
    device: torch.device,
    representative_only: bool,
) -> pd.DataFrame:
    model.eval()
    if representative_only:
        rows = _one_slide_per_patient(index, rows, seed=42, epoch=-1)
    probabilities = []
    with torch.no_grad():
        for row_index in rows:
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logit = _forward_row(
                    model,
                    store,
                    reader,
                    int(row_index),
                    epoch=-1,
                    bag_tiles=spec.eval_bag_tiles,
                    raw_tiles=spec.raw_tiles_per_slide,
                    device=device,
                )
            probabilities.append(float(torch.sigmoid(logit).cpu().item()))
    output = index.iloc[rows][
        ["slide_id", "patient_id", "site", "y", "outer_fold"]
    ].copy()
    output["p_msih"] = probabilities
    return output.reset_index(drop=True)


def _patient_auroc(slides: pd.DataFrame) -> float:
    from sklearn.metrics import roc_auc_score

    if slides["y"].nunique() != 2:
        return float("nan")
    return float(roc_auc_score(slides["y"], slides["p_msih"]))


def _new_model(spec: PEFTSpec, device: torch.device) -> MixedFeatureCTransPath:
    model = MixedFeatureCTransPath(
        spec.ctranspath_mode,
        wagner_mode="full",
    )
    return model.to(device)


def _select_epoch(
    spec: PEFTSpec,
    store: MixedBagStore | FixedRawTileStore,
    reader: RawTileReader | None,
    index: pd.DataFrame,
    train_rows: np.ndarray,
    validation_rows: np.ndarray,
    *,
    device: torch.device,
    seed: int,
) -> tuple[int, float, list[dict[str, float]]]:
    _seed_everything(seed)
    model = _new_model(spec, device)
    optimizer = _optimizer(model, spec)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_auroc = -np.inf
    best_epoch = 1
    stale = 0
    history = []
    for epoch in range(1, spec.max_epochs + 1):
        loss = _train_one_epoch(
            model,
            store,
            reader,
            index,
            train_rows,
            optimizer,
            scaler,
            spec,
            device=device,
            seed=seed,
            epoch=epoch,
        )
        validation = _predict_rows(
            model,
            store,
            reader,
            index,
            validation_rows,
            spec,
            device=device,
            representative_only=True,
        )
        auroc = _patient_auroc(validation)
        history.append(
            {"epoch": epoch, "train_loss": loss, "validation_patient_auroc": auroc}
        )
        print(
            f"{spec.candidate_id} inner epoch {epoch}: loss={loss:.4f}, "
            f"patient AUROC={auroc:.3f}",
            flush=True,
        )
        if np.isfinite(auroc) and auroc > best_auroc + 1e-4:
            best_auroc = auroc
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if epoch >= 3 and stale >= spec.patience:
            break
    return best_epoch, float(best_auroc), history


def _fit_fold(
    spec: PEFTSpec,
    store: MixedBagStore | FixedRawTileStore,
    reader: RawTileReader | None,
    index: pd.DataFrame,
    outer_fold: int,
    *,
    device: torch.device,
    seed: int,
    checkpoint_dir: Path,
) -> tuple[pd.DataFrame, dict, dict[str, int]]:
    started = time.monotonic()
    outer_train = np.flatnonzero(index["outer_fold"].to_numpy() != outer_fold)
    test = np.flatnonzero(index["outer_fold"].to_numpy() == outer_fold)
    train, validation = _patient_split(
        index, outer_train, seed=seed + outer_fold
    )
    epochs, inner_auroc, history = _select_epoch(
        spec,
        store,
        reader,
        index,
        train,
        validation,
        device=device,
        seed=seed + 100 * outer_fold,
    )

    _seed_everything(seed + 10_000 + outer_fold)
    model = _new_model(spec, device)
    optimizer = _optimizer(model, spec)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    for epoch in range(1, epochs + 1):
        _train_one_epoch(
            model,
            store,
            reader,
            index,
            outer_train,
            optimizer,
            scaler,
            spec,
            device=device,
            seed=seed + 10_000 + outer_fold,
            epoch=epoch,
        )
    predictions = _predict_rows(
        model,
        store,
        reader,
        index,
        test,
        spec,
        device=device,
        representative_only=False,
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "candidate_id": spec.candidate_id,
            "outer_fold": outer_fold,
            "selected_epochs": epochs,
            "spec": asdict(spec),
            "state_dict": copy.deepcopy(model.state_dict()),
        },
        checkpoint_dir / f"outer_fold_{outer_fold}.pt",
    )
    train_patients = set(index.iloc[outer_train]["patient_id"])
    test_patients = set(index.iloc[test]["patient_id"])
    audit = {
        "outer_fold": outer_fold,
        "n_train_patients": len(train_patients),
        "n_validation_patients": index.iloc[validation]["patient_id"].nunique(),
        "n_test_patients": len(test_patients),
        "n_train_slides": len(outer_train),
        "n_validation_slides": len(validation),
        "n_test_slides": len(test),
        "patient_overlap": len(train_patients & test_patients),
        "selected_epochs": epochs,
        "best_inner_patient_auroc": inner_auroc,
        "elapsed_seconds": time.monotonic() - started,
        "validation_history": history,
    }
    return predictions, audit, parameter_counts(model)


def run_peft_candidate(
    candidate_id: str,
    *,
    w1_dir: str | Path = DEFAULT_OUTDIR,
    device: str | None = None,
    seed: int = 42,
) -> Path:
    if candidate_id not in PEFT_CANDIDATES:
        raise ValueError(f"choose one of {sorted(PEFT_CANDIDATES)}")
    spec = PEFT_CANDIDATES[candidate_id]
    w1_dir = Path(w1_dir)
    index = pd.read_csv(w1_dir / "slide_index.csv")
    feature_cache = w1_dir / "bag_cache_uniform_1024"
    raw_cache = w1_dir / f"raw_tile_cache_{spec.raw_tiles_per_slide}"
    if fixed_raw_tile_cache_ready(raw_cache):
        store = FixedRawTileStore(feature_cache, raw_cache, index, seed=seed)
        reader = None
    else:
        store = MixedBagStore(feature_cache, index, seed=seed)
        reader = RawTileReader(index, open_slide_cache=16)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    outdir = w1_dir / "candidates" / candidate_id
    outdir.mkdir(parents=True, exist_ok=True)
    fold_predictions = []
    audits = []
    counts = None
    for outer_fold in sorted(index["outer_fold"].unique()):
        predictions, audit, counts = _fit_fold(
            spec,
            store,
            reader,
            index,
            int(outer_fold),
            device=torch_device,
            seed=seed,
            checkpoint_dir=outdir / "checkpoints",
        )
        fold_predictions.append(predictions)
        audits.append(audit)
        print(
            f"{candidate_id} fold {outer_fold}: inner AUROC="
            f"{audit['best_inner_patient_auroc']:.3f}, "
            f"epochs={audit['selected_epochs']}, "
            f"seconds={audit['elapsed_seconds']:.1f}",
            flush=True,
        )
    oof = pd.concat(fold_predictions, ignore_index=True).sort_values("slide_id")
    if len(oof) != len(index) or oof["slide_id"].duplicated().any():
        raise RuntimeError("PEFT outer folds did not cover each W1 slide exactly once")
    oof.to_csv(outdir / "slide_scores.csv", index=False)
    metrics, patient = _metrics(oof, candidate_id=candidate_id)
    metrics["peft_design"] = asdict(spec)
    patient.to_csv(outdir / "patient_scores.csv", index=False)
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (outdir / "fold_audit.json").write_text(json.dumps(audits, indent=2))
    metadata = {
        "candidate": asdict(spec),
        "parameter_count": counts,
        "seed": seed,
        "device": str(torch_device),
        "fold_contract": json.loads((w1_dir / "fold_contract.json").read_text()),
        "runtime_dependencies": {
            "wagner": "argo_deepmsi.models.wagner",
            "ctranspath": "argo_deepmsi.models.ctranspath",
            "archived_old_tree": False,
            "fixed_raw_tile_cache": isinstance(store, FixedRawTileStore),
        },
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return outdir
