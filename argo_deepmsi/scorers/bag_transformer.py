"""Nested-CV transformer top layer for arbitrary frozen tile-feature bags."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from ..eval.metrics import aggregate_to_patient
from .base import Scorer, ScoreColumn
from .clam_tilemil import _load_bags
from .registry import register


def _stable_seed(slide_id: str, seed: int, epoch: int) -> int:
    payload = f"{slide_id}\0{seed}\0{epoch}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


class _ArchiveBagStore:
    def __init__(
        self, bags: list[np.ndarray], index: pd.DataFrame, tile_cap: int | None, seed: int
    ):
        self.bags = bags
        self.index = index
        self.tile_cap = tile_cap
        self.seed = seed
        self.input_dim = bags[0].shape[1]

    def read(self, row: int, epoch: int) -> np.ndarray:
        return self.read_with_indices(row, epoch)[0]

    def read_with_indices(self, row: int, epoch: int) -> tuple[np.ndarray, np.ndarray]:
        bag = self.bags[row]
        if self.tile_cap is None or len(bag) <= self.tile_cap:
            return bag, np.arange(len(bag))
        rng = np.random.default_rng(
            _stable_seed(str(self.index.iloc[row]["slide_id"]), self.seed, epoch)
        )
        selected = np.sort(rng.choice(len(bag), self.tile_cap, replace=False))
        return bag[selected], selected


class _ZarrBagStore:
    def __init__(
        self,
        index: pd.DataFrame,
        feature_key: str,
        zarr_column: str,
        tile_cap: int | None,
        seed: int,
    ):
        self.index = index
        self.feature_key = feature_key
        self.zarr_column = zarr_column
        self.tile_cap = tile_cap
        self.seed = seed
        self.input_dim = int(self._array_path(index.iloc[0]).shape[1])

    def _array_path(self, row: pd.Series):
        import zarr

        path = Path(str(row[self.zarr_column]))
        if path.name.endswith("_tiles"):
            path = path.with_name(f"{self.feature_key}_tiles")
        elif path.name.endswith(".zarr"):
            path = path / "tables" / f"{self.feature_key}_tiles"
        else:
            raise ValueError(
                f"{self.zarr_column} must point to a .zarr or *_tiles array, got {path}"
            )
        return zarr.open_array(path, mode="r")

    def read(self, row: int, epoch: int) -> np.ndarray:
        return self.read_with_indices(row, epoch)[0]

    def read_with_indices(self, row: int, epoch: int) -> tuple[np.ndarray, np.ndarray]:
        array = self._array_path(self.index.iloc[row])
        if int(array.shape[1]) != self.input_dim:
            raise ValueError("Feature dimension differs across zarr bags")
        selected = np.arange(int(array.shape[0]))
        if self.tile_cap is not None and len(selected) > self.tile_cap:
            rng = np.random.default_rng(
                _stable_seed(str(self.index.iloc[row]["slide_id"]), self.seed, epoch)
            )
            selected = np.sort(rng.choice(selected, self.tile_cap, replace=False))
        return np.asarray(array.oindex[selected, :], dtype=np.float32), selected


def _patient_auroc(frame: pd.DataFrame, scores: np.ndarray, rows: np.ndarray, agg: str) -> float:
    slides = frame.iloc[rows].copy()
    slides["p_msih"] = scores
    patients = aggregate_to_patient(slides, "p_msih", agg=agg)
    if patients["y"].nunique() != 2:
        return float("nan")
    return float(roc_auc_score(patients["y"], patients["score"]))


class BagTransformer(Scorer):
    """Train a Wagner-style slide transformer from scratch on any bag archive."""

    name = "bag_transformer"
    description = (
        "Configurable Wagner-style projection/CLS/transformer top layer trained from "
        "scratch on frozen tile-feature bags. Outer predictions use patient-grouped CV; "
        "checkpoint selection uses a patient-grouped inner split. The encoder, bag cap, "
        "architecture, optimizer, and fold contract are all run parameters."
    )
    needs_training_on_our_data = True
    resolution = "slide"
    patient_aggregation = "max_sqrtn"
    score_columns = [ScoreColumn("p_msih", "nested-CV transformer p(MSI-H)", primary=True)]

    def write_run_artifacts(self, outdir: Path) -> None:
        outdir.mkdir(parents=True, exist_ok=True)
        if hasattr(self, "_last_fold_audit"):
            self._last_fold_audit.to_csv(outdir / "fold_audit.csv", index=False)
        if getattr(self, "_last_attention", None):
            records = sorted(self._last_attention, key=lambda record: record["row"])
            np.savez_compressed(
                outdir / "attention.npz",
                slide_ids=np.asarray([record["slide_id"] for record in records]),
                lengths=np.asarray([len(record["weights"]) for record in records], dtype=np.int64),
                tile_indices=np.concatenate([record["tile_indices"] for record in records]),
                weights=np.concatenate([record["weights"] for record in records]),
            )

    def compute_batch(
        self,
        slide_table: pd.DataFrame,
        *,
        clean_slide_ids: set[str] | None = None,
        bag_file: str | Path | None = None,
        index_file: str | Path | None = None,
        feature_key: str | None = None,
        zarr_column: str = "zarr_path",
        tile_cap: int | None = None,
        clinical_table: str | Path = "results/data/clinical_table.csv",
        epochs: int = 20,
        device: str = "cuda",
        seed: int = 42,
        outer_splits: int = 5,
        inner_splits: int = 5,
        fold_column: str | None = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        grad_clip: float = 1.0,
        dim: int = 512,
        depth: int = 2,
        heads: int = 8,
        mlp_dim: int = 512,
        dim_head: int = 64,
        dropout: float = 0.0,
        embedding_dropout: float = 0.0,
        deterministic: bool = True,
        capture_attention: bool = False,
        write_outputs: bool = True,
        **_,
    ) -> pd.DataFrame:
        import torch
        import torch.nn as nn

        from ..models.wagner import (
            WagnerTransformer,
            capture_attention as capture_wagner_attention,
        )

        if epochs < 1 or outer_splits < 2 or inner_splits < 2:
            raise ValueError("epochs must be positive and CV split counts must be at least two")
        if dim != heads * dim_head:
            raise ValueError("dim must equal heads * dim_head for WagnerTransformer")
        if bag_file is not None:
            loaded = _load_bags(clean_slide_ids, bag_file=bag_file, index_file=index_file)
            if loaded is None:
                raise FileNotFoundError(f"No usable bags at {bag_file}")
            bags, index = loaded
        else:
            if index_file is None or feature_key is None:
                raise ValueError("Provide bag_file, or provide index_file plus feature_key")
            index = pd.read_csv(index_file)
            if clean_slide_ids is not None:
                index = index[index["slide_id"].astype(str).isin(clean_slide_ids)]
            index = index.drop_duplicates("slide_id").reset_index(drop=True)
            bags = None
        if "y" not in index:
            clinical = pd.read_csv(clinical_table)[["PATIENT", "isMSIH"]]
            clinical["y"] = (clinical["isMSIH"] == "MSI-H").astype(int)
            index = index.merge(
                clinical[["PATIENT", "y"]],
                left_on="patient_id",
                right_on="PATIENT",
                how="inner",
            )
        required = {"slide_id", "patient_id", "y"}
        missing = required - set(index)
        if missing:
            raise ValueError(f"Bag index is missing columns: {sorted(missing)}")
        inconsistent = index.groupby("patient_id")["y"].nunique()
        if (inconsistent > 1).any():
            patients = inconsistent[inconsistent > 1].index.astype(str).tolist()
            raise ValueError(f"Patients have inconsistent labels: {patients[:10]}")
        if bags is not None and len(index) != len(bags):
            raise ValueError("Bag labels are not aligned with the bag archive")
        if bags is None:
            if zarr_column not in index:
                raise ValueError(f"Bag index has no zarr column {zarr_column!r}")
            store = _ZarrBagStore(index, feature_key, zarr_column, tile_cap, seed)
        else:
            store = _ArchiveBagStore(bags, index, tile_cap, seed)

        target = torch.device(device)
        if target.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
        y = index["y"].to_numpy(dtype=int)
        groups = index["patient_id"].astype(str).to_numpy()
        input_dim = store.input_dim

        if fold_column is not None:
            if fold_column not in index:
                raise ValueError(f"Bag index has no fold column {fold_column!r}")
            if index[fold_column].isna().any():
                raise ValueError(f"Bag index has missing values in {fold_column!r}")
            folds = [
                (
                    np.flatnonzero(index[fold_column].to_numpy() != fold),
                    np.flatnonzero(index[fold_column].to_numpy() == fold),
                )
                for fold in sorted(index[fold_column].unique())
            ]
            for fold, (train, test) in enumerate(folds):
                overlap = set(groups[train]) & set(groups[test])
                if overlap:
                    raise ValueError(
                        f"{fold_column!r} leaks {len(overlap)} patients in fold {fold}"
                    )
        else:
            cv = StratifiedGroupKFold(n_splits=outer_splits, shuffle=True, random_state=seed)
            folds = list(cv.split(np.zeros(len(index)), y, groups=groups))

        oof = np.full(len(index), np.nan, dtype=np.float32)
        audit = []
        attention_records: dict[int, dict] = {}
        for fold, (train_all, test) in enumerate(folds):
            inner_cv = StratifiedGroupKFold(
                n_splits=inner_splits, shuffle=True, random_state=seed + fold
            )
            train_y = y[train_all]
            train_groups = groups[train_all]
            inner_train_rel, validation_rel = next(
                inner_cv.split(np.zeros(len(train_all)), train_y, groups=train_groups)
            )
            train = train_all[inner_train_rel]
            validation = train_all[validation_rel]

            torch.manual_seed(seed + 10_000 + fold)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed + 10_000 + fold)
            model = WagnerTransformer(
                input_dim=input_dim,
                dim=dim,
                depth=depth,
                heads=heads,
                mlp_dim=mlp_dim,
                dim_head=dim_head,
                dropout=dropout,
                emb_dropout=embedding_dropout,
            ).to(target)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
            train_patient_labels = index.iloc[train].drop_duplicates("patient_id")["y"].to_numpy()
            pos_weight = float(
                (train_patient_labels == 0).sum() / max(1, (train_patient_labels == 1).sum())
            )
            loss_function = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_weight], device=target)
            )
            rng = np.random.default_rng(seed + fold)
            best_auc = -np.inf
            best_test = None
            best_epoch = None
            for epoch in range(1, epochs + 1):
                model.train()
                order = np.array(train, copy=True)
                rng.shuffle(order)
                for row in order:
                    optimizer.zero_grad()
                    bag = torch.from_numpy(store.read(row, epoch)).unsqueeze(0).to(target)
                    logit = model(bag).reshape(-1)
                    label = torch.tensor([float(y[row])], device=target)
                    loss_function(logit, label).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

                def predict(rows: np.ndarray, capture: bool = False):
                    model.eval()
                    values = []
                    captured = []
                    with torch.no_grad():
                        for row in rows:
                            array, tile_indices = store.read_with_indices(row, -1)
                            bag = torch.from_numpy(array).unsqueeze(0).to(target)
                            if capture:
                                with capture_wagner_attention(model):
                                    logit = model(bag).reshape(-1)
                                captured.append(
                                    {
                                        "row": int(row),
                                        "slide_id": str(index.iloc[row]["slide_id"]),
                                        "tile_indices": tile_indices.astype(np.int64),
                                        "weights": model.last_cls_attn.cpu()
                                        .numpy()
                                        .astype(np.float32),
                                    }
                                )
                            else:
                                logit = model(bag).reshape(-1)
                            values.append(torch.sigmoid(logit).item())
                    return np.asarray(values, dtype=np.float32), captured

                validation_scores, _ = predict(validation)
                validation_auc = _patient_auroc(
                    index, validation_scores, validation, self.patient_aggregation
                )
                if not np.isnan(validation_auc) and validation_auc > best_auc:
                    best_auc = validation_auc
                    best_epoch = epoch
                    best_test, best_attention = predict(test, capture=capture_attention)
            if best_test is None:
                best_test, best_attention = predict(test, capture=capture_attention)
                best_epoch = epochs
            oof[test] = best_test
            for record in best_attention:
                attention_records[record["row"]] = record
            audit.append(
                {
                    "fold": fold,
                    "best_epoch": best_epoch,
                    "inner_patient_auroc": best_auc,
                    "n_train": len(train),
                    "n_validation": len(validation),
                    "n_test": len(test),
                }
            )

        columns = [column for column in ("slide_id", "patient_id", "site", "y") if column in index]
        scores = index[columns].copy()
        scores["p_msih"] = oof
        if scores["p_msih"].isna().any():
            raise RuntimeError("Outer folds did not produce exactly one score for every slide")
        scores = scores.dropna(subset=["p_msih"]).reset_index(drop=True)
        self._last_fold_audit = pd.DataFrame(audit)
        self._last_attention = list(attention_records.values())
        if write_outputs and self.score_path is not None:
            self.score_path.parent.mkdir(parents=True, exist_ok=True)
            scores.to_csv(self.score_path, index=False)
            self._last_fold_audit.to_csv(self.score_path.parent / "fold_audit.csv", index=False)
        return scores


register("bag_transformer", BagTransformer)
