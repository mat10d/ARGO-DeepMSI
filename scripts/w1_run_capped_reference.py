"""Evaluate frozen Wagner on the W1 1024-tile reservoir as a sampling control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from argo_deepmsi.models.wagner import load_wagner, parameter_counts
from argo_deepmsi.w1 import CachedCTransPathBagStore
from argo_deepmsi.w1_training import _metrics, _predict_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--w1-dir", type=Path, default="results/experiments/w1_ctranspath"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    index = pd.read_csv(args.w1_dir / "slide_index.csv")
    cache_dir = args.w1_dir / "bag_cache_uniform_1024"
    store = CachedCTransPathBagStore(
        cache_dir / "bags.npy",
        cache_dir / "bags_index.csv",
        index,
        max_tiles=1024,
        seed=42,
    )
    model = load_wagner(device=device)
    rows = np.arange(len(index))
    predictions = _predict_rows(model, store, index, rows, device=device)
    outdir = args.w1_dir / "candidates" / "W1-0c"
    outdir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(outdir / "slide_scores.csv", index=False)
    metrics, patient = _metrics(predictions, candidate_id="W1-0c")
    metrics["validation_design"] = "frozen sampling control; no fitting"
    metrics["confirmatory_valid"] = True
    metrics["confirmatory_note"] = "No labels or cohort outcomes influence this scorer."
    patient.to_csv(outdir / "patient_scores.csv", index=False)
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    metadata = {
        "candidate_id": "W1-0c",
        "description": "Frozen Wagner on deterministic 1024-tile reservoirs",
        "tile_cap": 1024,
        "parameter_count": parameter_counts(model),
        "uses_archived_old_tree": False,
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(
        f"W1-0c patient AUROC={metrics['patient_auroc']:.3f}; "
        f"paired delta vs full Wagner="
        f"{metrics['paired_delta_vs_frozen_wagner']['estimate']:+.3f}"
    )


if __name__ == "__main__":
    main()
