"""W1 raw-tile smoke: reconstruction, backward pass, and one optimizer step."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from argo_deepmsi.models.wagner import parameter_counts
from argo_deepmsi.w1_peft import (
    MixedFeatureCTransPath,
    RawTileReader,
    transform_raw_tiles,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("bitfit_norm", "last_stage"), required=True)
    parser.add_argument(
        "--w1-dir", type=Path, default="results/experiments/w1_ctranspath"
    )
    parser.add_argument("--raw-tiles", type=int, default=8)
    parser.add_argument("--bag-tiles", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    started = time.monotonic()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    index = pd.read_csv(args.w1_dir / "slide_index.csv")
    cache_dir = args.w1_dir / "bag_cache_uniform_1024"
    cache = np.load(cache_dir / "bags.npy", mmap_mode="r")
    tile_indices = np.load(cache_dir / "tile_indices.npy", mmap_mode="r")
    cache_index = pd.read_csv(cache_dir / "bags_index.csv")
    row = index.iloc[0]
    cache_row = cache_index[cache_index["slide_id"] == row["slide_id"]].iloc[0]
    start = int(cache_row["start"])
    length = min(int(cache_row["length"]), args.bag_tiles)
    raw_count = min(args.raw_tiles, length)
    cached_features = np.array(
        cache[start : start + length], dtype=np.float32, copy=True
    )
    original_indices = np.asarray(
        tile_indices[start : start + raw_count], dtype=np.int64
    )

    model = MixedFeatureCTransPath(args.mode).to(device)
    reader = RawTileReader(index)
    images = reader.read(0, original_indices)
    raw_images = transform_raw_tiles(images, model.transform).to(device)
    cached_tensor = torch.from_numpy(cached_features).unsqueeze(0).to(device)
    positions = torch.arange(raw_count, dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        reencoded = model.ctranspath(raw_images)
        old = cached_tensor[:, :raw_count, :]
        cosine = torch.nn.functional.cosine_similarity(reencoded, old.squeeze(0)).mean()
        initial_logit = model(cached_tensor, raw_images, positions).reshape(-1)

    model.train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-5, weight_decay=1e-4)
    target = torch.tensor([float(row["y"])], device=device)
    optimizer.zero_grad(set_to_none=True)
    logit = model(cached_tensor, raw_images, positions).reshape(-1)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, target)
    loss.backward()
    encoder_gradients = [
        parameter.grad
        for parameter in model.ctranspath.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not encoder_gradients:
        raise RuntimeError("no CTransPath gradient reached a trainable parameter")
    optimizer.step()
    model.eval()
    with torch.no_grad():
        updated_logit = model(cached_tensor, raw_images, positions).reshape(-1)

    output_dir = args.w1_dir / "peft_smoke" / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "one_step_state.pt")
    report = {
        "mode": args.mode,
        "slide_id": str(row["slide_id"]),
        "bag_tiles": length,
        "raw_tiles": raw_count,
        "reencoded_cached_mean_cosine": float(cosine.cpu()),
        "initial_logit": float(initial_logit.cpu()),
        "updated_logit": float(updated_logit.cpu()),
        "absolute_logit_change": float((updated_logit - initial_logit).abs().cpu()),
        "loss": float(loss.detach().cpu()),
        "encoder_gradient_tensors": len(encoder_gradients),
        "parameters": parameter_counts(model),
        "ctranspath_parameters": asdict(model.ctranspath.report),
        "device": str(device),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated())
        if device.type == "cuda"
        else 0,
        "elapsed_seconds": float(time.monotonic() - started),
        "uses_archived_old_tree": False,
    }
    (output_dir / "smoke.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
