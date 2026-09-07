#!/bin/bash
#SBATCH --job-name=mdiberna_waiv_smoke
#SBATCH --output=scripts/logs/waiv_smoke_%j.out
#SBATCH --time=00:20:00
#SBATCH --partition=nvidia-A6000-20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail
source /lab/barcheese01/mdiberna/miniconda3/etc/profile.d/conda.sh
conda activate argo
cd /lab/barcheese01/mdiberna/ARGO-DeepMSI

python - <<'PY'
import torch
import argo_deepmsi.models  # registers phaet + mascaret into LazySlide MODEL_REGISTRY
from lazyslide.models._model_registry import MODEL_REGISTRY

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev, "| cuda:", torch.cuda.get_device_name(0) if dev == "cuda" else "n/a")

for key, exp_dim in [("phaet", 1024), ("mascaret", 1536)]:
    model = MODEL_REGISTRY[key]()   # entry IS the class; instantiate (downloads gated weights)
    tfm = model.get_transform()
    # a fake H&E-ish tile
    img = (torch.rand(1, 3, 256, 256) * 255).to(torch.uint8)
    x = tfm(img).unsqueeze(0).to(dev) if tfm(img).ndim == 3 else tfm(img).to(dev)
    model.model.to(dev).eval()
    out = model.encode_image(x)
    print(f"{key}: encode_image -> {tuple(out.shape)} (expected D={exp_dim})  OK={out.shape[-1]==exp_dim}")
print("WAIV_SMOKE_OK")
PY
