"""Training adapter for the current LazySlide CTransPath implementation.

Feature extraction and W1 fine-tuning use the same registered LazySlide model,
preprocessing, and Hugging Face weight source.  This module deliberately has no
dependency on the archived code under ``old/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

CTransPathMode = Literal["frozen", "bitfit_norm", "last_stage"]

CTRANSPATH_REPOSITORY = "RendeiroLab/LazySlide-models-gpl"
CTRANSPATH_FILENAME = "CTransPath/ctranspath_jit.pt"
CTRANSPATH_LICENSE = "GPL-3.0"
CTRANSPATH_INPUT_PIXELS = 224
CTRANSPATH_FEATURE_DIM = 768


@dataclass(frozen=True)
class CTransPathTrainableReport:
    mode: str
    selected_names: tuple[str, ...]
    total_parameters: int
    trainable_parameters: int


def _final_stage(model: nn.Module) -> nn.Module:
    modules = dict(model.named_modules())
    if "layers.3" not in modules:
        raise RuntimeError("CTransPath final-stage module 'layers.3' is missing")
    return modules["layers.3"]


def _is_norm_parameter(name: str) -> bool:
    components = name.lower().split(".")
    return any(component.startswith("norm") for component in components)


def configure_ctranspath_trainable(
    model: nn.Module, mode: CTransPathMode
) -> CTransPathTrainableReport:
    """Apply a prespecified W1 trainability policy to a CTransPath module."""
    valid = {"frozen", "bitfit_norm", "last_stage"}
    if mode not in valid:
        raise ValueError(f"unknown CTransPath mode {mode!r}; choose {sorted(valid)}")

    selected: list[str] = []
    total = 0
    trainable = 0
    for name, parameter in model.named_parameters():
        total += parameter.numel()
        if mode == "frozen":
            enabled = False
        elif mode == "bitfit_norm":
            enabled = name.endswith(".bias") or _is_norm_parameter(name)
        else:
            enabled = name.startswith("layers.3.") or name.startswith("norm.")
        parameter.requires_grad = enabled
        if enabled:
            selected.append(name)
            trainable += parameter.numel()

    if mode != "frozen" and not selected:
        raise RuntimeError(
            f"CTransPath mode {mode!r} selected no parameters; the upstream "
            "module naming contract may have changed"
        )
    return CTransPathTrainableReport(
        mode=mode,
        selected_names=tuple(selected),
        total_parameters=total,
        trainable_parameters=trainable,
    )


class TrainableCTransPath(nn.Module):
    """Differentiable wrapper around LazySlide's checksum-cached TorchScript model."""

    def __init__(self, mode: CTransPathMode = "frozen"):
        super().__init__()
        try:
            from lazyslide_models.vision.ctranspath import CTransPath
        except ImportError:  # LazySlide 0.10
            from lazyslide.models.vision.ctranspath import CTransPath

        source = CTransPath()
        self.encoder = source.model
        self.transform = source.get_transform()
        self.report = configure_ctranspath_trainable(self.encoder, mode)
        self.mode = mode
        if mode == "frozen":
            self.encoder.eval()
        else:
            # Keep frozen stages deterministic. Only the final stage is placed in
            # training mode for last-stage adaptation; BitFit/LayerNorm does not
            # require stochastic layers to be enabled.
            self.encoder.eval()
            if mode == "last_stage":
                _final_stage(self.encoder).train()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)

    def train(self, mode: bool = True):
        """Keep frozen stages deterministic under parent ``model.train()`` calls."""
        super().train(mode)
        self.encoder.eval()
        if mode and self.mode == "last_stage":
            _final_stage(self.encoder).train()
        return self


def load_trainable_ctranspath(
    mode: CTransPathMode = "frozen",
    *,
    device: str | torch.device = "cpu",
) -> TrainableCTransPath:
    """Load the current registered CTransPath model for W1 adaptation."""
    model = TrainableCTransPath(mode=mode)
    return model.to(device)
