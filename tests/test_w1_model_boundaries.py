from __future__ import annotations

from pathlib import Path

import pytest
import torch

from argo_deepmsi.models.ctranspath import configure_ctranspath_trainable
from argo_deepmsi.models.wagner import (
    DEFAULT_WAGNER_CHECKPOINT,
    WagnerTransformer,
    configure_wagner_trainable,
    load_wagner,
    parameter_counts,
)


class _FakeCTransPath(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = torch.nn.Linear(4, 4)
        self.layers = torch.nn.ModuleList(
            [torch.nn.Sequential(torch.nn.LayerNorm(4), torch.nn.Linear(4, 4)) for _ in range(4)]
        )
        self.norm = torch.nn.LayerNorm(4)


def test_wagner_trainable_ladder_is_monotonic():
    counts = []
    for mode in ("frozen", "head", "last_block", "full"):
        model = WagnerTransformer()
        configure_wagner_trainable(model, mode)
        counts.append(parameter_counts(model)["trainable"])
    assert counts[0] == 0
    assert counts == sorted(counts)
    assert len(set(counts)) == len(counts)


def test_ctranspath_trainable_policies_are_bounded():
    frozen = _FakeCTransPath()
    report0 = configure_ctranspath_trainable(frozen, "frozen")
    assert report0.trainable_parameters == 0

    bitfit = _FakeCTransPath()
    report1 = configure_ctranspath_trainable(bitfit, "bitfit_norm")
    assert 0 < report1.trainable_parameters < report1.total_parameters
    assert all(
        name.endswith(".bias") or "norm" in name.lower()
        for name in report1.selected_names
    )

    last = _FakeCTransPath()
    report2 = configure_ctranspath_trainable(last, "last_stage")
    assert 0 < report2.trainable_parameters < report2.total_parameters
    assert all(
        name.startswith("layers.3.") or name.startswith("norm.")
        for name in report2.selected_names
    )


@pytest.mark.integration
@pytest.mark.skipif(
    not DEFAULT_WAGNER_CHECKPOINT.exists(), reason="external Wagner checkpoint not staged"
)
def test_staged_wagner_checkpoint_loads_strictly():
    model = load_wagner(device="cpu")
    assert parameter_counts(model) == {"total": 3_549_697, "trainable": 3_549_697}


def test_w1_runtime_sources_do_not_reference_archived_tree():
    root = Path(__file__).resolve().parents[1]
    runtime_files = [
        root / "argo_deepmsi/models/ctranspath.py",
        root / "argo_deepmsi/models/wagner.py",
        root / "scripts/wagner_zeroshot.py",
    ]
    runtime_files.extend(sorted((root / "argo_deepmsi").glob("w1*.py")))
    runtime_files.extend(sorted((root / "scripts").glob("w1*.py")))
    for path in runtime_files:
        source = path.read_text()
        assert "from old" not in source
        assert "import old" not in source
        assert 'default="old/' not in source
        assert "default='old/" not in source
