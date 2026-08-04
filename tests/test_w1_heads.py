from __future__ import annotations

import pytest
import torch

from argo_deepmsi.models.w1_heads import (
    AdaptedWagner,
    GatedAttentionMIL,
    QueryPooler,
    ResidualFeatureAdapter,
)
from argo_deepmsi.models.wagner import WagnerTransformer, configure_wagner_trainable


@pytest.mark.parametrize("model", [GatedAttentionMIL(), QueryPooler()])
def test_new_poolers_accept_variable_tile_bags(model):
    for n_tiles in (7, 23):
        output = model(torch.randn(1, n_tiles, 768))
        assert output.shape == (1, 1)
        assert torch.isfinite(output).all()


def test_zero_initialized_adapter_preserves_features():
    adapter = ResidualFeatureAdapter()
    tiles = torch.randn(2, 11, 768)
    torch.testing.assert_close(adapter(tiles), tiles)


def test_adapted_wagner_preserves_frozen_initial_output():
    torch.manual_seed(3)
    baseline = WagnerTransformer()
    adapted_base = WagnerTransformer()
    adapted_base.load_state_dict(baseline.state_dict())
    configure_wagner_trainable(adapted_base, "last_block")
    adapted = AdaptedWagner(adapted_base)
    baseline.eval()
    adapted.eval()
    tiles = torch.randn(1, 13, 768)
    with torch.no_grad():
        torch.testing.assert_close(adapted(tiles), baseline(tiles))
