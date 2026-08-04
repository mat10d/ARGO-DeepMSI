from __future__ import annotations

import torch


def test_mixed_feature_replacement_propagates_gradient():
    cached = torch.zeros(1, 7, 3)
    encoded = torch.nn.Parameter(torch.ones(2, 3))
    positions = torch.tensor([1, 5])
    mixed = cached.clone()
    mixed[:, positions, :] = encoded.unsqueeze(0)
    mixed.sum().backward()
    torch.testing.assert_close(encoded.grad, torch.ones_like(encoded))
