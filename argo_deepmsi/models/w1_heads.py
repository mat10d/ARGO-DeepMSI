"""Prespecified cached-feature models for the W1 CTransPath search."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from .wagner import (
    WagnerTransformer,
    configure_wagner_trainable,
    load_wagner,
    parameter_counts,
)


@dataclass(frozen=True)
class CachedCandidateSpec:
    candidate_id: str
    description: str
    learning_rate: float
    weight_decay: float = 1e-4
    max_epochs: int = 20
    patience: int = 4
    train_tile_cap: int = 256
    eval_tile_cap: int = 1024
    sampling: str = "uniform"

    def to_dict(self) -> dict:
        return asdict(self)


CACHED_CANDIDATES = {
    "W1-1": CachedCandidateSpec(
        "W1-1",
        "Warm-started Wagner final normalization and MSI head",
        learning_rate=3e-4,
    ),
    "W1-2": CachedCandidateSpec(
        "W1-2",
        "Warm-started Wagner second transformer block and MSI head",
        learning_rate=1e-4,
    ),
    "W1-3": CachedCandidateSpec(
        "W1-3",
        "Warm-started full Wagner projection, transformer, CLS token, and head",
        learning_rate=5e-5,
    ),
    "W1-4": CachedCandidateSpec(
        "W1-4",
        "Compact gated-attention MIL over frozen CTransPath tiles",
        learning_rate=2e-4,
    ),
    "W1-5": CachedCandidateSpec(
        "W1-5",
        "Fixed-query cross-attention pooler over frozen CTransPath tiles",
        learning_rate=2e-4,
    ),
    "W1-6": CachedCandidateSpec(
        "W1-6",
        "Residual feature adapter plus warm-started Wagner final block",
        learning_rate=1e-4,
    ),
}


class GatedAttentionMIL(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
        projection_dim: int = 256,
        attention_dim: int = 128,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attention_v = nn.Linear(projection_dim, attention_dim)
        self.attention_u = nn.Linear(projection_dim, attention_dim)
        self.attention_w = nn.Linear(attention_dim, 1)
        self.head = nn.Sequential(
            nn.LayerNorm(projection_dim), nn.Linear(projection_dim, 1)
        )

    def forward(self, tiles: torch.Tensor) -> torch.Tensor:
        features = self.projection(tiles)
        attention = self.attention_w(
            torch.tanh(self.attention_v(features))
            * torch.sigmoid(self.attention_u(features))
        )
        attention = torch.softmax(attention, dim=1)
        pooled = (attention * features).sum(dim=1)
        return self.head(pooled)


class QueryPooler(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        n_queries: int = 8,
        heads: int = 8,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU()
        )
        self.queries = nn.Parameter(torch.randn(1, n_queries, hidden_dim) * 0.02)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, batch_first=True
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, tiles: torch.Tensor) -> torch.Tensor:
        features = self.projection(tiles)
        queries = self.queries.expand(features.shape[0], -1, -1)
        pooled, _ = self.cross_attention(
            queries, features, features, need_weights=False
        )
        return self.head(self.output_norm(pooled).mean(dim=1))


class ResidualFeatureAdapter(nn.Module):
    def __init__(self, dim: int = 768, bottleneck: int = 64, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, bottleneck)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, tiles: torch.Tensor) -> torch.Tensor:
        delta = self.up(self.dropout(self.activation(self.down(self.norm(tiles)))))
        return tiles + delta


class AdaptedWagner(nn.Module):
    def __init__(self, wagner: WagnerTransformer):
        super().__init__()
        self.adapter = ResidualFeatureAdapter()
        self.wagner = wagner

    def forward(self, tiles: torch.Tensor) -> torch.Tensor:
        return self.wagner(self.adapter(tiles))


def build_cached_candidate(
    candidate_id: str,
    *,
    device: str | torch.device = "cpu",
) -> tuple[nn.Module, CachedCandidateSpec, dict[str, int]]:
    """Instantiate one prespecified W1 cached-feature candidate."""
    if candidate_id not in CACHED_CANDIDATES:
        raise ValueError(
            f"unknown cached W1 candidate {candidate_id!r}; "
            f"choose {sorted(CACHED_CANDIDATES)}"
        )
    spec = CACHED_CANDIDATES[candidate_id]
    if candidate_id in {"W1-1", "W1-2", "W1-3"}:
        model = load_wagner(device=device, eval_mode=False)
        mode = {"W1-1": "head", "W1-2": "last_block", "W1-3": "full"}[
            candidate_id
        ]
        configure_wagner_trainable(model, mode)
    elif candidate_id == "W1-4":
        model = GatedAttentionMIL().to(device)
    elif candidate_id == "W1-5":
        model = QueryPooler().to(device)
    else:
        wagner = load_wagner(device=device, eval_mode=False)
        configure_wagner_trainable(wagner, "last_block")
        model = AdaptedWagner(wagner).to(device)
    model.train()
    return model, spec, parameter_counts(model)
