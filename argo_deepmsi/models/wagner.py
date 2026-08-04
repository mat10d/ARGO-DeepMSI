"""Canonical Wagner MSI slide transformer used by ARGO-DeepMSI.

This module owns the runtime implementation.  The archived HistoBistro checkout
under ``old/`` is provenance only and must never be imported by production or W1
code.  The external checkpoint is staged separately under ``artifacts/`` and is
verified before loading.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

DEFAULT_WAGNER_CHECKPOINT = Path(
    "artifacts/checkpoints/wagner/MSI_high_CRC_model.pth"
)
WAGNER_CHECKPOINT_SHA256 = (
    "201678419198039bcc48614ef1c262928b61e2d79dab494be99c97cc60169b15"
)


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of ``path`` without reading it all at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_wagner_checkpoint(path: str | Path | None = None) -> Path:
    """Resolve and verify the project-local external Wagner checkpoint."""
    candidate = Path(
        path
        or os.environ.get("ARGO_WAGNER_CHECKPOINT", "")
        or DEFAULT_WAGNER_CHECKPOINT
    )
    if not candidate.exists():
        raise FileNotFoundError(
            f"Wagner checkpoint not found at {candidate}. Stage the published "
            "checkpoint at artifacts/checkpoints/wagner/MSI_high_CRC_model.pth "
            "or set ARGO_WAGNER_CHECKPOINT. Runtime code does not read old/."
        )
    actual = sha256_file(candidate)
    if actual != WAGNER_CHECKPOINT_SHA256:
        raise ValueError(
            f"Wagner checkpoint checksum mismatch for {candidate}: "
            f"expected {WAGNER_CHECKPOINT_SHA256}, got {actual}"
        )
    return candidate


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim: int = 512, hidden_dim: int = 512, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int = 512,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if heads != 1 or dim_head != dim
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (
            rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in qkv
        )
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.to_out[1].p if self.training and isinstance(self.to_out, nn.Sequential) else 0.0,
        )
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class TransformerBlocks(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim,
                                heads=heads,
                                dim_head=dim_head,
                                dropout=dropout,
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attention, feed_forward in self.layers:
            x = attention(x) + x
            x = feed_forward(x) + x
        return x


class WagnerTransformer(nn.Module):
    """Published HistoBistro MSI aggregator over 768-D CTransPath tiles."""

    def __init__(
        self,
        num_classes: int = 1,
        input_dim: int = 768,
        dim: int = 512,
        depth: int = 2,
        heads: int = 8,
        mlp_dim: int = 512,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
    ):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, heads * dim_head, bias=True), nn.ReLU()
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(mlp_dim), nn.Linear(mlp_dim, num_classes)
        )
        self.transformer = TransformerBlocks(
            dim, depth, heads, dim_head, mlp_dim, dropout
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(emb_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=batch_size)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.norm(x[:, 0])
        return self.mlp_head(x)


def load_wagner(
    checkpoint: str | Path | None = None,
    *,
    device: str | torch.device = "cpu",
    eval_mode: bool = True,
) -> WagnerTransformer:
    """Load the checksum-verified external Wagner weights."""
    path = resolve_wagner_checkpoint(checkpoint)
    model = WagnerTransformer()
    state = torch.load(path, map_location="cpu", weights_only=False)
    cleaned = {key.removeprefix("model."): value for key, value in state.items()}
    cleaned.pop("criterion.pos_weight", None)
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Wagner checkpoint contract mismatch; missing={missing}, "
            f"unexpected={unexpected}"
        )
    model.to(device)
    if eval_mode:
        model.eval()
    return model


def configure_wagner_trainable(model: WagnerTransformer, mode: str) -> list[str]:
    """Freeze Wagner parameters according to a prespecified W1 adaptation level."""
    valid = {"frozen", "head", "last_block", "full"}
    if mode not in valid:
        raise ValueError(f"unknown Wagner trainable mode {mode!r}; choose {sorted(valid)}")
    for parameter in model.parameters():
        parameter.requires_grad = False
    prefixes: tuple[str, ...]
    if mode == "frozen":
        prefixes = ()
    elif mode == "head":
        prefixes = ("norm.", "mlp_head.")
    elif mode == "last_block":
        prefixes = ("transformer.layers.1.", "norm.", "mlp_head.")
    else:
        prefixes = ("",)
    selected = []
    for name, parameter in model.named_parameters():
        if name.startswith(prefixes):
            parameter.requires_grad = True
            selected.append(name)
    return selected


def parameter_counts(model: nn.Module) -> dict[str, int]:
    """Return total and trainable scalar parameter counts."""
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }
