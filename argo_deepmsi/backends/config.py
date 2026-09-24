"""Backend extraction configs: one checked-in TOML per (backend, model)."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .._environment import project_root
from . import BACKENDS

_TOP_LEVEL_KEYS = {"backend", "params"}
_BACKEND_KEYS = {"name", "model"}
_MODEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

LAZYSLIDE_PARAMS = {
    "model_key",
    "tile_px",
    "mpp",
    "amp",
    "batch_size",
    "num_workers",
    "slide_encoder",
}
MUSSEL_RESERVED_PARAMS = {"slide_path", "output_h5_path", "output_pt_path"}


@dataclass(frozen=True)
class BackendConfig:
    """One validated backend configuration.

    Attributes:
        backend: ``"lazyslide"`` or ``"mussel"``.
        model: ARGO model name (e.g. ``"hoptimus0"``, ``"titan"``).
        params: Backend parameters exactly as written in the TOML ``[params]`` table.
        source: TOML file the config was loaded from, if any.
    """

    backend: str
    model: str
    params: dict[str, Any] = field(default_factory=dict)
    source: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "backend": self.backend,
            "model": self.model,
            "params": self.params,
            "source": str(self.source) if self.source else None,
        }


def validate_backend_config(cfg: BackendConfig) -> BackendConfig:
    """Raise ``ValueError`` if a config is malformed; return it unchanged otherwise."""
    if cfg.backend not in BACKENDS:
        raise ValueError(f"Unknown backend {cfg.backend!r}; expected one of {BACKENDS}")
    if not isinstance(cfg.model, str) or not _MODEL_PATTERN.match(cfg.model):
        raise ValueError(f"Invalid model name {cfg.model!r}")
    if not isinstance(cfg.params, dict):
        raise ValueError("[params] must be a table")
    if cfg.backend == "lazyslide":
        unknown = set(cfg.params) - LAZYSLIDE_PARAMS
        if unknown:
            raise ValueError(
                f"Unknown lazyslide params {sorted(unknown)}; allowed: {sorted(LAZYSLIDE_PARAMS)}"
            )
    else:
        reserved = set(cfg.params) & MUSSEL_RESERVED_PARAMS
        if reserved:
            raise ValueError(f"Mussel params {sorted(reserved)} are set per slide by ARGO")
    return cfg


def load_backend_config(path: str | Path) -> BackendConfig:
    """Load and validate a backend TOML config.

    Args:
        path: TOML file with a ``[backend]`` table (``name``, ``model``) and an
            optional ``[params]`` table.

    Returns:
        The validated :class:`BackendConfig`.

    Raises:
        ValueError: On unknown keys, an unknown backend, or invalid params.
    """
    path = Path(path)
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    unknown = set(raw) - _TOP_LEVEL_KEYS
    if unknown:
        raise ValueError(f"{path}: unknown top-level keys {sorted(unknown)}")
    backend = raw.get("backend")
    if not isinstance(backend, dict):
        raise ValueError(f"{path}: missing [backend] table")
    unknown = set(backend) - _BACKEND_KEYS
    if unknown:
        raise ValueError(f"{path}: unknown [backend] keys {sorted(unknown)}")
    missing = _BACKEND_KEYS - set(backend)
    if missing:
        raise ValueError(f"{path}: [backend] is missing {sorted(missing)}")
    cfg = BackendConfig(
        backend=backend["name"],
        model=backend["model"],
        params=dict(raw.get("params", {})),
        source=path.resolve(),
    )
    try:
        return validate_backend_config(cfg)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc


def default_config_path(backend: str, model: str) -> Path:
    """Return ``configs/backends/<backend>-<model>.toml`` under the project root."""
    return project_root() / "configs" / "backends" / f"{backend}-{model}.toml"
