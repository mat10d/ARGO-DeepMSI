"""Per-output provenance for backend extractions."""

from __future__ import annotations

import json
import platform
import socket
from pathlib import Path
from typing import Any

from .._environment import project_root
from ..reproducibility import file_sha256, git_state, jsonable, utc_now, write_json
from .config import BackendConfig

PROVENANCE_SCHEMA = 1


def collect_provenance(
    cfg: BackendConfig,
    *,
    tool_version: str | None = None,
    tool_commit: str | None = None,
    env_lock: Path | None = None,
    extra: dict | None = None,
) -> dict[str, Any]:
    """Describe how a feature file was produced.

    Args:
        cfg: Backend config used for the extraction.
        tool_version: Version of the extraction tool (e.g. ``mussel-pathology``).
        tool_commit: Pinned source commit of the extraction tool, if installed from git.
        env_lock: Lock file of the environment that ran the tool; hashed, not copied.
        extra: Additional JSON-compatible fields (slide path, command, timings).

    Returns:
        A JSON-compatible provenance dict.
    """
    env_lock = Path(env_lock) if env_lock else None
    lock_hash = file_sha256(env_lock) if env_lock and env_lock.is_file() else None
    prov: dict[str, Any] = {
        "schema": PROVENANCE_SCHEMA,
        "backend": cfg.backend,
        "model": cfg.model,
        "params": cfg.params,
        "config_path": str(cfg.source) if cfg.source else None,
        "config_sha256": (
            file_sha256(cfg.source) if cfg.source and Path(cfg.source).is_file() else None
        ),
        "tool": {"version": tool_version, "commit": tool_commit},
        "env_lock": {"path": str(env_lock) if env_lock else None, "sha256": lock_hash},
        "argo_git": git_state(project_root()),
        "created_at": utc_now(),
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
    }
    if extra:
        prov["extra"] = extra
    return jsonable(prov)


def write_provenance(path: str | Path, prov: dict[str, Any]) -> None:
    """Atomically write a provenance dict as JSON."""
    write_json(Path(path), prov)


def read_provenance(path: str | Path) -> dict[str, Any]:
    """Read a provenance JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
