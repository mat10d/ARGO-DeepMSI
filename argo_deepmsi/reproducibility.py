"""Run provenance helpers shared by CLI experiments and scorer jobs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PROVENANCE_PACKAGES = (
    "argo-deepmsi",
    "lazyslide",
    "lazyslide-models",
    "wsidata",
    "torch",
    "transformers",
    "numpy",
    "pandas",
    "scikit-learn",
    "anndata",
    "zarr",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_versions(packages: Iterable[str] = PROVENANCE_PACKAGES) -> dict[str, str]:
    """Read installed versions without importing heavyweight ML packages."""
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def git_state(workspace: Path) -> dict[str, Any]:
    """Return the checkout revision and dirty flag, or an explicit fallback."""

    def _git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=workspace,
            text=True,
            capture_output=True,
            check=False,
        )

    revision = _git("rev-parse", "HEAD")
    if revision.returncode != 0:
        return {"revision": None, "dirty": None}
    status = _git("status", "--porcelain")
    return {
        "revision": revision.stdout.strip(),
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def environment_snapshot(workspace: Path) -> dict[str, Any]:
    """Capture enough environment state to diagnose cross-machine drift."""
    return {
        "captured_at": utc_now(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions(),
        "git": git_state(workspace),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def jsonable(value: Any) -> Any:
    """Convert common scientific-Python values into JSON-compatible data."""
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, set):
        return [jsonable(item) for item in sorted(value, key=repr)]
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    """Atomically write formatted JSON so interrupted runs do not corrupt manifests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(serialized)
        temporary = Path(handle.name)
    temporary.replace(path)
