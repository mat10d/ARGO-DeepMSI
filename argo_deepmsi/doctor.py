"""Read-only environment and experiment preflight checks for agents and humans."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from packaging.specifiers import SpecifierSet

from .experiment import experiment_plan, load_experiment

REQUIRED_PACKAGES = {
    "lazyslide": ">=0.12,<0.13",
    "lazyslide-models": ">=0.0.4,<0.1",
    "wsidata": ">=0.10,<0.12",
}


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    message: str
    details: dict[str, Any] | None = None


class _Checks:
    def __init__(self) -> None:
        self.items: list[Check] = []

    def add(
        self,
        name: str,
        status: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.items.append(Check(name, status, message, details))


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _check_packages(checks: _Checks) -> None:
    versions: dict[str, str | None] = {}
    incompatible = []
    for name, requirement in REQUIRED_PACKAGES.items():
        version = _package_version(name)
        versions[name] = version
        if version is None or version not in SpecifierSet(requirement):
            incompatible.append(f"{name}={version or 'missing'} (need {requirement})")
    if incompatible:
        checks.add(
            "dependencies",
            "error",
            "Locked pathology stack is not active: " + "; ".join(incompatible),
            versions,
        )
    else:
        checks.add("dependencies", "ok", "Locked pathology stack is compatible", versions)


def _check_git(checks: _Checks, workspace: Path) -> None:
    try:
        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=workspace,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        porcelain = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=workspace,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        checks.add("git", "warning", "Workspace is not a readable Git checkout")
        return
    status = "warning" if porcelain else "ok"
    message = f"Branch {branch or '(detached)'}; {len(porcelain)} uncommitted paths"
    checks.add("git", status, message, {"branch": branch, "dirty_paths": len(porcelain)})


def _check_device(checks: _Checks, requested: str) -> None:
    if not requested.startswith("cuda"):
        checks.add("accelerator", "ok", f"Experiment requests {requested}")
        return
    try:
        import torch

        available = torch.cuda.is_available()
        devices = []
        for index in range(torch.cuda.device_count() if available else 0):
            props = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "vram_gib": round(props.total_memory / (1024**3), 1),
                }
            )
    except (ImportError, RuntimeError) as error:
        checks.add("accelerator", "error", f"Cannot inspect requested CUDA device: {error}")
        return
    if not available:
        checks.add("accelerator", "error", "Experiment requests CUDA but no GPU is available")
    else:
        checks.add(
            "accelerator", "ok", f"CUDA available on {len(devices)} device(s)", {"devices": devices}
        )


def _read_table(checks: _Checks, name: str, path: Path, required: set[str]) -> pd.DataFrame | None:
    if not path.is_file():
        checks.add(name, "error", f"Missing input table: {path}")
        return None
    try:
        table = pd.read_csv(path)
    except (OSError, ValueError) as error:
        checks.add(name, "error", f"Cannot read {path}: {error}")
        return None
    missing = required - set(table)
    if missing:
        checks.add(name, "error", f"{path} is missing columns {sorted(missing)}")
        return table
    checks.add(name, "ok", f"{path} has {len(table)} rows")
    return table


def _check_slide_files(
    checks: _Checks,
    table: pd.DataFrame,
    workspace: Path,
    *,
    tile_px: int,
    mpp: float,
    require_current: bool,
) -> None:
    slide_paths = [
        _resolve(workspace, str(value)) for value in table["FILENAME"].dropna().drop_duplicates()
    ]
    missing = [str(path) for path in slide_paths if not path.is_file()]
    if missing:
        checks.add(
            "slides",
            "error",
            f"{len(missing)}/{len(slide_paths)} slide files are missing",
            {"examples": missing[:5]},
        )
    else:
        checks.add("slides", "ok", f"All {len(slide_paths)} referenced slide files exist")

    existing_stores = [
        path.with_suffix(".zarr") for path in slide_paths if path.with_suffix(".zarr").is_dir()
    ]
    if not require_current:
        checks.add(
            "feature_stores",
            "warning",
            f"{len(existing_stores)} stores found; tiling_policy=reuse does not enforce provenance",
        )
        return
    expected_versions = {
        "lazyslide": _package_version("lazyslide"),
        "wsidata": _package_version("wsidata"),
    }
    invalid = []
    for store in existing_stores:
        manifest_path = store / "argo_manifest.json"
        try:
            tiling = json.loads(manifest_path.read_text())["tiling"]
        except (OSError, ValueError, KeyError, TypeError):
            invalid.append(str(store))
            continue
        try:
            matching_mpp = float(tiling.get("mpp", -1)) == float(mpp)
        except (TypeError, ValueError):
            matching_mpp = False
        if (
            tiling.get("lazyslide") != expected_versions["lazyslide"]
            or tiling.get("wsidata") != expected_versions["wsidata"]
            or tiling.get("tile_px") != tile_px
            or not matching_mpp
        ):
            invalid.append(str(store))
    if invalid:
        checks.add(
            "feature_stores",
            "error",
            f"{len(invalid)}/{len(existing_stores)} existing stores do not match current tiling",
            {
                "examples": invalid[:5],
                "expected": {**expected_versions, "tile_px": tile_px, "mpp": mpp},
            },
        )
    else:
        checks.add(
            "feature_stores",
            "ok",
            f"{len(existing_stores)} existing stores match current tiling; missing stores will be created",
        )


def _check_models(checks: _Checks, config: dict[str, Any]) -> None:
    requested = list(config.get("extract", {}).get("models", []))
    if not requested:
        checks.add("models", "ok", "No feature extraction models requested")
        return
    try:
        from .feature_extraction import PATCH_MODELS
    except Exception as error:  # noqa: BLE001 - report broken optional model stack
        checks.add("models", "error", f"Cannot load the model registry: {error}")
        return
    missing = sorted(set(requested) - set(PATCH_MODELS))
    if missing:
        checks.add("models", "error", f"Unknown or unsupported patch models: {missing}")
        return
    gated = sorted(name for name in requested if PATCH_MODELS[name].requires_auth)
    if gated and not os.environ.get("HF_TOKEN"):
        checks.add(
            "model_credentials",
            "error",
            f"HF_TOKEN is required for selected gated models: {gated}",
        )
    else:
        message = f"All {len(requested)} requested models are registered"
        if gated:
            message += "; HF_TOKEN is set"
        checks.add("models", "ok", message, {"requested": requested, "gated": gated})


def run_doctor(
    *,
    workspace: Path | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Run non-mutating checks and return a stable machine-readable report."""
    checks = _Checks()
    config: dict[str, Any] | None = None
    if config_path is not None:
        config_path = config_path.resolve()
        try:
            config = load_experiment(config_path)
            plan = experiment_plan(config)
            checks.add("config", "ok", f"Valid experiment with {len(plan)} stages")
        except (OSError, ValueError) as error:
            checks.add("config", "error", str(error))
    if workspace is None and config is not None and config_path is not None:
        workspace = _resolve(config_path.parent, config["run"].get("workspace", "."))
    workspace = (workspace or Path.cwd()).resolve()

    if not workspace.is_dir():
        checks.add("workspace", "error", f"Workspace does not exist: {workspace}")
    else:
        writable = os.access(workspace, os.W_OK)
        checks.add(
            "workspace",
            "ok" if writable else "error",
            f"Workspace {'is' if writable else 'is not'} writable: {workspace}",
        )
        total, _used, free = shutil.disk_usage(workspace)
        free_gib = free / (1024**3)
        checks.add(
            "disk",
            "warning" if free_gib < 100 else "ok",
            f"{free_gib:.1f} GiB free",
            {"free_gib": round(free_gib, 1), "total_gib": round(total / (1024**3), 1)},
        )
        _check_git(checks, workspace)

    supported_python = (3, 11) <= sys.version_info[:2] < (3, 13)
    checks.add(
        "python",
        "ok" if supported_python else "error",
        f"Python {platform.python_version()} ({'supported' if supported_python else 'need 3.11 or 3.12'})",
    )
    lock_path = workspace / "uv.lock"
    checks.add(
        "lockfile",
        "ok" if lock_path.is_file() else "error",
        f"{'Found' if lock_path.is_file() else 'Missing'} {lock_path}",
    )
    _check_packages(checks)

    requested_device = "cpu"
    if config is not None:
        run = config["run"]
        budget = run.get("budget", {})
        offline = not budget.get("allow_network", True) or not budget.get(
            "allow_model_downloads", True
        )
        checks.add(
            "execution_policy",
            "ok",
            (
                "Offline model-hub mode will be enforced"
                if offline
                else "Network/model downloads are allowed by the run budget"
            ),
            {"budget": budget},
        )
        requested_device = str(config.get("extract", {}).get("device", run.get("device", "cpu")))
        input_specs = {
            "slide_table": ({"FILENAME"}, run.get("slide_table")),
            "clinical_table": (set(), run.get("clinical_table")),
            "cohort": ({"slide_id", "patient_id", "site", "y"}, run.get("cohort")),
        }
        loaded: dict[str, pd.DataFrame] = {}
        for name, (required, value) in input_specs.items():
            if value is not None:
                table = _read_table(checks, name, _resolve(workspace, value), required)
                if table is not None:
                    loaded[name] = table
        if "slide_table" in loaded:
            extract = config.get("extract", {})
            _check_slide_files(
                checks,
                loaded["slide_table"],
                workspace,
                tile_px=int(extract.get("tile_px", 256)),
                mpp=float(extract.get("mpp", 0.5)),
                require_current=extract.get("tiling_policy", "require-current")
                == "require-current",
            )
        _check_models(checks, config)
    _check_device(checks, requested_device)

    counts = {
        status: sum(item.status == status for item in checks.items)
        for status in ("ok", "warning", "error")
    }
    status = "error" if counts["error"] else "warning" if counts["warning"] else "ok"
    return {
        "status": status,
        "workspace": str(workspace),
        "config": str(config_path) if config_path is not None else None,
        "counts": counts,
        "checks": [asdict(item) for item in checks.items],
    }
