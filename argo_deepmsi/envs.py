"""Registry and runner for ARGO's isolated tool environments.

The core environment holds the CLI, cohort tasks, readers, and analyses. Slide encoders
run in their own locked environments under ``<repo>/envs/<name>/`` and are reached
through ``uv run --frozen --project envs/<name>``. Commands that need LazySlide call
:func:`ensure_env` first; when LazySlide is not importable the same ``argo`` argv is
re-executed inside ``envs/lazyslide`` (which installs this package as a path
dependency), guarded against loops by ``ARGO_ENV``.

This module imports only the standard library.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


def _repo_root() -> Path:
    configured = os.environ.get("ARGO_REPO_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    package_parent = Path(__file__).resolve().parents[1]
    if (package_parent / "pyproject.toml").is_file():
        return package_parent
    from ._environment import project_root

    return project_root()


REPO_ROOT = _repo_root()
ENVS_DIR = REPO_ROOT / "envs"


@dataclass(frozen=True)
class EnvSpec:
    """One isolated environment ARGO can dispatch to."""

    name: str
    project_dir: Path
    kind: Literal["uv", "venv"]
    probe_module: str | None
    description: str

    @property
    def venv_dir(self) -> Path:
        return self.project_dir / ".venv"

    @property
    def lock_file(self) -> Path | None:
        return self.project_dir / "uv.lock" if self.kind == "uv" else None


ENVS: dict[str, EnvSpec] = {
    "core": EnvSpec(
        "core",
        REPO_ROOT,
        "uv",
        "argo_deepmsi",
        "CLI, setup, cohort tasks, readers, analyses",
    ),
    "lazyslide": EnvSpec(
        "lazyslide",
        ENVS_DIR / "lazyslide",
        "uv",
        "lazyslide",
        "argo-deepmsi + LazySlide stack (extract, aggregate, qc, visualize)",
    ),
    "mussel": EnvSpec(
        "mussel",
        ENVS_DIR / "mussel",
        "uv",
        "mussel",
        "Mussel only; called as `tessellate_extract_features` subprocess",
    ),
    "paladin": EnvSpec(
        "paladin",
        ENVS_DIR / "paladin",
        "venv",
        "lightning",
        "PALADIN/Aeon inference venv built by envs/paladin/setup.sh (not locked)",
    ),
}

DEFAULT_SETUP_ENVS = ("core", "lazyslide", "mussel")

# Replaced in tests; production re-exec replaces the current process.
_exec: Callable[[str, list[str], dict[str, str]], Any] = os.execvpe


class EnvDispatchError(RuntimeError):
    """Raised when a command cannot run here and must not be re-executed."""


class EnvSetupError(RuntimeError):
    """Raised when an environment sync command fails."""


def get_env(name: str) -> EnvSpec:
    """Return the registered environment ``name``.

    Raises:
        KeyError: If ``name`` is not registered.
    """
    try:
        return ENVS[name]
    except KeyError:
        raise KeyError(f"Unknown environment {name!r}; choose one of {sorted(ENVS)}") from None


def env_command(name: str, argv: Sequence[str]) -> list[str]:
    """Build the command that runs ``argv`` inside environment ``name``.

    uv environments (including core) become
    ``uv run --frozen --project <project_dir> *argv``; an empty ``argv`` yields the bare
    prefix, suitable as a ``command_prefix``. The ``venv`` kind (PALADIN) has no uv
    project, so ``argv[0]`` is resolved against ``<project_dir>/.venv/bin/`` (``python``,
    ``pip``, or an installed console script); an empty ``argv`` yields that venv's
    ``python``.

    Args:
        name: Registered environment name.
        argv: Command and arguments to run inside the environment.

    Returns:
        The full command as a list suitable for ``subprocess.run`` or ``os.execvpe``.
    """
    spec = get_env(name)
    argv = list(argv)
    if spec.kind == "uv":
        return ["uv", "run", "--frozen", "--project", str(spec.project_dir), *argv]
    bin_dir = spec.venv_dir / "bin"
    if not argv:
        return [str(bin_dir / "python")]
    return [str(bin_dir / argv[0]), *argv[1:]]


def module_available(module: str) -> bool:
    """Return whether ``module`` is importable here, without importing it."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def ensure_env(name: str) -> None:
    """Make sure the current command runs in environment ``name``.

    Call this at the top of a command body. When the environment's probe module is
    importable, return. Otherwise re-execute ``argo <current argv>`` inside the target
    environment with ``ARGO_ENV=<name>``. If ``ARGO_ENV`` already names the target (the
    environment exists but is broken) or ``ARGO_NO_DISPATCH=1``, raise instead.

    Args:
        name: Registered environment name.

    Raises:
        EnvDispatchError: If the command cannot run here and re-execution is disabled,
            would loop, or fails to start.
    """
    spec = get_env(name)
    if spec.probe_module is None or module_available(spec.probe_module):
        return
    cmd = env_command(name, ["argo", *sys.argv[1:]])
    if os.environ.get("ARGO_ENV") == name:
        raise EnvDispatchError(
            f"Already dispatched to the {name!r} environment but {spec.probe_module!r} is "
            f"still not importable. Repair it with `argo setup --env {name}`."
        )
    if os.environ.get("ARGO_NO_DISPATCH") == "1":
        raise EnvDispatchError(
            f"{spec.probe_module!r} is not importable and ARGO_NO_DISPATCH=1 disables "
            f"re-execution. Run instead: {' '.join(cmd)}"
        )
    if spec.kind == "uv" and not (spec.project_dir / "pyproject.toml").is_file():
        raise EnvDispatchError(
            f"{spec.probe_module!r} is not importable and {spec.project_dir} is not a uv "
            f"project. Run `argo setup --env {name}`."
        )
    env = dict(os.environ)
    env["ARGO_ENV"] = name
    env.pop("VIRTUAL_ENV", None)
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        _exec(cmd[0], cmd, env)
    except OSError as error:
        raise EnvDispatchError(f"Could not re-execute in {name!r}: {error}") from error


def env_status(
    name: str,
    *,
    probe: bool = True,
    lock_check: bool = False,
    timeout: float = 60.0,
) -> dict[str, Any]:
    """Summarise an environment without syncing or modifying it.

    The probe runs ``<project_dir>/.venv/bin/python -c "import <probe_module>"`` directly
    rather than ``uv run``, which would sync (and possibly build) the environment.

    Args:
        name: Registered environment name.
        probe: Import the probe module in the environment's interpreter.
        lock_check: Run ``uv lock --check`` for uv environments (may resolve over network).
        timeout: Seconds allowed for each subprocess.

    Returns:
        Dictionary with ``name``, ``kind``, ``project_dir``, ``exists``, ``lock``,
        ``venv``, ``lock_check``, ``probe`` (``True``/``False``/``None`` when skipped),
        and a short ``detail`` string.
    """
    spec = get_env(name)
    lock = spec.lock_file.is_file() if spec.lock_file is not None else None
    python = spec.venv_dir / "bin" / "python"
    status: dict[str, Any] = {
        "name": name,
        "kind": spec.kind,
        "project_dir": str(spec.project_dir),
        "description": spec.description,
        "exists": spec.project_dir.is_dir(),
        "lock": lock,
        "venv": python.exists(),
        "lock_check": None,
        "probe": None,
        "detail": "",
    }
    if lock_check and spec.kind == "uv" and lock:
        result = _run_quiet(["uv", "lock", "--check", "--project", str(spec.project_dir)], timeout)
        status["lock_check"] = result is not None and result.returncode == 0
    if probe and spec.probe_module and status["venv"]:
        result = _run_quiet([str(python), "-c", f"import {spec.probe_module}"], timeout)
        status["probe"] = result is not None and result.returncode == 0
        if result is None:
            status["detail"] = "probe timed out"
        elif result.returncode != 0:
            lines = [line for line in result.stderr.splitlines() if line.strip()]
            status["detail"] = (lines[-1] if lines else "import failed")[:160]
    elif not status["exists"]:
        status["detail"] = "project directory missing"
    elif not status["venv"]:
        status["detail"] = f"not synced; run `argo setup --env {name}`"
    return status


def _run_quiet(cmd: list[str], timeout: float) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None


def setup_commands(name: str, *, extras: Iterable[str] = ()) -> list[list[str]]:
    """Return the commands that create or sync environment ``name``.

    Args:
        name: Registered environment name.
        extras: Optional-dependency extras to include (uv environments only).

    Returns:
        Commands in execution order.
    """
    spec = get_env(name)
    if spec.kind == "venv":
        return [["bash", str(spec.project_dir / "setup.sh")]]
    cmd = ["uv", "sync", "--frozen", "--project", str(spec.project_dir)]
    if name == "core":
        extras = ("dev", *(extra for extra in extras if extra != "dev"))
    for extra in extras:
        cmd += ["--extra", extra]
    return [cmd]


def setup_env(
    name: str,
    *,
    extras: Iterable[str] = (),
    dry_run: bool = False,
    runner: Callable[..., Any] = subprocess.run,
) -> list[list[str]]:
    """Create or sync environment ``name`` from its lock (or setup script).

    Args:
        name: Registered environment name.
        extras: Optional-dependency extras to include (uv environments only).
        dry_run: Return the commands without running them.
        runner: ``subprocess.run``-compatible callable.

    Returns:
        The commands that were (or would be) run.

    Raises:
        EnvSetupError: If the project is missing or a command exits nonzero.
    """
    spec = get_env(name)
    commands = setup_commands(name, extras=extras)
    if dry_run:
        return commands
    marker = spec.project_dir / ("pyproject.toml" if spec.kind == "uv" else "setup.sh")
    if not marker.is_file():
        raise EnvSetupError(f"{name}: {marker} does not exist")
    for cmd in commands:
        result = runner(cmd, check=False, cwd=str(REPO_ROOT))
        code = getattr(result, "returncode", 0)
        if code != 0:
            raise EnvSetupError(f"{name}: `{' '.join(cmd)}` exited with {code}")
    return commands


def hf_token_source() -> str | None:
    """Name where a Hugging Face token would come from, never the token itself."""
    if os.environ.get("HF_TOKEN"):
        return "$HF_TOKEN"
    if module_available("huggingface_hub"):
        try:
            from huggingface_hub import get_token

            if get_token():
                return "huggingface_hub"
        except Exception:  # noqa: BLE001
            pass
    candidates = [os.environ.get("HF_TOKEN_PATH")]
    if os.environ.get("HF_HOME"):
        candidates.append(str(Path(os.environ["HF_HOME"]) / "token"))
    candidates.append(str(Path.home() / ".cache" / "huggingface" / "token"))
    for candidate in candidates:
        if candidate and Path(candidate).expanduser().is_file():
            if Path(candidate).expanduser().read_text().strip():
                return candidate
    return None


def dotenv_keys(path: Path) -> set[str]:
    """Return variable names assigned in a ``.env`` file, ignoring their values."""
    if not path.is_file():
        return set()
    keys = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key = line.split("=", 1)[0].strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        if line.split("=", 1)[1].strip().strip("'\""):
            keys.add(key)
    return keys
