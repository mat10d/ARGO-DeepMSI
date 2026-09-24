"""Run Mussel's ``tessellate_extract_features`` per slide, exactly as documented upstream.

ARGO only builds the documented command line from a checked-in config, isolates each
slide, lays outputs next to the slide, and records provenance. Parameters absent from
the config are not passed, so Mussel's own documented defaults apply.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import pandas as pd

from .._environment import project_root
from .config import BackendConfig
from .provenance import collect_provenance, read_provenance, write_provenance

MUSSEL_COMMAND = "tessellate_extract_features"
MUSSEL_PACKAGE = "mussel-pathology"
MUSSEL_PINNED_COMMIT = "d4cfce9437d92706811b465ff46cfb247c462249"

# ARGO model name -> Mussel ``ModelType`` names (``mussel/models/model_factory.py`` at the pin).
# The README patch-encoder table lists H-Optimus-0 as ``OPTIMUS``; configs may override it.
MUSSEL_MODEL_TYPES: dict[str, dict[str, str]] = {
    "hoptimus0": {"model_type": "OPTIMUS"},
    "titan": {"model_type": "CONCH1_5", "slide_model_type": "TITAN_SLIDE"},
}
_LEADING_KEYS = ("model_type", "slide_model_type")
_ERROR_TAIL_CHARS = 2000


@dataclass
class MusselOutputs:
    """Output locations for one (slide, model) Mussel extraction."""

    dir: Path
    h5: Path
    pt: Path
    provenance: Path

    @property
    def log(self) -> Path:
        return self.dir / self.h5.name.replace(".features.h5", ".log")

    @property
    def failed_log(self) -> Path:
        return self.dir / self.h5.name.replace(".features.h5", ".failed.log")

    @property
    def tiles_h5(self) -> Path:
        """Patch-encoder tile features kept alongside a slide-encoder output (e.g. TITAN)."""
        return self.dir / self.h5.name.replace(".features.h5", ".tiles.features.h5")

    @property
    def tiles_pt(self) -> Path:
        return self.dir / self.h5.name.replace(".features.h5", ".tiles.features.pt")


def _outputs_in(directory: Path, model: str) -> MusselOutputs:
    return MusselOutputs(
        dir=directory,
        h5=directory / f"{model}.features.h5",
        pt=directory / f"{model}.features.pt",
        provenance=directory / f"{model}.provenance.json",
    )


def output_paths(
    slide_path: str | Path, model: str, out_root: str | Path | None = None
) -> MusselOutputs:
    """Return ``<slide_dir>/<slide_stem>.mussel/<model>.features.{h5,pt}`` + provenance.

    Args:
        slide_path: Whole-slide image path.
        model: ARGO model name (used verbatim in file names).
        out_root: If given, mirror the slide's absolute directory under this root
            instead of writing next to the slide.
    """
    slide_path = Path(slide_path).absolute()
    parent = slide_path.parent
    if out_root is not None:
        parent = Path(out_root).absolute() / parent.relative_to(parent.anchor)
    return _outputs_in(parent / f"{slide_path.stem}.mussel", model)


def _flatten(params: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in params.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{name}."))
        else:
            flat[name] = value
    return flat


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_format_value(item) for item in value) + "]"
    if isinstance(value, (Path, str)):
        text = str(value)
        if text == "" or any(ch in text for ch in " ,=[]{}()'\"\\:"):
            return "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"
        return text
    return repr(value) if isinstance(value, float) else str(value)


def mussel_overrides(cfg: BackendConfig) -> list[tuple[str, str]]:
    """Return the Hydra overrides taken from the config, in deterministic order.

    If the config omits ``model_type`` (required by Mussel), the model's entry in
    :data:`MUSSEL_MODEL_TYPES` is used; every other key appears only if the config sets it.
    """
    flat = _flatten(cfg.params)
    if "model_type" not in flat:
        flat.update(MUSSEL_MODEL_TYPES.get(cfg.model, {}))
    if "model_type" not in flat:
        raise ValueError(f"No Mussel model_type for model {cfg.model!r}; set params.model_type")
    leading = [key for key in _LEADING_KEYS if key in flat]
    rest = sorted(key for key in flat if key not in _LEADING_KEYS)
    return [(key, _format_value(flat[key])) for key in leading + rest]


def build_argv(
    cfg: BackendConfig,
    slide_path: str | Path,
    outputs: MusselOutputs,
    *,
    hydra_dir: str | Path | None = None,
) -> list[str]:
    """Build the ``tessellate_extract_features`` command for one slide.

    Args:
        cfg: Mussel backend config.
        slide_path: Slide to process.
        outputs: Where Mussel should write its h5/pt files.
        hydra_dir: Optional Hydra run directory (keeps Hydra logs out of the cwd).

    Returns:
        The argv list, starting with the Mussel command (without any ``uv run`` prefix).
    """
    if cfg.backend != "mussel":
        raise ValueError(f"build_argv needs a mussel config, got {cfg.backend!r}")
    argv = [
        MUSSEL_COMMAND,
        f"slide_path={_format_value(Path(slide_path).absolute())}",
        f"output_h5_path={_format_value(outputs.h5)}",
        f"output_pt_path={_format_value(outputs.pt)}",
    ]
    argv += [f"{key}={value}" for key, value in mussel_overrides(cfg)]
    if hydra_dir is not None:
        argv.append(f"hydra.run.dir={_format_value(hydra_dir)}")
    return argv


def _h5_rows(path: Path) -> int:
    import h5py

    try:
        with h5py.File(path, "r") as handle:
            for name in ("coords", "features"):
                if name in handle and handle[name].shape:
                    shape = handle[name].shape
                    # a 1-D features dataset is one slide-level embedding, not D rows
                    return int(shape[0]) if len(shape) > 1 else 1
    except OSError:
        return 0
    return 0


def is_complete(outputs: MusselOutputs) -> bool:
    """True if the h5 has at least one row, and the pt and provenance files exist.

    Tile-level outputs carry ``coords``; slide-level outputs (e.g. TITAN) carry only
    ``features``, so either dataset counts.
    """
    if not (outputs.h5.is_file() and outputs.pt.is_file() and outputs.provenance.is_file()):
        return False
    return _h5_rows(outputs.h5) > 0


def _collect_multi_model(cfg: BackendConfig, partial: MusselOutputs) -> None:
    """Rename Mussel's multi-model layout into the single-model file names.

    With both ``model_type`` and ``slide_model_type`` set, Mussel runs in multi-model
    mode and writes ``<dir>/<TYPE>/{h5,pt}/<slide_id>.features.{h5,pt}`` for each model
    (``dir`` = parent of ``output_h5_path``) instead of the requested paths.
    """
    if partial.h5.is_file():
        return
    flat = _flatten(cfg.params)
    targets = [
        (flat.get("slide_model_type"), partial.h5, partial.pt),
        (flat.get("model_type"), partial.tiles_h5, partial.tiles_pt),
    ]
    for model_type, h5_target, pt_target in targets:
        if not isinstance(model_type, str):
            continue
        for sub, target in (("h5", h5_target), ("pt", pt_target)):
            found = sorted((partial.dir / model_type / sub).glob(f"*.features.{sub}"))
            if len(found) == 1:
                os.replace(found[0], target)


def _short_tmpdir() -> Path:
    """Create a per-run temp dir with a short path under ``<project>/.tmp_build``.

    Unix-domain socket paths are limited to 108 bytes; PyTorch DataLoader workers and
    multiprocessing create sockets under ``TMPDIR``, and a TMPDIR nested inside a mirrored
    ``--out-root`` exceeds that limit and deadlocks Mussel's DataLoader.
    """
    import tempfile

    base = project_root() / ".tmp_build"
    base.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="mussel-", dir=base))


def _tail(text: str | bytes | None) -> str:
    if text is None:
        return ""
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    return text[-_ERROR_TAIL_CHARS:]


def _row(slide: Path, status: str, **fields: Any) -> dict[str, Any]:
    row = {
        "slide": str(slide),
        "status": status,
        "returncode": None,
        "seconds": 0.0,
        "error": "",
        "output_h5": "",
    }
    row.update(fields)
    return row


def run_slide(
    cfg: BackendConfig,
    slide_path: str | Path,
    *,
    command_prefix: Sequence[str],
    out_root: str | Path | None = None,
    force: bool = False,
    runner: Callable[..., Any] = subprocess.run,
    tool_version: str | None = None,
    tool_commit: str | None = MUSSEL_PINNED_COMMIT,
    env_lock: Path | None = None,
) -> dict[str, Any]:
    """Extract one slide with Mussel, isolating failures.

    Mussel writes into ``<stem>.mussel/.partial-<model>/``; ``TMPDIR`` points at a short
    per-run directory under ``<project>/.tmp_build`` (never ``/tmp``; see :func:`_short_tmpdir`); on success the files are moved into place and the
    provenance JSON is written last, so a present provenance file marks a complete run.

    Args:
        cfg: Mussel backend config.
        slide_path: Slide to process.
        command_prefix: Launcher for the Mussel environment, e.g.
            ``["uv", "run", "--frozen", "--project", "envs/mussel"]``.
        out_root: Optional root that mirrors the slide layout (see :func:`output_paths`).
        force: Re-run even if complete outputs exist.
        runner: ``subprocess.run``-compatible callable (injectable for tests).
        tool_version: Mussel version recorded in provenance.
        tool_commit: Mussel source commit recorded in provenance.
        env_lock: Lock file of the Mussel env; defaults to ``envs/mussel/uv.lock``.

    Returns:
        Status row: ``slide``, ``status`` (ok/skipped/failed), ``returncode``,
        ``seconds``, ``error`` (output tail on failure), ``output_h5``.
    """
    slide_path = Path(slide_path).absolute()
    outputs = output_paths(slide_path, cfg.model, out_root)
    if env_lock is None:
        env_lock = project_root() / "envs" / "mussel" / "uv.lock"

    if not force and is_complete(outputs):
        previous = read_provenance(outputs.provenance).get("params")
        if previous == json.loads(json.dumps(cfg.params)):
            return _row(slide_path, "skipped", output_h5=str(outputs.h5))
        return _row(
            slide_path,
            "failed",
            error=(
                f"{outputs.provenance} was produced with different params; "
                "use force=True or a different out_root"
            ),
            output_h5=str(outputs.h5),
        )
    if not slide_path.is_file():
        return _row(slide_path, "failed", error=f"slide not found: {slide_path}")

    partial = _outputs_in(outputs.dir / f".partial-{cfg.model}", cfg.model)
    if partial.dir.exists():
        shutil.rmtree(partial.dir)
    partial.dir.mkdir(parents=True)
    tmp_dir = _short_tmpdir()
    argv = list(command_prefix) + build_argv(
        cfg, slide_path, partial, hydra_dir=partial.dir / "hydra"
    )
    env = {**os.environ, "TMPDIR": str(tmp_dir), "HYDRA_FULL_ERROR": "1"}

    start = time.monotonic()
    returncode: int | None = None
    output = ""
    error = ""
    try:
        result = runner(argv, env=env, capture_output=True, text=True, check=False)
        returncode = int(result.returncode)
        output = (_tail(getattr(result, "stdout", "")) + "\n" + _tail(result.stderr)).strip()
        if returncode != 0:
            error = f"exit code {returncode}"
        else:
            _collect_multi_model(cfg, partial)
            if _h5_rows(partial.h5) == 0 or not partial.pt.is_file():
                error = "Mussel exited 0 but produced no h5 rows or no .pt file"
    except Exception as exc:  # per-slide isolation: never abort the batch
        error = f"{type(exc).__name__}: {exc}"
    seconds = round(time.monotonic() - start, 3)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    outputs.dir.mkdir(parents=True, exist_ok=True)
    if error:
        outputs.failed_log.write_text(" ".join(argv) + "\n\n" + output + "\n", encoding="utf-8")
        shutil.rmtree(partial.dir, ignore_errors=True)
        tail = _tail(output)[-500:]
        return _row(
            slide_path,
            "failed",
            returncode=returncode,
            seconds=seconds,
            error=f"{error}: {tail}" if tail else error,
        )

    for stale in (outputs.provenance, outputs.failed_log):
        stale.unlink(missing_ok=True)
    os.replace(partial.h5, outputs.h5)
    os.replace(partial.pt, outputs.pt)
    for source, target in (
        (partial.tiles_h5, outputs.tiles_h5),
        (partial.tiles_pt, outputs.tiles_pt),
    ):
        if source.is_file():
            os.replace(source, target)
    outputs.log.write_text(" ".join(argv) + "\n\n" + output + "\n", encoding="utf-8")
    prov = collect_provenance(
        cfg,
        tool_version=tool_version,
        tool_commit=tool_commit,
        env_lock=env_lock,
        extra={
            "slide_path": str(slide_path),
            "argv": argv,
            "seconds": seconds,
            "n_rows": _h5_rows(outputs.h5),
            "outputs": {
                "h5": outputs.h5.name,
                "pt": outputs.pt.name,
                "tiles_h5": outputs.tiles_h5.name if outputs.tiles_h5.is_file() else None,
            },
        },
    )
    write_provenance(outputs.provenance, prov)
    shutil.rmtree(partial.dir, ignore_errors=True)
    return _row(slide_path, "ok", returncode=returncode, seconds=seconds, output_h5=str(outputs.h5))


def probe_tool_version(
    command_prefix: Sequence[str], runner: Callable[..., Any] = subprocess.run
) -> str | None:
    """Return the installed Mussel version inside the Mussel env, or ``None``."""
    code = f"import importlib.metadata as m; print(m.version('{MUSSEL_PACKAGE}'))"
    try:
        result = runner(
            list(command_prefix) + ["python", "-c", code],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return str(result.stdout).strip() or None


def run_table(
    cfg: BackendConfig,
    slide_table: str | Path,
    *,
    command_prefix: Sequence[str],
    slide_col: str = "FILENAME",
    indices: Iterable[int] | None = None,
    out_root: str | Path | None = None,
    force: bool = False,
    runner: Callable[..., Any] = subprocess.run,
    tool_version: str | None = None,
    tool_commit: str | None = MUSSEL_PINNED_COMMIT,
    env_lock: Path | None = None,
) -> pd.DataFrame:
    """Run :func:`run_slide` over rows of a slide table.

    Args:
        cfg: Mussel backend config.
        slide_table: CSV with a slide path column.
        command_prefix: Launcher for the Mussel environment.
        slide_col: Column holding slide paths.
        indices: Positional row indices to process (SLURM array sharding); all rows if
            None. Indices past the end are dropped so the last shard may overshoot.
        out_root: Optional mirrored output root.
        force: Re-run complete outputs.
        runner: ``subprocess.run``-compatible callable.
        tool_version: Mussel version; probed once from the env if not given.
        tool_commit: Mussel source commit for provenance.
        env_lock: Mussel env lock file for provenance.

    Returns:
        One status row per processed slide, plus the table ``row`` index.
    """
    table = pd.read_csv(slide_table)
    if slide_col not in table.columns:
        raise ValueError(f"{slide_table} has no column {slide_col!r}")
    positions = list(range(len(table))) if indices is None else [int(i) for i in indices]
    if any(i < 0 for i in positions):
        raise IndexError(f"Negative row indices: {[i for i in positions if i < 0]}")
    positions = [i for i in positions if i < len(table)]
    if tool_version is None and positions:
        tool_version = probe_tool_version(command_prefix, runner)
    rows = []
    for position in positions:
        row = run_slide(
            cfg,
            table.iloc[position][slide_col],
            command_prefix=command_prefix,
            out_root=out_root,
            force=force,
            runner=runner,
            tool_version=tool_version,
            tool_commit=tool_commit,
            env_lock=env_lock,
        )
        rows.append({"row": position, **row})
    columns = ["row", "slide", "status", "returncode", "seconds", "error", "output_h5"]
    return pd.DataFrame(rows, columns=columns)
