"""Slide preprocessing utilities.

Currently: convert non-pyramidal WSIs to tiled pyramidal TIFFs via libvips.

Rationale: LazySlide's ``find_tissues`` falls back to loading the highest-
resolution level when a slide has ``n_levels == 1``. On a 78k×75k image that
easily exceeds any SLURM memory budget we'd reasonably set. Converting to a
pyramidal TIFF with a sensible tile size keeps the same pixels but lets the
downstream code work with a small thumbnail.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import NamedTuple

import pandas as pd

logger = logging.getLogger(__name__)


class ConversionResult(NamedTuple):
    """Outcome for a single slide in :func:`convert_non_pyramidal_slides`."""

    slide: str
    status: str  # "ok", "converted", "already_converted", "unreadable", "failed", "would_convert"
    new_path: Path | None
    n_levels: int | None
    detail: str | None = None


def _check_vips_available() -> None:
    if shutil.which("vips") is None:
        raise RuntimeError(
            "vips binary not found. Install with: "
            "conda install -c conda-forge libvips"
        )


def pyramid_levels(slide_path: Path) -> int | None:
    """Return the number of pyramid levels, or ``None`` if the slide cannot
    be opened by OpenSlide (e.g. format unsupported)."""
    try:
        import openslide
    except ImportError as e:
        raise RuntimeError("openslide-python is required") from e

    try:
        with openslide.OpenSlide(str(slide_path)) as s:
            return s.level_count
    except Exception as e:
        logger.debug("OpenSlide cannot read %s: %s", slide_path.name, e)
        return None


def convert_to_pyramidal(
    slide_path: Path,
    output_path: Path,
    tile_size: int = 256,
    quality: int = 90,
) -> bool:
    """Convert a single slide to a tiled, JPEG-compressed pyramidal TIFF.

    Returns ``True`` on success. On failure, removes any partial output and
    returns ``False``.
    """
    cmd = [
        "vips",
        "tiffsave",
        str(slide_path),
        str(output_path),
        "--pyramid",
        "--tile",
        f"--tile-width={tile_size}",
        f"--tile-height={tile_size}",
        "--compression=jpeg",
        f"--Q={quality}",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error("vips failed for %s: %s", slide_path.name, e.stderr.strip())
        if output_path.exists():
            output_path.unlink()
        return False


def convert_non_pyramidal_slides(
    slide_table: Path,
    output_table: Path | None = None,
    slide_column: str = "FILENAME",
    tile_size: int = 256,
    quality: int = 90,
    dry_run: bool = False,
) -> list[ConversionResult]:
    """Walk a slide table, converting any non-pyramidal slides in place.

    For each non-pyramidal slide we write ``<slide>.pyramidal.tiff`` alongside
    the original and rewrite the slide-table row to point at the new file. The
    original SVS is left untouched.

    Parameters
    ----------
    slide_table
        CSV with a ``FILENAME`` column (configurable via ``slide_column``).
    output_table
        Where to write the updated CSV. Defaults to
        ``<slide_table>_pyramidal.csv``.
    dry_run
        If True, only report what would happen; no files are written.
    """
    _check_vips_available()

    if output_table is None:
        output_table = slide_table.with_name(slide_table.stem + "_pyramidal.csv")

    df = pd.read_csv(slide_table)
    if slide_column not in df.columns:
        raise ValueError(f"Column {slide_column!r} not found in {slide_table}")

    results: list[ConversionResult] = []
    new_paths: list[str] = []

    for raw_path in df[slide_column]:
        if pd.isna(raw_path):
            results.append(ConversionResult(str(raw_path), "unreadable", None, None, "missing path"))
            new_paths.append(raw_path)
            continue

        slide = Path(str(raw_path))
        if not slide.is_file():
            results.append(ConversionResult(slide.name, "unreadable", None, None, "file not found"))
            new_paths.append(str(slide))
            continue

        n_levels = pyramid_levels(slide)
        if n_levels is None:
            results.append(ConversionResult(slide.name, "unreadable", None, None, "openslide open failed"))
            new_paths.append(str(slide))
            continue

        if n_levels > 1:
            results.append(ConversionResult(slide.name, "ok", slide, n_levels, None))
            new_paths.append(str(slide))
            continue

        # Non-pyramidal → needs conversion
        converted = slide.with_suffix(".pyramidal.tiff")

        if converted.exists():
            results.append(ConversionResult(slide.name, "already_converted", converted, n_levels, None))
            new_paths.append(str(converted))
            continue

        if dry_run:
            results.append(ConversionResult(slide.name, "would_convert", converted, n_levels, None))
            new_paths.append(str(slide))
            continue

        ok = convert_to_pyramidal(slide, converted, tile_size=tile_size, quality=quality)
        if ok:
            results.append(ConversionResult(slide.name, "converted", converted, n_levels, None))
            new_paths.append(str(converted))
        else:
            results.append(ConversionResult(slide.name, "failed", None, n_levels, "vips error"))
            new_paths.append(str(slide))

    df[slide_column] = new_paths
    df.to_csv(output_table, index=False)
    return results


def summarize(results: list[ConversionResult]) -> dict[str, int]:
    """Count results by status. Useful for CLI summary output."""
    counts: dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    return counts
