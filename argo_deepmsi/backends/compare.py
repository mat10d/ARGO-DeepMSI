"""Characterise divergence between two backends' features for the same slides."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .readers import TileFeatures

IDENTICAL = "identical up to float noise"
GRID_ONLY = "grid-only divergence"
LOCATED = "located divergence"
_MATCH_FRAC_FOR_IDENTICAL = 0.99


def _one_to_one(ia: np.ndarray, ib: np.ndarray, dist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.lexsort((ib, ia, dist))
    used_a: set[int] = set()
    used_b: set[int] = set()
    keep = []
    for k in order:
        a, b = int(ia[k]), int(ib[k])
        if a in used_a or b in used_b:
            continue
        used_a.add(a)
        used_b.add(b)
        keep.append(k)
    keep_arr = np.asarray(keep, dtype=np.int64)
    final = np.argsort(ia[keep_arr], kind="stable")
    return ia[keep_arr][final], ib[keep_arr][final]


def _candidate_pairs(
    ca: np.ndarray, cb: np.ndarray, radius: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All (i, j) with Chebyshev distance <= radius between ca[i] and cb[j]."""
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        cKDTree = None
    if cKDTree is not None:
        pairs = cKDTree(cb).query_ball_point(ca, r=radius, p=np.inf)
        ia = np.repeat(np.arange(len(ca)), [len(p) for p in pairs])
        ib = np.fromiter((j for p in pairs for j in p), dtype=np.int64, count=len(ia))
    else:
        cell = max(radius, 1.0)
        buckets: dict[tuple[int, int], list[int]] = {}
        for j, (x, y) in enumerate(np.floor(cb / cell).astype(np.int64)):
            buckets.setdefault((int(x), int(y)), []).append(j)
        ia_list, ib_list = [], []
        for i, (x, y) in enumerate(np.floor(ca / cell).astype(np.int64)):
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for j in buckets.get((int(x) + dx, int(y) + dy), ()):
                        if np.max(np.abs(cb[j] - ca[i])) <= radius:
                            ia_list.append(i)
                            ib_list.append(j)
        ia = np.asarray(ia_list, dtype=np.int64)
        ib = np.asarray(ib_list, dtype=np.int64)
    dist = np.max(np.abs(cb[ib] - ca[ia]), axis=1) if len(ia) else np.zeros(0)
    return ia.astype(np.int64), ib.astype(np.int64), dist


def _edge(a: TileFeatures, b: TileFeatures) -> float:
    return float(min(a.tile_px_level0, b.tile_px_level0))


ANCHORS = ("topleft", "center")


def _anchored(t: TileFeatures, anchor: str) -> np.ndarray:
    """Level-0 tile top-left (``topleft``) or centre (``center``) as float64."""
    if anchor not in ANCHORS:
        raise ValueError(f"anchor must be one of {ANCHORS}, got {anchor!r}")
    coords = t.coords.astype(np.float64)
    return coords + t.tile_px_level0 / 2.0 if anchor == "center" else coords


def match_tiles(
    a: TileFeatures, b: TileFeatures, tol_frac: float = 0.25, anchor: str = "topleft"
) -> tuple[np.ndarray, np.ndarray]:
    """Match tiles one-to-one by level-0 anchor within ``tol_frac`` x tile edge.

    Distance is Chebyshev (both |dx| and |dy| within tolerance); the tile edge is the
    smaller of the two backends' level-0 edges. Closest pairs are matched first.
    ``anchor="center"`` compares tile centres, which is the fairer choice when the two
    grids use different tile sizes.

    Returns:
        ``(idx_a, idx_b)`` index arrays of equal length, sorted by ``idx_a``.
    """
    empty = np.zeros(0, dtype=np.int64)
    if a.n_tiles == 0 or b.n_tiles == 0:
        return empty, empty
    radius = tol_frac * _edge(a, b)
    ia, ib, dist = _candidate_pairs(_anchored(a, anchor), _anchored(b, anchor), radius)
    if len(ia) == 0:
        return empty, empty
    return _one_to_one(ia, ib, dist)


def _grid_offset(a: TileFeatures, b: TileFeatures) -> tuple[float, float]:
    """Median (dx, dy) from each ``a`` tile to its nearest ``b`` tile within half an edge."""
    if a.n_tiles == 0 or b.n_tiles == 0:
        return float("nan"), float("nan")
    ia, ib, dist = _candidate_pairs(
        a.coords.astype(np.float64), b.coords.astype(np.float64), 0.5 * _edge(a, b)
    )
    if len(ia) == 0:
        return float("nan"), float("nan")
    ia, ib = _one_to_one(ia, ib, dist)
    delta = b.coords[ib] - a.coords[ia]
    return float(np.median(delta[:, 0])), float(np.median(delta[:, 1]))


def coverage_iou(a: TileFeatures, b: TileFeatures, cells_per_tile: int = 8) -> float:
    """IoU of the level-0 areas covered by two tile sets (tissue-coverage agreement).

    Rasterises both tile sets on a common grid with cell edge
    ``min(edge_a, edge_b) / cells_per_tile`` level-0 px, independent of grid origin.
    """
    if a.n_tiles == 0 or b.n_tiles == 0:
        return float("nan")
    cell = _edge(a, b) / cells_per_tile
    origin = np.minimum(a.coords.min(axis=0), b.coords.min(axis=0)).astype(np.float64)
    far = np.maximum(
        a.coords.max(axis=0) + a.tile_px_level0, b.coords.max(axis=0) + b.tile_px_level0
    )
    shape = tuple(int(v) for v in np.ceil((far - origin) / cell)[::-1] + 1)

    def raster(t: TileFeatures) -> np.ndarray:
        mask = np.zeros(shape, dtype=bool)
        lo = np.floor((t.coords - origin) / cell + 0.5).astype(np.int64)
        hi = np.floor((t.coords + t.tile_px_level0 - origin) / cell + 0.5).astype(np.int64)
        for (x0, y0), (x1, y1) in zip(lo, hi):
            mask[y0:y1, x0:x1] = True
        return mask

    ma, mb = raster(a), raster(b)
    union = np.logical_or(ma, mb).sum()
    return float(np.logical_and(ma, mb).sum() / union) if union else float("nan")


def _rowwise_cosine(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    denom = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, np.sum(x * y, axis=1) / denom, np.nan)


def _cosine(x: np.ndarray, y: np.ndarray) -> float:
    return float(_rowwise_cosine(x[None, :], y[None, :])[0])


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64) - x.mean()
    y = y.astype(np.float64) - y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / denom) if denom > 0 else float("nan")


def compare_pair(
    a: TileFeatures, b: TileFeatures, tol_frac: float = 0.25, anchor: str = "topleft"
) -> dict[str, Any]:
    """Tile-grid agreement and feature agreement for one slide.

    Feature metrics (cosine, abs/relative differences) are computed on matched tiles
    only; slide-mean cosine and Pearson r are reported for all tiles and matched tiles.
    """
    nan = float("nan")
    idx_a, idx_b = match_tiles(a, b, tol_frac, anchor)
    n_matched = int(len(idx_a))
    dx, dy = _grid_offset(a, b)
    out: dict[str, Any] = {
        "backend_a": a.backend,
        "backend_b": b.backend,
        "model_a": a.model,
        "model_b": b.model,
        "n_a": a.n_tiles,
        "n_b": b.n_tiles,
        "n_matched": n_matched,
        "frac_matched_a": n_matched / a.n_tiles if a.n_tiles else nan,
        "frac_matched_b": n_matched / b.n_tiles if b.n_tiles else nan,
        "tile_px_level0_a": a.tile_px_level0,
        "tile_px_level0_b": b.tile_px_level0,
        "grid_dx_median": dx,
        "grid_dy_median": dy,
        "coverage_iou": coverage_iou(a, b),
        "dim_a": a.dim,
        "dim_b": b.dim,
        "dim_match": a.dim == b.dim,
    }
    metrics = {
        key: nan
        for key in (
            "cos_median",
            "cos_p05",
            "cos_min",
            "max_abs_diff",
            "mean_abs_diff",
            "rel_l2",
            "slide_mean_cos_all",
            "slide_mean_pearson_all",
            "slide_mean_cos_matched",
            "slide_mean_pearson_matched",
            "cos_median_exact",
            "cos_p05_exact",
        )
    }
    exact = (
        np.all(a.coords[idx_a] == b.coords[idx_b], axis=1) if n_matched else np.zeros(0, dtype=bool)
    )
    # Tiles whose level-0 top-left coincides exactly (identical boxes when edges match)
    out["n_exact"] = int(exact.sum())
    out.update(metrics)
    if not out["dim_match"]:
        return out
    if a.n_tiles and b.n_tiles:
        mean_a, mean_b = a.features.mean(axis=0), b.features.mean(axis=0)
        out["slide_mean_cos_all"] = _cosine(mean_a, mean_b)
        out["slide_mean_pearson_all"] = _pearson(mean_a, mean_b)
    if n_matched:
        fa = a.features[idx_a].astype(np.float64)
        fb = b.features[idx_b].astype(np.float64)
        cos = _rowwise_cosine(fa, fb)
        diff = np.abs(fa - fb)
        norm_a = np.linalg.norm(fa)
        out.update(
            {
                "cos_median": float(np.nanmedian(cos)),
                "cos_p05": float(np.nanpercentile(cos, 5)),
                "cos_min": float(np.nanmin(cos)),
                "max_abs_diff": float(diff.max()),
                "mean_abs_diff": float(diff.mean()),
                "rel_l2": float(np.linalg.norm(fa - fb) / norm_a) if norm_a > 0 else nan,
                "slide_mean_cos_matched": _cosine(fa.mean(axis=0), fb.mean(axis=0)),
                "slide_mean_pearson_matched": _pearson(fa.mean(axis=0), fb.mean(axis=0)),
            }
        )
        if exact.any():
            out["cos_median_exact"] = float(np.nanmedian(cos[exact]))
            out["cos_p05_exact"] = float(np.nanpercentile(cos[exact], 5))
    return out


def compare_slides(
    pairs: Iterable[tuple[str, TileFeatures, TileFeatures]],
    tol_frac: float = 0.25,
    anchor: str = "topleft",
) -> pd.DataFrame:
    """Run :func:`compare_pair` for ``(slide_id, a, b)`` triples; one row per slide."""
    rows = [
        {"slide_id": slide_id, **compare_pair(a, b, tol_frac, anchor)} for slide_id, a, b in pairs
    ]
    return pd.DataFrame(rows)


def verdict(df: pd.DataFrame, cos_tol: float = 0.999) -> str:
    """Summarise a comparison table as one of three verdicts.

    * ``identical up to float noise``: every slide's matched-tile 5th-percentile cosine
      is >= ``cos_tol`` and >= 99% of tiles match on both sides.
    * ``grid-only divergence``: matched tiles agree to ``cos_tol`` but the tile sets
      differ (segmentation / grid origin).
    * ``located divergence``: features disagree on matched tiles, dimensions differ, or
      nothing matched; the offending slides are named.
    """
    if df.empty:
        return f"{LOCATED}: no slides compared"
    if not df["dim_match"].all():
        bad = ", ".join(map(str, df.loc[~df["dim_match"], "slide_id"]))
        return f"{LOCATED}: feature dimensions differ ({bad})"
    unmatched = df["n_matched"] == 0
    if unmatched.any():
        bad = ", ".join(map(str, df.loc[unmatched, "slide_id"]))
        return f"{LOCATED}: no matched tiles ({bad})"
    low = ~(df["cos_p05"] >= cos_tol)
    if low.any():
        detail = ", ".join(
            f"{row.slide_id} (cos p05={row.cos_p05:.4f}, median={row.cos_median:.4f})"
            for row in df.loc[low].itertuples()
        )
        return f"{LOCATED}: matched-tile cosine below {cos_tol} on {detail}"
    min_frac = float(np.nanmin(df[["frac_matched_a", "frac_matched_b"]].to_numpy()))
    if min_frac >= _MATCH_FRAC_FOR_IDENTICAL:
        return f"{IDENTICAL} (cos p05 >= {cos_tol}; >= {min_frac:.1%} tiles matched)"
    return (
        f"{GRID_ONLY}: matched tiles agree (cos p05 >= {cos_tol}) but only "
        f"{min_frac:.1%} of tiles match on the worst side"
    )


def _format_cell(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        return "nan" if np.isnan(value) else f"{value:.6g}"
    return str(value).replace("|", "\\|")


def _markdown_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(map(str, df.columns)) + " |"
    rule = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(_format_cell(v) for v in row) + " |" for row in df.itertuples(False)]
    return "\n".join([header, rule, *body])


def write_report(
    df: pd.DataFrame,
    out_dir: str | Path,
    title: str,
    *,
    stem: str | None = None,
    cos_tol: float = 0.999,
    notes: str | None = None,
) -> tuple[Path, Path]:
    """Write ``<stem>.csv`` and ``<stem>.md`` (title, verdict, table) to ``out_dir``.

    Returns:
        ``(csv_path, markdown_path)``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = stem or re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_").lower() or "compare"
    csv_path = out_dir / f"{stem}.csv"
    md_path = out_dir / f"{stem}.md"
    df.to_csv(csv_path, index=False)
    lines = [f"# {title}", "", f"**Verdict:** {verdict(df, cos_tol)}", ""]
    if notes:
        lines += [notes, ""]
    lines += [_markdown_table(df), ""]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, md_path


def compare_slide_embeddings(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    """Compare two slide-level vectors (e.g. TITAN from each backend)."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        return {"dim_a": a.size, "dim_b": b.size, "dim_match": False}
    norm_a = np.linalg.norm(a)
    return {
        "dim_a": a.size,
        "dim_b": b.size,
        "dim_match": True,
        "cosine": _cosine(a, b),
        "pearson": _pearson(a, b),
        "max_abs_diff": float(np.max(np.abs(a - b))) if a.size else float("nan"),
        "rel_l2": float(np.linalg.norm(a - b) / norm_a) if norm_a > 0 else float("nan"),
    }
