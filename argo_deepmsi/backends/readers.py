"""Read per-tile features from LazySlide zarr stores and Mussel h5/pt files.

Both readers return :class:`TileFeatures` in level-0 pixel coordinates, without
converting either format on disk. Nothing here imports LazySlide, wsidata, or Mussel.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .mussel import output_paths

# ARGO model name -> LazySlide feature-table key (``<key>_tiles``).
LAZYSLIDE_MODEL_KEYS = {"hoptimus0": "h-optimus-0", "titan": "conch_v1.5"}


@dataclass
class TileFeatures:
    """Per-tile features of one slide from one backend.

    Attributes:
        features: ``(N, D)`` float32 features.
        coords: ``(N, 2)`` int64 level-0 top-left ``(x, y)`` of each tile.
        tile_px_level0: Tile edge in level-0 pixels.
        tile_px: Tile edge in pixels at the extraction ``mpp``.
        mpp: Extraction microns per pixel, if recorded.
        backend: ``"lazyslide"`` or ``"mussel"``.
        model: Model name as stored by the backend.
        slide_id: Slide identifier (file stem).
        provenance: Backend-specific metadata (tile spec, manifest, provenance JSON).
    """

    features: np.ndarray
    coords: np.ndarray
    tile_px_level0: float
    tile_px: int
    mpp: float | None
    backend: str
    model: str
    slide_id: str
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.features = np.asarray(self.features, dtype=np.float32)
        self.coords = np.asarray(self.coords, dtype=np.int64).reshape(-1, 2)
        if self.features.ndim != 2 or self.features.shape[0] != self.coords.shape[0]:
            raise ValueError(
                f"features {self.features.shape} do not align with coords {self.coords.shape}"
            )

    @property
    def n_tiles(self) -> int:
        return int(self.coords.shape[0])

    @property
    def dim(self) -> int:
        return int(self.features.shape[1])


def _read_group_attrs(group: Path) -> dict[str, Any]:
    """Read zarr v3 ``zarr.json`` attributes or zarr v2 ``.zattrs`` without importing zarr."""
    v3 = group / "zarr.json"
    if v3.is_file():
        return json.loads(v3.read_text(encoding="utf-8")).get("attributes", {}) or {}
    v2 = group / ".zattrs"
    if v2.is_file():
        return json.loads(v2.read_text(encoding="utf-8")) or {}
    return {}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def read_lazyslide(zarr_path: str | Path, model_key: str) -> TileFeatures:
    """Read ``<zarr>/tables/<model_key>_tiles`` joined to its tile polygons.

    Tile polygons come from ``<zarr>/shapes/<region>/shapes.parquet`` (``tile_id`` plus
    a WKB ``geometry`` in level-0 pixels under an identity transform), where ``region``
    is the table's spatialdata region (``tiles`` by default). The tile spec comes from
    the store root attributes (``tile_spec``).

    Args:
        zarr_path: LazySlide/spatialdata store (``<slide_stem>.zarr``).
        model_key: LazySlide feature key, e.g. ``"h-optimus-0"`` (without ``_tiles``).
    """
    import anndata as ad
    import pyarrow.parquet as pq
    import shapely

    zarr_path = Path(zarr_path)
    table_dir = zarr_path / "tables" / f"{model_key}_tiles"
    if not table_dir.is_dir():
        raise FileNotFoundError(f"No table {table_dir.name} in {zarr_path}")
    table_attrs = _read_group_attrs(table_dir)
    region = table_attrs.get("region") or "tiles"
    instance_key = table_attrs.get("instance_key") or "tile_id"

    adata = ad.read_zarr(str(table_dir))
    features = np.asarray(adata.X, dtype=np.float32)
    tile_ids = adata.obs[instance_key].to_numpy(dtype=np.int64)

    shapes_path = zarr_path / "shapes" / region / "shapes.parquet"
    shapes = pq.read_table(shapes_path, columns=["tile_id", "geometry"]).to_pydict()
    bounds = shapely.bounds(shapely.from_wkb(np.asarray(shapes["geometry"], dtype=object)))
    shape_ids = np.asarray(shapes["tile_id"], dtype=np.int64)
    order = np.argsort(shape_ids, kind="stable")
    pos = np.searchsorted(shape_ids[order], tile_ids)
    pos = np.clip(pos, 0, len(order) - 1)
    rows = order[pos]
    if len(order) == 0 or not np.array_equal(shape_ids[rows], tile_ids):
        raise ValueError(f"{table_dir.name}: tile_ids missing from {shapes_path}")
    tile_bounds = bounds[rows]
    coords = np.rint(tile_bounds[:, :2]).astype(np.int64)

    root_attrs = _read_group_attrs(zarr_path)
    spec = (root_attrs.get("tile_spec") or {}).get(region, {})
    slide_props = root_attrs.get("slide_properties") or {}
    tile_px = int(spec.get("width") or 0)
    base_downsample = spec.get("base_downsample")
    if tile_px and base_downsample:
        tile_px_level0 = float(tile_px * base_downsample)
    else:
        widths = tile_bounds[:, 2] - tile_bounds[:, 0]
        tile_px_level0 = float(np.median(widths)) if len(widths) else float(tile_px)
        tile_px = tile_px or int(round(tile_px_level0))
    return TileFeatures(
        features=features,
        coords=coords,
        tile_px_level0=tile_px_level0,
        tile_px=tile_px,
        mpp=spec.get("mpp"),
        backend="lazyslide",
        model=model_key,
        slide_id=zarr_path.name.removesuffix(".zarr"),
        provenance={
            "store": str(zarr_path),
            "table": table_dir.name,
            "tile_spec": spec,
            "slide_mpp": slide_props.get("mpp"),
            "argo_manifest": _read_json(zarr_path / "argo_manifest.json"),
            "provenance": _read_json(lazyslide_provenance_path(zarr_path, model_key)),
        },
    )


def read_lazyslide_slide_embedding(
    zarr_path: str | Path, model_key: str, agg_key: str = "agg_slide"
) -> np.ndarray:
    """Read a slide-encoder output (e.g. TITAN) stored on a LazySlide feature table.

    LazySlide writes slide representations to ``uns["agg_ops"][agg_key]["features"]``
    (or ``varm[agg_key]`` when the dimension matches the tile features).
    """
    import anndata as ad

    adata = ad.read_zarr(str(Path(zarr_path) / "tables" / f"{model_key}_tiles"))
    agg_ops = adata.uns.get("agg_ops", {})
    if agg_key in agg_ops and "features" in agg_ops[agg_key]:
        value = agg_ops[agg_key]["features"]
    elif agg_key in adata.varm:
        value = adata.varm[agg_key]
    elif agg_key in adata.uns:
        value = adata.uns[agg_key]
    else:
        raise KeyError(f"No {agg_key!r} slide embedding on {model_key}_tiles in {zarr_path}")
    return np.asarray(value, dtype=np.float32).ravel()


def _to_float32(array: np.ndarray) -> np.ndarray:
    """Convert h5 feature arrays to float32; bfloat16 is stored as opaque ``|V2``."""
    array = np.asarray(array)
    if array.dtype.kind == "V" and array.dtype.itemsize == 2:
        bits = array.view(np.uint16).astype(np.uint32) << 16
        return bits.view(np.float32)
    return array.astype(np.float32, copy=False)


def _load_pt(pt_path: Path) -> np.ndarray:
    import torch

    obj = torch.load(pt_path, map_location="cpu", weights_only=True)
    if isinstance(obj, dict):
        obj = obj["features"]
    return obj.detach().to(torch.float32).cpu().numpy()


def _attr(attrs: Any, key: str) -> Any:
    if key not in attrs:
        return None
    value = attrs[key]
    if isinstance(value, bytes):
        value = value.decode()
    if isinstance(value, np.ndarray) and value.size == 1:
        value = value.item()
    if isinstance(value, np.generic):
        value = value.item()
    return value


def _mussel_names(h5_path: Path) -> tuple[str, str, Path]:
    name = h5_path.name
    model = name.split(".")[0]
    slide_id = h5_path.parent.name.removesuffix(".mussel")
    return model, slide_id, h5_path.parent / f"{model}.provenance.json"


def read_mussel(h5_path: str | Path, pt_path: str | Path | None = None) -> TileFeatures:
    """Read Mussel tile features (mirrors ``mussel.utils.file.load_features_from_h5``).

    ``coords`` (level-0 top-left) always come from the h5. Features come from the h5
    ``features`` dataset when present, otherwise from ``pt_path`` (``torch.load``).
    ``coords.attrs["patch_size"]`` is Mussel's native (level-0) patch size,
    ``patch_size_to_resize_to_for_desired_mpp`` the tile edge at ``mpp``, and
    ``native_mpp`` the slide resolution; missing attrs are inferred where possible.

    Args:
        h5_path: ``<MODEL>.features.h5``.
        pt_path: Optional ``<MODEL>.features.pt``; defaults to the sibling file.
    """
    import h5py

    h5_path = Path(h5_path)
    if pt_path is None:
        sibling = h5_path.with_name(h5_path.name.replace(".features.h5", ".features.pt"))
        pt_path = sibling if sibling != h5_path and sibling.is_file() else None
    with h5py.File(h5_path, "r") as handle:
        if "coords" not in handle:
            raise KeyError(f"{h5_path} has no coords (slide-level output?)")
        coords = np.asarray(handle["coords"][:])
        attrs = {key: _attr(handle["coords"].attrs, key) for key in handle["coords"].attrs}
        features = _to_float32(handle["features"][:]) if "features" in handle else None
    if features is None:
        if pt_path is None:
            raise FileNotFoundError(f"{h5_path} has no features dataset and no .pt file")
        features = _load_pt(Path(pt_path))

    mpp = attrs.get("mpp") or attrs.get("tile_mpp")
    native_mpp = attrs.get("native_mpp")
    patch_size = attrs.get("patch_size")
    tile_px = attrs.get("patch_size_to_resize_to_for_desired_mpp") or attrs.get("tile_patch_size")
    if tile_px is None and patch_size and mpp and native_mpp:
        tile_px = int(round(float(patch_size) * float(native_mpp) / float(mpp)))
    if patch_size is None:
        diffs = np.diff(np.unique(coords[:, 0])) if len(coords) > 1 else np.array([])
        patch_size = float(diffs[diffs > 0].min()) if (diffs > 0).any() else float(tile_px or 0)
    model, slide_id, prov_path = _mussel_names(h5_path)
    prov = _read_json(prov_path)
    return TileFeatures(
        features=features,
        coords=coords,
        tile_px_level0=float(patch_size),
        tile_px=int(tile_px or round(float(patch_size))),
        mpp=float(mpp) if mpp is not None else None,
        backend="mussel",
        model=prov.get("model", model),
        slide_id=slide_id,
        provenance={"h5": str(h5_path), "coords_attrs": attrs, "provenance": prov},
    )


def read_mussel_slide_embedding(
    h5_path: str | Path, pt_path: str | Path | None = None
) -> np.ndarray:
    """Read a Mussel slide-encoder output (e.g. ``TITAN_SLIDE``) as a 1-D float32 vector."""
    import h5py

    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as handle:
        if "features" in handle:
            return _to_float32(handle["features"][:]).ravel()
    if pt_path is None:
        pt_path = h5_path.with_name(h5_path.name.replace(".features.h5", ".features.pt"))
    return _load_pt(Path(pt_path)).ravel()


def lazyslide_provenance_path(zarr_path: str | Path, model_key: str) -> Path:
    """Return ``<store>.<model_key>.provenance.json`` beside a LazySlide store.

    Kept outside the zarr so it never collides with spatialdata elements.
    """
    zarr_path = Path(zarr_path)
    return zarr_path.parent / f"{zarr_path.name}.{model_key}.provenance.json"


def lazyslide_store(slide_path: str | Path, out_root: str | Path | None = None) -> Path:
    """Return the store ``argo extract --backend lazyslide`` writes for a slide.

    ``<slide_dir>/<slide_stem>.zarr`` by default; with ``out_root`` the slide's absolute
    directory is mirrored under the root (``<out_root>/<slide_dir>/<slide_stem>.zarr``).
    """
    slide_path = Path(slide_path).absolute()
    parent = slide_path.parent
    if out_root is not None:
        parent = Path(out_root).absolute() / parent.relative_to(parent.anchor)
    return parent / f"{slide_path.stem}.zarr"


def read_tiles(
    slide_path: str | Path,
    backend: str,
    model: str,
    *,
    out_root: str | Path | None = None,
    model_key: str | None = None,
) -> TileFeatures:
    """Locate and read one slide's tile features for a backend.

    Args:
        slide_path: Whole-slide image path.
        backend: ``"lazyslide"`` or ``"mussel"``.
        model: ARGO model name (``"hoptimus0"``, ``"titan"``, ...). For Mussel slide
            encoders (TITAN) the kept patch-encoder tiles (``<model>.tiles.features.h5``)
            are read.
        out_root: Mirrored output root, if outputs are not next to the slide.
        model_key: LazySlide table key override; defaults to :data:`LAZYSLIDE_MODEL_KEYS`.
    """
    if backend == "lazyslide":
        key = model_key or LAZYSLIDE_MODEL_KEYS.get(model, model)
        return read_lazyslide(lazyslide_store(slide_path, out_root), key)
    if backend == "mussel":
        outputs = output_paths(slide_path, model, out_root)
        if outputs.tiles_h5.is_file():
            pt = outputs.tiles_pt if outputs.tiles_pt.is_file() else None
            tiles = read_mussel(outputs.tiles_h5, pt)
        else:
            tiles = read_mussel(outputs.h5, outputs.pt if outputs.pt.is_file() else None)
        tiles.model = model
        return tiles
    raise ValueError(f"Unknown backend {backend!r}")
