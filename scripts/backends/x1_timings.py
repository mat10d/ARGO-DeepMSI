"""Collect X1 per-slide wall times into results/analysis/backends/timings.csv.

Two clocks per (slide, run): ``wall_s`` is the job-level time around one ``argo extract``
call (uv start-up, imports, model load, tiling, extraction, write), and ``argo_s`` is the
per-slide time recorded in the backend provenance JSON.

Usage:
    uv run --frozen python scripts/backends/x1_timings.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from argo_deepmsi.backends.mussel import output_paths
from argo_deepmsi.backends.provenance import read_provenance
from argo_deepmsi.backends.readers import lazyslide_provenance_path, lazyslide_store

ROOT = Path("results/analysis/backends")
STORES = ROOT / "stores"
LAZYSLIDE_KEYS = {
    "lazyslide_256": "h-optimus-0",
    "lazyslide_224": "h-optimus-0",
    "lazyslide_224_amp": "h-optimus-0",
    "lazyslide_512": "conch_v1.5",
}
MUSSEL_MODELS = {"mussel_hoptimus0": "hoptimus0", "mussel_titan": "titan"}


def _provenance(name: str, slide: Path) -> dict:
    if name in LAZYSLIDE_KEYS:
        store = lazyslide_store(slide, STORES / name)
        path = lazyslide_provenance_path(store, LAZYSLIDE_KEYS[name])
    else:
        path = output_paths(slide, MUSSEL_MODELS[name], STORES / "mussel").provenance
    return read_provenance(path) if path.is_file() else {}


def _mussel_tiles(slide: Path) -> int | None:
    """CONCH1_5 tile count behind a Mussel TITAN run (the slide h5 holds one vector)."""
    import h5py

    path = output_paths(slide, "titan", STORES / "mussel").tiles_h5
    if not path.is_file():
        return None
    with h5py.File(path, "r") as handle:
        return int(handle["coords"].shape[0])


def main() -> None:
    slides = pd.read_csv(ROOT / "x1_slides.csv")
    frames = [pd.read_csv(path, sep="\t") for path in sorted((ROOT / "timings").glob("*.tsv"))]
    df = pd.concat(frames, ignore_index=True)
    df["slide"] = [Path(slides.loc[row, "FILENAME"]).name for row in df["row"]]
    df["site"] = [slides.loc[row, "SITE"] for row in df["row"]]
    extra = [
        _provenance(name, Path(slides.loc[row, "FILENAME"])).get("extra", {})
        for name, row in zip(df["name"], df["row"])
    ]
    df["argo_s"] = [e.get("seconds") for e in extra]
    df["n_tiles"] = [
        _mussel_tiles(Path(slides.loc[row, "FILENAME"]))
        if name == "mussel_titan"
        else e.get("n_tiles", e.get("n_rows"))
        for name, row, e in zip(df["name"], df["row"], extra)
    ]
    df = df.sort_values(["name", "row"])
    df.to_csv(ROOT / "timings.csv", index=False)
    summary = df.groupby("name").agg(
        slides=("row", "count"),
        wall_s_median=("wall_s", "median"),
        wall_s_total=("wall_s", "sum"),
        argo_s_median=("argo_s", "median"),
        n_tiles_median=("n_tiles", "median"),
    )
    summary.to_csv(ROOT / "timings_summary.csv")
    print(summary.round(1).to_string())


if __name__ == "__main__":
    main()
