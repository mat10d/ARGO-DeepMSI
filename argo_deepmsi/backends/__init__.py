"""Extraction backends (LazySlide, Mussel): configs, runners, readers, and comparison.

Submodules import only light dependencies at module scope (numpy, pandas); h5py,
pyarrow, shapely, anndata, scipy, and torch are imported inside the functions that
need them so this package is importable from the core environment.
"""

from __future__ import annotations

BACKENDS = ("lazyslide", "mussel")

__all__ = ["BACKENDS"]
