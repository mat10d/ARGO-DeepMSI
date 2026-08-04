"""Project-local model definitions layered on top of LazySlide's registry.

Importing this package registers every model class it defines into LazySlide's
global ``MODEL_REGISTRY``, so they resolve by name through the normal
``zs.tl.feature_extraction(model="<name>")`` path — no change to the extraction
pipeline required. Import this module anywhere the registry must be populated
(feature_extraction, the dask worker).
"""

from __future__ import annotations

from . import waiv  # noqa: F401  (import triggers @register side effects)
from .ctranspath import load_trainable_ctranspath
from .wagner import load_wagner

__all__ = ["load_trainable_ctranspath", "load_wagner", "waiv"]
