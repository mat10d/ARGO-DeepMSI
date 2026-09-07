"""Small compatibility boundary for LazySlide's model package split.

LazySlide 0.11 moved model definitions into ``lazyslide-models``.  Keeping the
compatibility logic here prevents project models from depending on either
package's private module layout.
"""

from __future__ import annotations

try:  # LazySlide >=0.11
    from lazyslide_models import MODEL_REGISTRY, ImageModel, ModelTask, hf_access, register
except ImportError:  # LazySlide 0.10 (support reading existing environments)
    from lazyslide.models import MODEL_REGISTRY, ImageModel, ModelTask, register
    from lazyslide.models._utils import hf_access

__all__ = [
    "MODEL_REGISTRY",
    "ImageModel",
    "ModelTask",
    "hf_access",
    "register",
]
