"""
ARGO-DeepMSI: MSI prediction from whole slide images using LazySlide.

A simplified pipeline for:
- Data ingestion from REDCap
- Feature extraction with foundation models (UNI, Virchow, etc.)
- Visualization and exploration
- Training lightweight classifiers

Usage:
    pip install -e .
    argo --help
"""

__version__ = "0.2.0"

# ---------------------------------------------------------------------------
# HuggingFace cache default.
#
# Runs BEFORE any downstream import (huggingface_hub, transformers,
# lazyslide) so the right cache is picked up globally. Rules:
#
#   1. If the user set $HF_HOME explicitly, respect it. Always.
#   2. Else if the package lives inside a checkout (pyproject.toml sits one
#      level above the package), default to <repo>/.huggingface_cache —
#      create if missing, and export HF_HOME + HF_HUB_CACHE +
#      TRANSFORMERS_CACHE so every downstream library agrees.
#   3. Else leave everything unset. Downstream libs default to
#      ~/.cache/huggingface; we log a warning so the user notices.
#
# Override anytime by exporting HF_HOME in your shell or .env.
# ---------------------------------------------------------------------------


def _configure_hf_cache() -> None:
    import os
    from pathlib import Path
    import warnings

    if os.environ.get("HF_HOME"):
        return  # user chose explicitly; don't second-guess

    here = Path(__file__).resolve().parent
    repo_root = here.parent
    if not (repo_root / "pyproject.toml").exists():
        warnings.warn(
            "argo_deepmsi: HF_HOME is not set and the package is not in a "
            "checkout layout — HuggingFace will default to "
            "~/.cache/huggingface. Export HF_HOME to a large scratch volume "
            "before using gated or large models.",
            stacklevel=2,
        )
        return

    cache = repo_root / ".huggingface_cache"
    cache.mkdir(parents=True, exist_ok=True)
    cache_str = str(cache)
    os.environ["HF_HOME"] = cache_str
    # huggingface_hub >= 0.20 reads HF_HUB_CACHE; transformers reads
    # TRANSFORMERS_CACHE. Set both so pre-existing shells behave the same.
    os.environ.setdefault("HF_HUB_CACHE", str(cache / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache / "hub"))


_configure_hf_cache()
del _configure_hf_cache


from . import io_utils
from . import data_ingestion
from . import feature_extraction
from . import visualization
from . import training

__all__ = [
    "io_utils",
    "data_ingestion",
    "feature_extraction",
    "visualization",
    "training",
]
