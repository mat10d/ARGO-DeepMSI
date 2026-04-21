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


def _configure_hf_env() -> None:
    """Set HF_HOME + load .env so HF_TOKEN / HF_HOME are consistent across
    argo CLI, pytest, and SLURM jobs started from a checkout. See module
    docstring for rules."""
    import os
    from pathlib import Path
    import warnings

    here = Path(__file__).resolve().parent
    repo_root = here.parent
    in_checkout = (repo_root / "pyproject.toml").exists()

    # Step 1: HF_HOME default (only applies in a checkout; user override wins)
    if not os.environ.get("HF_HOME"):
        if not in_checkout:
            warnings.warn(
                "argo_deepmsi: HF_HOME is not set and the package is not in "
                "a checkout layout — HuggingFace will default to "
                "~/.cache/huggingface. Export HF_HOME to a large scratch "
                "volume before using gated or large models.",
                stacklevel=2,
            )
        else:
            cache = repo_root / ".huggingface_cache"
            cache.mkdir(parents=True, exist_ok=True)
            os.environ["HF_HOME"] = str(cache)
            os.environ.setdefault("HF_HUB_CACHE", str(cache / "hub"))
            os.environ.setdefault("TRANSFORMERS_CACHE", str(cache / "hub"))

    # Step 2: load .env from the repo root so HF_TOKEN (+ any REDCAP_* vars)
    # are available in pytest, `argo env`, and sbatch scripts that import
    # argo_deepmsi without running their own dotenv logic. Existing
    # environment variables always win — we only fill in what's missing.
    if in_checkout:
        env_file = repo_root / ".env"
        if env_file.exists():
            try:
                from dotenv import load_dotenv

                load_dotenv(env_file, override=False)
            except ImportError:  # python-dotenv is a core dep, shouldn't happen
                pass


_configure_hf_env()
del _configure_hf_env


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
