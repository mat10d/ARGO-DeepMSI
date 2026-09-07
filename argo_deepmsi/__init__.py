"""ARGO-DeepMSI: MSI prediction from whole slide images using LazySlide.

A simplified pipeline for:
- Data ingestion from REDCap
- Feature extraction with foundation models (UNI, Virchow, etc.)
- Visualization and exploration
- Training lightweight classifiers

Usage:
    pip install -e .
    argo --help
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

from ._environment import configure_environment

__version__ = "0.2.0"

# Cache variables must be set before libraries such as transformers are
# imported. This performs no authentication, network access, or directory
# creation; those actions belong to the command that needs them.
configure_environment()

_LAZY_MODULES = {
    "data_ingestion",
    "feature_extraction",
    "io_utils",
    "training",
    "visualization",
}

__all__ = [
    "__version__",
    "io_utils",
    "data_ingestion",
    "feature_extraction",
    "visualization",
    "training",
]


def __getattr__(name: str) -> ModuleType:
    """Load public submodules only when they are first accessed."""
    if name not in _LAZY_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f".{name}", __name__)
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LAZY_MODULES)
