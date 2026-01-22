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
