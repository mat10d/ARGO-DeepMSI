"""
ARGO-DeepMSI: Multi-model MSI prediction pipeline for whole slide images.

This package provides a complete pipeline for MSI prediction from H&E-stained
colorectal cancer whole slide images, including:
- Data ingestion from REDCap and Halo Link
- Quality control and feature validation
- Feature extraction using multiple foundation models (via STAMP)
- Baseline testing with pre-trained models (HistoBistro)
- MIL training with k-fold cross-validation
- Statistics and visualization
"""

__version__ = "0.1.0"
__author__ = "ARGO-DeepMSI Team"

# Import key modules for convenient access
from . import io_utils
from . import data_ingestion
from . import feature_validation
from . import visualization

__all__ = [
    "io_utils",
    "data_ingestion",
    "feature_validation",
    "visualization",
    # Future modules:
    # "quality_control",
    # "feature_extraction",
    # "baseline_testing",
    # "training",
    # "statistics",
    # "config_utils",
]
