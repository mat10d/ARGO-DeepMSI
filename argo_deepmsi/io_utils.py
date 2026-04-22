"""
I/O utilities for ARGO-DeepMSI pipeline.

Simplified path management and file operations.
"""

import os
from pathlib import Path
from typing import Union, Optional


def get_project_root() -> Path:
    """Get the absolute path to the project root directory."""
    return Path(__file__).parent.parent


def setup_huggingface_cache() -> Path:
    """Set up HuggingFace cache directory.

    If HF_HOME is not set, defaults to .huggingface_cache in project root.
    Creates the directory if it doesn't exist.

    Returns:
        Path to HuggingFace cache directory
    """
    if "HF_HOME" not in os.environ:
        cache_dir = get_project_root() / ".huggingface_cache"
        os.environ["HF_HOME"] = str(cache_dir)
    else:
        cache_dir = Path(os.environ["HF_HOME"])

    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


# Set up HuggingFace cache on module import
setup_huggingface_cache()


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if it doesn't."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ============================================================================
# Path getters - simplified structure
# ============================================================================


def get_data_dir() -> Path:
    """Get path to data directory (raw slides, metadata)."""
    return get_project_root() / "data"


def get_results_dir() -> Path:
    """Get path to results directory (all outputs)."""
    return get_project_root() / "results"


def get_features_dir(model: Optional[str] = None) -> Path:
    """Get path to extracted features directory.

    Args:
        model: Model name (uni2, virchow2, etc.)

    Returns:
        Path to results/features/ or results/features/{model}/
    """
    features_dir = get_results_dir() / "features"
    if model:
        features_dir = features_dir / model
    return features_dir


def get_embeddings_dir(model: Optional[str] = None) -> Path:
    """Get path to slide-level embeddings directory.

    Args:
        model: Model name (uni2, virchow2, etc.)

    Returns:
        Path to results/embeddings/ or results/embeddings/{model}/
    """
    embeddings_dir = get_results_dir() / "embeddings"
    if model:
        embeddings_dir = embeddings_dir / model
    return embeddings_dir


def get_visualizations_dir() -> Path:
    """Get path to visualizations directory."""
    return get_results_dir() / "visualizations"


def get_models_dir() -> Path:
    """Get path to trained models directory."""
    return get_results_dir() / "models"
