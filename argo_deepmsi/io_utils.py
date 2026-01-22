"""
I/O utilities for ARGO-DeepMSI pipeline.

Simplified path management and file operations.
"""

import logging
from pathlib import Path
from typing import Union, Optional
from datetime import datetime


def get_project_root() -> Path:
    """Get the absolute path to the project root directory."""
    return Path(__file__).parent.parent


def setup_logging(
    name: str,
    log_dir: Optional[Union[str, Path]] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Set up logger with file and console handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_dir is None:
        log_dir = get_project_root() / "logs" / name
    else:
        log_dir = Path(log_dir)

    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{name}_{timestamp}.log"

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


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
