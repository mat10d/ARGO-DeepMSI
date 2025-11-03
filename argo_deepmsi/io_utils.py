"""
I/O utilities for ARGO-DeepMSI pipeline.

Provides logging setup, path management, and file operations.
"""

import os
import logging
from pathlib import Path
from typing import Union, Optional
from datetime import datetime


def get_project_root() -> Path:
    """Get the absolute path to the project root directory.

    Returns:
        Path object pointing to ARGO-DeepMSI root directory.
    """
    # This file is in argo_deepmsi/, so parent is the project root
    return Path(__file__).parent.parent


def setup_logging(
    name: str,
    log_dir: Optional[Union[str, Path]] = None,
    level: int = logging.INFO,
    console: bool = True,
    file_logging: bool = True
) -> logging.Logger:
    """Set up logger with both file and console handlers.

    Args:
        name: Name of the logger (typically script or module name)
        log_dir: Directory to save log files. If None, uses logs/{name}/
        level: Logging level (default: INFO)
        console: Whether to log to console
        file_logging: Whether to log to file

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if file_logging:
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

        logger.info(f"Logging to file: {log_file}")

    return logger


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if it doesn't.

    Args:
        path: Path to directory

    Returns:
        Path object to the directory.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_data_dir(subdirectory: Optional[str] = None) -> Path:
    """Get path to data directory (INPUT DATA ONLY - raw files, metadata).

    Args:
        subdirectory: Optional subdirectory ('raw', 'metadata')

    Returns:
        Path to data directory.
    """
    data_dir = get_project_root() / "data"
    if subdirectory:
        return data_dir / subdirectory
    return data_dir


def get_raw_data_dir(site: Optional[str] = None) -> Path:
    """Get path to raw WSI files.

    Args:
        site: Optional site name (OAUTHC, LUTH, etc.)

    Returns:
        Path to data/raw/ or data/raw/{site}/
    """
    raw_dir = get_data_dir("raw")
    if site:
        return raw_dir / site
    return raw_dir


def get_metadata_dir() -> Path:
    """Get path to metadata directory (Halo Link CSVs, etc.).

    Returns:
        Path to data/metadata/
    """
    return get_data_dir("metadata")


def get_results_dir(subdir: Optional[str] = None) -> Path:
    """Get path to results directory (ALL PIPELINE OUTPUTS).

    Args:
        subdir: Optional subdirectory (stage1_data_ingestion, stage4_feature_validation, etc.)

    Returns:
        Path to results directory.
    """
    results_dir = get_project_root() / "results"
    if subdir:
        return results_dir / subdir
    return results_dir


def get_stage_dir(stage: Union[int, str]) -> Path:
    """Get path to stage-specific results directory.

    Args:
        stage: Stage number (1-8) or name ('data_ingestion', 'feature_validation', etc.)

    Returns:
        Path to results/stage{N}_{name}/

    Examples:
        get_stage_dir(1) → results/stage1_data_ingestion/
        get_stage_dir('feature_validation') → results/stage4_feature_validation/
    """
    stage_names = {
        1: "stage1_data_ingestion",
        2: "stage2_qc",
        3: "stage3_features",
        4: "stage4_feature_validation",
        5: "stage5_baseline",
        6: "stage6_training",
        7: "stage7_statistics",
        8: "stage8_visualization",
        "data_ingestion": "stage1_data_ingestion",
        "qc": "stage2_qc",
        "features": "stage3_features",
        "feature_validation": "stage4_feature_validation",
        "baseline": "stage5_baseline",
        "training": "stage6_training",
        "statistics": "stage7_statistics",
        "visualization": "stage8_visualization",
    }

    if stage in stage_names:
        return get_results_dir(stage_names[stage])
    else:
        # Assume it's already a stage name like "stage1_data_ingestion"
        return get_results_dir(str(stage))


def get_features_dir(model: Optional[str] = None, site: Optional[str] = None) -> Path:
    """Get path to extracted features directory.

    Features are stored in results/stage3_features/{model}/{site}/

    Args:
        model: Model name (ctranspath, virchow2, etc.)
        site: Site name (OAUTHC, LUTH, etc.)

    Returns:
        Path to features directory.

    Examples:
        get_features_dir() → results/stage3_features/
        get_features_dir('ctranspath') → results/stage3_features/ctranspath/
        get_features_dir('ctranspath', 'OAUTHC') → results/stage3_features/ctranspath/OAUTHC/
    """
    features_dir = get_stage_dir(3)
    if model:
        features_dir = features_dir / model
    if site:
        features_dir = features_dir / site
    return features_dir


def get_training_dir(model: Optional[str] = None) -> Path:
    """Get path to training outputs directory.

    Args:
        model: Model name (ctranspath, virchow2, etc.)

    Returns:
        Path to results/stage6_training/{model}/
    """
    training_dir = get_stage_dir(6)
    if model:
        return training_dir / model
    return training_dir


# Deprecated functions (for backward compatibility)
def get_tables_dir(stage: Optional[Union[int, str]] = None) -> Path:
    """[DEPRECATED] Get path to tables directory.

    Use get_stage_dir() instead:
    - tables/0/ → results/stage1_data_ingestion/
    - tables/2/ → results/stage4_feature_validation/tables/

    Args:
        stage: Optional stage number (0, 2, etc.)

    Returns:
        Path to tables directory.
    """
    import warnings
    warnings.warn(
        "get_tables_dir() is deprecated. Use get_stage_dir() instead:\n"
        "  tables/0/ → results/stage1_data_ingestion/\n"
        "  tables/2/ → results/stage4_feature_validation/tables/",
        DeprecationWarning,
        stacklevel=2
    )
    tables_dir = get_project_root() / "tables"
    if stage is not None:
        return tables_dir / str(stage)
    return tables_dir


def get_configs_dir(model: Optional[str] = None) -> Path:
    """Get path to configs directory.

    Args:
        model: Optional model name (ctranspath, h-optimus-0, etc.)

    Returns:
        Path to configs directory.
    """
    configs_dir = get_project_root() / "configs"
    if model:
        return configs_dir / model
    return configs_dir


def validate_file_exists(path: Union[str, Path], name: str = "File") -> Path:
    """Validate that a file exists.

    Args:
        path: Path to file
        name: Name of the file for error message

    Returns:
        Path object if file exists

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    return path


def validate_dir_exists(path: Union[str, Path], name: str = "Directory") -> Path:
    """Validate that a directory exists.

    Args:
        path: Path to directory
        name: Name of the directory for error message

    Returns:
        Path object if directory exists

    Raises:
        NotADirectoryError: If directory doesn't exist
    """
    path = Path(path)
    if not path.is_dir():
        raise NotADirectoryError(f"{name} not found: {path}")
    return path
