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


def get_data_dir(site: Optional[str] = None) -> Path:
    """Get path to data directory.

    Args:
        site: Optional site name (OAUTHC, LUTH, etc.). If None, returns base data dir.

    Returns:
        Path to data directory.
    """
    data_dir = get_project_root() / "data"
    if site:
        return data_dir / site
    return data_dir


def get_tables_dir(stage: Optional[Union[int, str]] = None) -> Path:
    """Get path to tables directory.

    Args:
        stage: Optional stage number (0, 2, etc.). If None, returns base tables dir.

    Returns:
        Path to tables directory.
    """
    tables_dir = get_project_root() / "tables"
    if stage is not None:
        return tables_dir / str(stage)
    return tables_dir


def get_results_dir(subdir: Optional[str] = None) -> Path:
    """Get path to results directory.

    Args:
        subdir: Optional subdirectory (qc, feature_validation, statistics, etc.)

    Returns:
        Path to results directory.
    """
    results_dir = get_project_root() / "results"
    if subdir:
        return results_dir / subdir
    return results_dir


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
