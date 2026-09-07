"""Configure environment defaults before optional ML dependencies are imported.

This module intentionally avoids importing Hugging Face, PyTorch, or LazySlide.
Package import may set environment variables, but it must not perform network or
filesystem writes.
"""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """Return the configured workspace, source checkout, or current directory."""
    configured = os.environ.get("ARGO_WORKSPACE")
    if configured:
        return Path(configured).expanduser().resolve()
    package_path = Path(__file__).resolve()
    for parent in package_path.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "argo_deepmsi").is_dir():
            return parent
    return Path.cwd().resolve()


def configure_environment() -> Path | None:
    """Load checkout-local settings and configure Hugging Face cache paths.

    Explicit environment variables take precedence over values from ``.env``;
    values from ``.env`` take precedence over the checkout-local default. No
    directories are created here, and authentication remains the responsibility
    of the command that actually needs it.
    """
    root = project_root()
    in_checkout = (root / "pyproject.toml").is_file()

    if in_checkout:
        env_file = root / ".env"
        if env_file.is_file():
            try:
                from dotenv import load_dotenv
            except ImportError:  # pragma: no cover - declared core dependency
                pass
            else:
                load_dotenv(env_file, override=False)

    hf_home = os.environ.get("HF_HOME")
    if not hf_home and in_checkout:
        hf_home = str(root / ".huggingface_cache")
        os.environ["HF_HOME"] = hf_home

    if not hf_home:
        cache_dir = None
    else:
        cache_dir = Path(hf_home).expanduser()
        os.environ.setdefault("HF_HUB_CACHE", str(cache_dir / "hub"))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir / "hub"))

    if in_checkout:
        local_cache = root / ".cache"
        os.environ.setdefault("MPLCONFIGDIR", str(local_cache / "matplotlib"))
        os.environ.setdefault("NUMBA_CACHE_DIR", str(local_cache / "numba"))
    return cache_dir
