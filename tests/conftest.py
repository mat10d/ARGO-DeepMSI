"""Shared fixtures for argo_deepmsi test modules."""

from __future__ import annotations

# Import the package first so its HF_HOME default is applied BEFORE any test
# module, fixture, or downstream dep pulls in huggingface_hub / transformers.
# Without this, model-download tests could write to ~/.cache/huggingface
# instead of the repo-local .huggingface_cache.
import argo_deepmsi  # noqa: F401

import importlib.util
import shutil
import tempfile
from pathlib import Path

import pytest

# Opt-in marks: skip by default unless the user asks for them explicitly via
# `-m <mark>` or a boolean expression that mentions the mark.
_OPT_IN_MARKS = (
    "gpu",
    "integration",
    "model_download",
    "network",
    "requires_hf_token",
)


# The LazySlide stack lives in envs/lazyslide, not the core env. Tests marked
# `lazyslide` skip cleanly when it is not importable.
LAZYSLIDE_AVAILABLE = all(
    importlib.util.find_spec(name) is not None
    for name in ("lazyslide", "lazyslide_models", "wsidata")
)


def pytest_collection_modifyitems(config, items):
    selected = config.getoption("-m") or ""
    skip_lazyslide = pytest.mark.skip(
        reason="LazySlide stack not installed: run in envs/lazyslide "
        "(`uv run --frozen --project envs/lazyslide pytest`)"
    )
    for item in items:
        if "lazyslide" in item.keywords and not LAZYSLIDE_AVAILABLE:
            item.add_marker(skip_lazyslide)
            continue
        for mark in _OPT_IN_MARKS:
            if mark in item.keywords and mark not in selected:
                item.add_marker(pytest.mark.skip(reason=f"opt-in: run with `-m {mark}`"))
                break


@pytest.fixture(scope="session")
def data_dir() -> Path:
    d = Path(tempfile.mkdtemp(prefix="argo_pipe_", dir="."))
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(scope="session")
def sample_slide(data_dir: Path) -> Path:
    """Small GTEx artery slide from LazySlide's public dataset (~20Kx20K, ~253 tiles)."""
    from huggingface_hub import hf_hub_download

    slide_path = hf_hub_download(
        "rendeirolab/lazyslide-data",
        "GTEX-1117F-0526.svs",
        repo_type="dataset",
        cache_dir=str(data_dir),
        token=False,
    )
    return Path(slide_path)


@pytest.fixture(scope="session")
def has_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False
