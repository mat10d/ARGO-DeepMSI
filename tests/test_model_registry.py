"""Validate that ``argo_deepmsi``'s model catalog stays in sync with LazySlide.

Three layers, ordered by cost:

1. **Static consistency** (always runs, no network): every name we expose in
   ``PATCH_MODELS``/``QC_MODELS``/``SLIDE_ENCODERS`` exists in LazySlide's
   ``MODEL_REGISTRY``, and the ``requires_auth`` flag matches
   ``is_gated`` upstream. Catches typos and LazySlide-side renames.
2. **Weight download** (opt-in, ``-m model_download``): instantiates each
   non-gated patch model. Triggers HF download on first run. Does NOT run
   inference. Catches broken weights / deleted HF repos without the cost of
   a real slide-level extraction.
3. **Gated weight download** (opt-in, ``-m requires_hf_token``): same as (2)
   but for gated models; auto-skips if ``HF_TOKEN`` isn't set.

Run:
    pytest tests/test_model_registry.py -v                    # layer 1 only
    pytest tests/test_model_registry.py -v -m model_download  # layer 1 + 2
    pytest tests/test_model_registry.py -v -m "model_download or requires_hf_token"
"""

from __future__ import annotations

import os

import pytest

from argo_deepmsi.feature_extraction import (
    PATCH_MODELS,
    QC_MODELS,
    SLIDE_ENCODERS,
)
from argo_deepmsi.models._lazyslide import MODEL_REGISTRY

REGISTRY = MODEL_REGISTRY


def _has_task(model_class, name: str) -> bool:
    tasks = getattr(model_class, "task", ())
    tasks = tasks if isinstance(tasks, (list, tuple, set)) else (tasks,)
    return any(str(task).rsplit(".", 1)[-1] == name for task in tasks)


# --------------------------------------------------------------------------
# Layer 1 — static consistency against LazySlide's registry
# --------------------------------------------------------------------------


class TestRegistryConsistency:
    def test_every_patch_model_exists_in_lazyslide_registry(self):
        missing = sorted(n for n in PATCH_MODELS if n not in REGISTRY)
        assert not missing, (
            f"PATCH_MODELS names not in LazySlide MODEL_REGISTRY: {missing}. "
            "Either LazySlide renamed/removed the model, or the entry in "
            "feature_extraction.PATCH_MODELS is a typo."
        )

    def test_gating_matches_lazyslide(self):
        mismatches = []
        for name, cfg in PATCH_MODELS.items():
            if name not in REGISTRY:
                continue
            upstream = bool(getattr(REGISTRY[name], "is_gated", False))
            if cfg.requires_auth != upstream:
                mismatches.append(
                    f"{name}: argo.requires_auth={cfg.requires_auth}, "
                    f"lazyslide.is_gated={upstream}"
                )
        assert not mismatches, "requires_auth disagrees with lazyslide.is_gated:\n  " + "\n  ".join(
            mismatches
        )

    def test_qc_models_have_qc_appropriate_task(self):
        """LazySlide QC models are tagged segmentation / tile_prediction —
        never generic 'vision'."""
        allowed = {"ModelTask.segmentation", "ModelTask.tile_prediction"}
        bad = []
        for name in QC_MODELS:
            if name not in REGISTRY:
                continue
            task = str(getattr(REGISTRY[name], "task", ""))
            if task not in allowed:
                bad.append(f"{name}: task={task}")
        assert not bad, (
            f"QC models with unexpected task types: {bad}. " f"Expected one of {sorted(allowed)}."
        )

    def test_neural_slide_encoders_exist_in_registry(self):
        """Statistical methods (mean/max/median/sum) are not in the registry
        by design; the neural encoders must be."""
        statistical = {"mean", "max", "median", "sum"}
        neural = set(SLIDE_ENCODERS) - statistical
        missing = sorted(neural - set(REGISTRY))
        assert not missing, f"Neural slide encoders missing from registry: {missing}"
        wrong_task = sorted(
            name for name in neural if not _has_task(REGISTRY[name], "slide_encoder")
        )
        assert not wrong_task, f"Non-slide models exposed as neural encoders: {wrong_task}"

    def test_statistical_methods_not_treated_as_models(self):
        """mean/max/median/sum should never leak into PATCH_MODELS."""
        leaked = sorted(set(PATCH_MODELS) & {"mean", "max", "median", "sum"})
        assert not leaked, f"Statistical methods leaked into PATCH_MODELS: {leaked}"

    def test_patch_models_implement_image_encoding(self):
        invalid = sorted(
            name
            for name in PATCH_MODELS
            if not callable(getattr(REGISTRY[name], "encode_image", None))
        )
        assert not invalid, f"Non-image models exposed as patch extractors: {invalid}"

    def test_vision_coverage_is_intentional(self):
        """Every upstream image encoder is automatically CLI-addressable."""
        upstream_patch = {
            name
            for name, model_class in REGISTRY.items()
            if callable(getattr(model_class, "encode_image", None))
            and (_has_task(model_class, "vision") or _has_task(model_class, "multimodal"))
        }
        exposed_classes = {REGISTRY[name] for name in PATCH_MODELS}
        new = sorted(name for name in upstream_patch if REGISTRY[name] not in exposed_classes)
        assert not new, (
            f"LazySlide has new vision models that are not in PATCH_MODELS "
            f"or the known-excluded list: {new}. Add them to PATCH_MODELS "
            f"(preferred) or append to known_excluded in this test."
        )


class TestWaivModels:
    """The Waiv robust encoders (Phaet, Mascaret) are project-local additions
    to the registry — not upstream LazySlide. Lock their contract statically
    (no network / no weight download)."""

    EXPECTED = {"phaet": 1024, "mascaret": 1536}

    def test_registered_in_lazyslide_registry(self):
        missing = sorted(n for n in self.EXPECTED if n not in REGISTRY)
        assert not missing, (
            f"Waiv encoders not registered: {missing}. Importing "
            "argo_deepmsi.models must run the @register decorators."
        )

    def test_encode_dim_and_task(self):
        for name, dim in self.EXPECTED.items():
            cls = REGISTRY[name]
            assert getattr(cls, "encode_dim", None) == dim, f"{name} encode_dim"
            assert str(getattr(cls, "task", "")) == "ModelTask.vision", f"{name} task"

    def test_exposed_as_gated_patch_models(self):
        for name in self.EXPECTED:
            assert name in PATCH_MODELS, f"{name} missing from PATCH_MODELS"
            assert PATCH_MODELS[name].requires_auth is True, f"{name} must be gated"


# --------------------------------------------------------------------------
# Layer 2 / 3 — instantiate weights (no inference)
# --------------------------------------------------------------------------

_NON_GATED_PATCH = sorted(
    n for n, cfg in PATCH_MODELS.items() if not cfg.requires_auth and n in REGISTRY
)
_GATED_PATCH = sorted(n for n, cfg in PATCH_MODELS.items() if cfg.requires_auth and n in REGISTRY)


def _instantiate(name: str) -> None:
    """Import the model class — triggers HF download of weights on first
    call, but does NOT run inference. On subsequent runs this is a fast
    cache hit."""
    cls = REGISTRY[name]
    # All ImageModel subclasses in LazySlide construct lazily; constructing
    # is enough to trigger the weight download.
    model = cls()
    # Smoke: the class reports a finite feature dim (for vision models) or
    # a callable encoder. This catches malformed entries without inference.
    if hasattr(cls, "encode_dim") and cls.encode_dim not in (None, 0):
        assert cls.encode_dim > 0
    del model


@pytest.mark.model_download
@pytest.mark.parametrize("name", _NON_GATED_PATCH)
def test_non_gated_model_downloads(name: str):
    """Each non-gated patch model instantiates cleanly from HuggingFace.

    Opt-in via ``-m model_download`` — first run may be slow and downloads
    weights into ``$HF_HOME``.
    """
    _instantiate(name)


@pytest.mark.requires_hf_token
@pytest.mark.parametrize("name", _GATED_PATCH)
def test_gated_model_downloads(name: str):
    """Each gated patch model instantiates when HF_TOKEN is set in env."""
    if not os.environ.get("HF_TOKEN"):
        pytest.skip("HF_TOKEN not set; skipping gated-model download")
    _instantiate(name)
