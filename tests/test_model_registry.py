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

import lazyslide as zs

from argo_deepmsi.feature_extraction import (
    PATCH_MODELS,
    QC_MODELS,
    SLIDE_ENCODERS,
)


REGISTRY = zs.models.MODEL_REGISTRY


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
        assert not mismatches, (
            "requires_auth disagrees with lazyslide.is_gated:\n  "
            + "\n  ".join(mismatches)
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
            f"QC models with unexpected task types: {bad}. "
            f"Expected one of {sorted(allowed)}."
        )

    def test_neural_slide_encoders_exist_in_registry(self):
        """Statistical methods (mean/max/median/sum) are not in the registry
        by design; the neural encoders must be."""
        neural = {"prism", "titan", "chief", "madeleine", "gigapath-slide-encoder"}
        missing = sorted(n for n in neural if n in SLIDE_ENCODERS and n not in REGISTRY)
        assert not missing, f"Neural slide encoders missing from registry: {missing}"

    def test_statistical_methods_not_treated_as_models(self):
        """mean/max/median/sum should never leak into PATCH_MODELS."""
        leaked = sorted(set(PATCH_MODELS) & {"mean", "max", "median", "sum"})
        assert not leaked, f"Statistical methods leaked into PATCH_MODELS: {leaked}"

    def test_vision_coverage_is_intentional(self):
        """If LazySlide adds a new vision model upstream, catch it so we can
        decide whether to expose it. Currently allow-listed: midnight (larger
        and we haven't evaluated it yet)."""
        upstream_vision = {
            n for n, cls in REGISTRY.items()
            if str(getattr(cls, "task", "")) == "ModelTask.vision"
        }
        known_excluded = {"midnight"}
        new = sorted(upstream_vision - set(PATCH_MODELS) - known_excluded)
        assert not new, (
            f"LazySlide has new vision models that are not in PATCH_MODELS "
            f"or the known-excluded list: {new}. Add them to PATCH_MODELS "
            f"(preferred) or append to known_excluded in this test."
        )


# --------------------------------------------------------------------------
# Layer 2 / 3 — instantiate weights (no inference)
# --------------------------------------------------------------------------

_NON_GATED_PATCH = sorted(
    n for n, cfg in PATCH_MODELS.items()
    if not cfg.requires_auth and n in REGISTRY
)
_GATED_PATCH = sorted(
    n for n, cfg in PATCH_MODELS.items()
    if cfg.requires_auth and n in REGISTRY
)


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
