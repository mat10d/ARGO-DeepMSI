"""Correctness of the screening / operating-point metrics (E00)."""

import numpy as np
import pytest

from argo_deepmsi.eval.screening import (
    npv_at_threshold,
    screening_block,
    specificity_at_sensitivity,
)


def test_perfect_separation():
    # positives strictly above negatives -> spec 1.0 at any sens floor
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    d = specificity_at_sensitivity(y, s, 0.95)
    assert d["sensitivity"] >= 0.95
    assert d["specificity"] == 1.0
    assert d["npv"] == 1.0


def test_sensitivity_floor_is_met():
    rng = np.random.default_rng(42)
    y = (rng.random(500) < 0.2).astype(int)
    s = rng.random(500) + 0.3 * y  # weak signal
    for floor in (0.90, 0.95, 0.98):
        d = specificity_at_sensitivity(y, s, floor)
        # achieved sensitivity must meet or exceed the floor (allow tiny tie slack)
        assert d["sensitivity"] >= floor - 1e-9, (floor, d["sensitivity"])


def test_specificity_monotone_in_sensitivity():
    # higher sensitivity floor => lower (or equal) specificity
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.25).astype(int)
    s = rng.random(400) + 0.5 * y
    spec90 = specificity_at_sensitivity(y, s, 0.90)["specificity"]
    spec98 = specificity_at_sensitivity(y, s, 0.98)["specificity"]
    assert spec98 <= spec90 + 1e-9


def test_npv_definition():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    # threshold 0.5: predicted-neg = {0.1,0.2} both true neg -> npv 1.0
    assert npv_at_threshold(y, s, 0.5) == 1.0


def test_block_has_flat_keys():
    y = np.array([0, 1, 0, 1, 0, 1, 0, 0])
    s = np.array([0.2, 0.6, 0.1, 0.9, 0.3, 0.7, 0.15, 0.25])
    b = screening_block(y, s)
    for k in ("spec_at_sens90", "spec_at_sens95", "npv_at_sens95", "op_sens96"):
        assert k in b


def test_rejects_nonbinary():
    with pytest.raises(ValueError):
        specificity_at_sensitivity([0, 1, 2], [0.1, 0.2, 0.3], 0.9)
