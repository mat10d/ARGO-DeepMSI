"""Enforce the NO-NEW-TRAINING-DATA regime.

This is the guard that makes the regime real rather than aspirational. It is a
STATIC check: every scorer source file is scanned for references to external
slide/label sources. Fitting a head on our own cohort is allowed; loading TCGA
or any external pretraining slide set is not.

Frozen pretrained *feature-extractor weights* (UNI/CONCH/Virchow/TITAN/PRISM) are
permitted — they are not training on external labels. The distinction we enforce:
no external *slide tables / label sets / raw-slide directories* are read by a
scorer.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SCORERS_DIR = Path(__file__).resolve().parents[1] / "argo_deepmsi" / "scorers"

# Substrings that indicate reading an external labelled slide set into a scorer.
# Case-insensitive. Keep this list conservative and explicit.
FORBIDDEN_PATTERNS = [
    r"\btcga\b",
    r"\bcptac\b",
    r"\bpaip\b",
    r"\bcamelyon\b",
    r"\bnct[-_ ]?crc\b",
    r"/external/",
    r"external_slides",
    r"pretrain.*\.csv",  # loading an external pretraining label table
    r"download.*slides",
]

# Files that are allowed to mention these terms in *comments only* would still
# trip this; that is intentional — keep external-dataset names out of scorer
# source entirely (put such notes in docs/).
ALLOW_FILES = {"__init__.py", "base.py", "registry.py"}


def _scorer_files() -> list[Path]:
    if not SCORERS_DIR.exists():
        pytest.skip(f"scorers dir not found at {SCORERS_DIR}")
    return [p for p in SCORERS_DIR.glob("*.py") if p.name not in ALLOW_FILES]


@pytest.mark.parametrize("pattern", FORBIDDEN_PATTERNS)
def test_no_forbidden_external_source(pattern):
    rx = re.compile(pattern, re.IGNORECASE)
    hits = []
    for f in _scorer_files():
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if rx.search(line):
                hits.append(f"{f.name}:{i}: {line.strip()}")
    assert not hits, (
        f"External-data reference matching /{pattern}/ found in scorer source:\n"
        + "\n".join(hits)
        + "\n\nThe NO-NEW-TRAINING-DATA regime forbids loading external labelled "
        "slide sets in a scorer. Frozen FM weights are fine; external slide "
        "TABLES/LABELS are not."
    )


def test_fit_only_sees_our_cohort_patients():
    """Runtime guard stub: documents the contract that fit(train_df) must only
    ever receive patient_ids present in results/data/cohort_clean.csv.

    The full runtime assertion lives in the scorer contract test (which builds a
    tiny cohort fixture and asserts fit() never widens the patient set). Here we
    only assert the canonical cohort file is the single declared training source.
    """
    canonical = Path(__file__).resolve().parents[1] / "results" / "data" / "cohort_clean.csv"
    # Not required to exist yet (Q-phase builds it); if present, it must carry the
    # in_clean_set flag that defines the permitted training rows.
    if canonical.exists():
        header = canonical.read_text().splitlines()[0]
        assert "patient_id" in header and "in_clean_set" in header
