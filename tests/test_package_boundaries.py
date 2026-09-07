"""Tests for lightweight package and scorer imports."""

from __future__ import annotations

import subprocess
import sys


def _run_isolated(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )


def test_package_import_does_not_load_ml_stacks():
    result = _run_isolated(
        "import sys; import argo_deepmsi; "
        "assert 'torch' not in sys.modules; "
        "assert 'lazyslide' not in sys.modules; "
        "assert 'huggingface_hub' not in sys.modules"
    )
    assert result.returncode == 0, result.stderr


def test_scorer_catalog_is_lazy_and_targeted():
    result = _run_isolated(
        "import sys; import argo_deepmsi.scorers as scorers; "
        "assert 'argo_deepmsi.scorers.clam_tilemil' not in sys.modules; "
        "assert 'wagner_zeroshot' in scorers.list_scorers(); "
        "scorers.get_scorer('wagner_zeroshot'); "
        "assert 'argo_deepmsi.scorers.wagner_zeroshot' in sys.modules; "
        "assert 'argo_deepmsi.scorers.clam_tilemil' not in sys.modules"
    )
    assert result.returncode == 0, result.stderr
