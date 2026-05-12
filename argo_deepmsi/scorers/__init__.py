"""MSI scorers — uniform interface across zero-shot, head-on-our-data,
and aggregator-fusion approaches."""

from .base import Scorer, ScoreColumn
from .registry import get_scorer, list_scorers, register

# Import each module so its register() call runs on package import.
from . import calibrated_pool  # noqa: F401
from . import nuclear_morphology  # noqa: F401
from . import score_fusion  # noqa: F401
from . import simple_grid  # noqa: F401
from . import slide_attention_mil  # noqa: F401
from . import transductive_smoothing  # noqa: F401
from . import vl_text_cosine  # noqa: F401
from . import wagner_zeroshot  # noqa: F401

__all__ = ["Scorer", "ScoreColumn", "get_scorer", "list_scorers", "register"]
